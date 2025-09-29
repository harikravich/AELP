import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../lib/dataset'
import { createBigQueryClient } from '../../../../../lib/bigquery-client'

export const dynamic = 'force-dynamic'

function classify(name: string | null): {brand: boolean} {
  // Brand only if explicit token 'Brand' or known naming pattern (avoid matching 'Aura')
  const s = String(name||'')
  const brand = /(^|[\s_\-])Brand([\s_\-]|$)/i.test(s) || /Search_Brand/i.test(s)
  return { brand }
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const days = Math.min(90, Math.max(1, Number(url.searchParams.get('days') || '14')))
    const group = (url.searchParams.get('group') || 'campaign').toLowerCase() as 'campaign'|'daily'
    const brandFilter = (url.searchParams.get('brand') || 'all').toLowerCase() as 'all'|'brand'|'nonbrand'
    const channelFilter = (url.searchParams.get('channel') || 'all').toUpperCase() as 'ALL'|'SEARCH'
    const minSpend = Number(url.searchParams.get('min_spend') || '0')
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)
    const excludeToday = (url.searchParams.get('excl_today') || '1') !== '0'
    const dateFilter = excludeToday
      ? `DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)`
      : `DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()`
    const [rows] = await bq.query({ query: `
      SELECT 
        DATE(date) AS date,
        campaign_id,
        ANY_VALUE(campaign_name) AS name,
        SUM(impressions) AS impressions,
        SUM(clicks) AS clicks,
        SUM(cost_micros)/1e6 AS spend,
        SUM(conversions) AS conversions,
        ANY_VALUE(advertising_channel_type) AS channel_type,
        IF(ANY_VALUE(advertising_channel_type)='SEARCH', AVG(impression_share), NULL) AS impression_share,
        IF(ANY_VALUE(advertising_channel_type)='SEARCH', AVG(lost_is_rank), NULL) AS lost_is_rank,
        IF(ANY_VALUE(advertising_channel_type)='SEARCH', AVG(lost_is_budget), NULL) AS lost_is_budget
      FROM \`${projectId}.${dataset}.ads_campaign_performance\`
      WHERE ${dateFilter}
      GROUP BY date, campaign_id
      ORDER BY date DESC, spend DESC
    `, params: { days } })

    // Daily rows with computed metrics and brand flag
    const daily = (rows as any[]).map(r => {
      const ctr = r.impressions ? r.clicks / r.impressions : 0
      const cvr = r.clicks ? r.conversions / r.clicks : 0
      const cac = r.conversions ? r.spend / r.conversions : null
      const { brand } = classify(r.name)
      // Alerts
      const alerts: any[] = []
      if (r.lost_is_rank != null && r.lost_is_rank > 0.6) alerts.push({type: 'rank_limited', severity: 'high'})
      if (brand && r.impression_share != null && r.impression_share < 0.8) alerts.push({type: 'brand_is_low', severity: 'high'})
      if (!brand && r.impression_share != null && r.impression_share < 0.2) alerts.push({type: 'is_low', severity: 'med'})
      if (ctr < 0.02) alerts.push({type: 'ctr_low', severity: 'med'})
      // Suggestions
      const suggestions: string[] = []
      if (alerts.find(a=>a.type==='rank_limited')) suggestions.push('Improve Ad Rank: raise bids 5–10% on top STAGs, tighten RSAs, split intents, improve LP relevance')
      if (alerts.find(a=>a.type==='brand_is_low')) suggestions.push('Defend brand: exact brand ad groups, pin brand headlines, raise bids to hit IS ≥ 90%')
      if (alerts.find(a=>a.type==='is_low') && !brand) suggestions.push('Expand coverage: add exact/phrase for top SQRs, add negatives to protect intent')
      if (alerts.find(a=>a.type==='ctr_low')) suggestions.push('RSA refresh: replace Low‑rated assets, add 12–15 headlines, 4 descriptions, use DKI where safe')
      return { ...r, ctr, cvr, cac, brand, alerts, suggestions }
    })
    // Apply filters on daily rows
    let filteredDaily = daily.filter(r => {
      if (channelFilter === 'SEARCH' && r.channel_type !== 'SEARCH') return false
      if (brandFilter === 'brand' && !r.brand) return false
      if (brandFilter === 'nonbrand' && r.brand) return false
      return true
    })

    // Grouping
    let rowsOut: any[] = []
    if (group === 'daily') {
      rowsOut = filteredDaily.filter(r => r.spend >= minSpend)
    } else {
      // group by campaign over the period
      const byCamp = new Map<string, any>()
      for (const r of filteredDaily) {
        const key = String(r.campaign_id)
        const agg = byCamp.get(key) || {
          campaign_id: r.campaign_id,
          name: r.name,
          channel_type: r.channel_type,
          brand: r.brand,
          impressions: 0, clicks: 0, spend: 0, conversions: 0,
          // For IS metrics: average weighted by impressions if available
          _is_weight: 0, _is_sum: 0, _rank_sum: 0, _budget_sum: 0,
          days: 0,
        }
        agg.impressions += r.impressions
        agg.clicks += r.clicks
        agg.spend += r.spend
        agg.conversions += r.conversions
        if (r.impression_share != null && r.impressions != null) {
          agg._is_sum += r.impression_share * (r.impressions || 0)
          agg._rank_sum += (r.lost_is_rank ?? 0) * (r.impressions || 0)
          agg._budget_sum += (r.lost_is_budget ?? 0) * (r.impressions || 0)
          agg._is_weight += (r.impressions || 0)
        }
        agg.days += 1
        byCamp.set(key, agg)
      }
      rowsOut = Array.from(byCamp.values()).map(a => {
        const ctr = a.impressions ? a.clicks / a.impressions : 0
        const cvr = a.clicks ? a.conversions / a.clicks : 0
        const cac = a.conversions ? a.spend / a.conversions : null
        const impression_share = a._is_weight ? a._is_sum / a._is_weight : null
        const lost_is_rank = a._is_weight ? a._rank_sum / a._is_weight : null
        const lost_is_budget = a._is_weight ? a._budget_sum / a._is_weight : null
        const alerts: any[] = []
        if (lost_is_rank != null && lost_is_rank > 0.6) alerts.push({type: 'rank_limited', severity: 'high'})
        if (a.brand && impression_share != null && impression_share < 0.8) alerts.push({type: 'brand_is_low', severity: 'high'})
        if (!a.brand && impression_share != null && impression_share < 0.2) alerts.push({type: 'is_low', severity: 'med'})
        if (ctr < 0.02) alerts.push({type: 'ctr_low', severity: 'med'})
        const suggestions: string[] = []
        if (alerts.find(x=>x.type==='rank_limited')) suggestions.push('Improve Ad Rank: raise bids 5–10% on top STAGs, tighten RSAs, split intents, improve LP relevance')
        if (alerts.find(x=>x.type==='brand_is_low')) suggestions.push('Defend brand: exact brand ad groups, pin brand headlines, raise bids to hit IS ≥ 90%')
        if (alerts.find(x=>x.type==='is_low') && !a.brand) suggestions.push('Expand coverage: add exact/phrase for top SQRs, add negatives to protect intent')
        if (alerts.find(x=>x.type==='ctr_low')) suggestions.push('RSA refresh: replace Low‑rated assets, add 12–15 headlines, 4 descriptions, use DKI where safe')
        return { ...a, ctr, cvr, cac, impression_share, lost_is_rank, lost_is_budget, alerts, suggestions }
      }).filter(r => r.spend >= minSpend)
      // order by spend desc
      rowsOut.sort((a,b)=> b.spend - a.spend)
    }

    // Summary and issues
    const totalSpend = filteredDaily.reduce((s,r)=> s + (r.spend||0), 0)
    const totalConv = filteredDaily.reduce((s,r)=> s + (r.conversions||0), 0)
    const zeroConvSpend = filteredDaily.filter(r=> (r.conversions||0)===0).reduce((s,r)=> s + (r.spend||0), 0)
    const rankLimitedSpend = filteredDaily.filter(r=> (r.lost_is_rank ?? 0) > 0.6).reduce((s,r)=> s + (r.spend||0), 0)
    const brandRows = filteredDaily.filter(r=> r.brand)
    const brandIS = (()=>{
      const w = brandRows.reduce((s,r)=> s + (r.impressions||0), 0)
      const sum = brandRows.reduce((s,r)=> s + ((r.impression_share||0) * (r.impressions||0)), 0)
      return w ? (sum / w) : null
    })()
    const issues = {
      rank_limited: filteredDaily
        .filter(r=> (r.lost_is_rank ?? 0) > 0.6)
        .sort((a,b)=> (b.lost_is_rank||0) - (a.lost_is_rank||0))
        .slice(0, 20),
      brand_is_low: filteredDaily
        .filter(r=> r.brand && r.impression_share != null && r.impression_share < 0.8)
        .sort((a,b)=> (a.impression_share||0) - (b.impression_share||0))
        .slice(0, 20),
      low_ctr: filteredDaily
        .filter(r=> (r.impressions||0) > 100 && (r.ctr||0) < 0.02)
        .sort((a,b)=> (a.ctr||0) - (b.ctr||0))
        .slice(0, 20),
      zero_conv_spend: filteredDaily
        .filter(r=> (r.conversions||0)===0)
        .sort((a,b)=> b.spend - a.spend)
        .slice(0, 20),
    }

    const summary = {
      days,
      group,
      filters: { brand: brandFilter, channel: channelFilter, min_spend: minSpend, excl_today: excludeToday },
      total_spend: totalSpend,
      total_conversions: totalConv,
      avg_cac: totalConv ? totalSpend / totalConv : null,
      zero_conv_spend: zeroConvSpend,
      zero_conv_spend_share: totalSpend ? zeroConvSpend / totalSpend : null,
      rank_limited_spend: rankLimitedSpend,
      rank_limited_spend_share: totalSpend ? rankLimitedSpend / totalSpend : null,
      brand_is_health: brandIS,
    }

    return NextResponse.json({ group, summary, issues, rows: rowsOut })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message||String(e) }, { status: 500 })
  }
}
