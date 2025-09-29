import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

async function getAccessToken(client_id: string, client_secret: string, refresh_token: string): Promise<string> {
  const params = new URLSearchParams({ client_id, client_secret, refresh_token, grant_type: 'refresh_token' })
  const r = await fetch('https://oauth2.googleapis.com/token', { method:'POST', headers:{'content-type':'application/x-www-form-urlencoded'}, body: params.toString() })
  const j = await r.json(); if(!r.ok) throw new Error(j?.error_description||'oauth token error'); return j.access_token
}

async function lookupCustomerId(projectId: string, dataset: string, campaignId?: string|null): Promise<string|null> {
  if (!campaignId) return null
  try {
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `SELECT ANY_VALUE(customer_id) AS cid FROM \`${projectId}.${dataset}.ads_campaign_performance\` WHERE CAST(campaign_id AS STRING)=@cid`, params:{ cid: String(campaignId) } })
    return rows?.[0]?.cid ? String(rows[0].cid) : null
  } catch { return null }
}

export async function POST(req: NextRequest) {
  try {
    const { dataset, allowed, reason } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok: false, error: reason }, { status: 403 })
    const form = await req.formData().catch(()=>null)
    const run_id = String(form?.get('run_id') || '')
    if (!run_id) return NextResponse.json({ ok: false, error: 'run_id required' }, { status: 400 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const bq = new BigQuery({ projectId })

    // Load queued payload
    const [qrows] = await bq.query({ query: `SELECT platform, type, payload FROM \`${projectId}.${dataset}.creative_publish_queue\` WHERE run_id=@run LIMIT 1`, params: { run: run_id } })
    const payloadRaw = qrows?.[0]?.payload
    const payload = typeof payloadRaw === 'string' ? JSON.parse(payloadRaw) : (payloadRaw?.value ? JSON.parse(payloadRaw.value) : (payloadRaw||{}))

    const action = String(payload?.action || 'create')
    const campaign_id = payload?.campaign_id ? String(payload.campaign_id) : null
    const ad_group_id = payload?.ad_group_id ? String(payload.ad_group_id) : null
    const final_url = String(payload?.final_url || payload?.payload?.final_url || '')
    const headlines: string[] = (payload?.headlines || payload?.payload?.headlines || []).map((s:any)=> String(s)).filter(Boolean)
    const descriptions: string[] = (payload?.descriptions || payload?.payload?.descriptions || []).map((s:any)=> String(s)).filter(Boolean)
    const source_ad_id = payload?.source_ad_id ? String(payload.source_ad_id) : String(payload?.payload?.source_ad_id||'')

    // Gating for live Google Ads mutation
    const allowMutations = String(process.env.AELP2_ALLOW_GOOGLE_MUTATIONS||'0') === '1'
    const allowedCustomers = String(process.env.GOOGLE_ADS_ALLOWED_CUSTOMERS||'8448503866').split(',').map(s=> s.replace(/\D/g,''))
    const validateOnly = String(process.env.GOOGLE_ADS_VALIDATE_ONLY||'0') !== '0'

    // Resolve customer id (from campaign in BQ) and allowlist
    let customer_id = String(payload?.customer_id || '')
    if (!customer_id && allowedCustomers.length>0) customer_id = allowedCustomers[0]
    if (!customer_id) customer_id = (await lookupCustomerId(projectId, dataset, campaign_id)) || ''
    if (!customer_id) customer_id = '8448503866'
    const cust = customer_id.replace(/\D/g,'')

    let ad_id_created: string | null = null
    let status = 'processed'
    let error: string | null = null
    const platform = 'google_ads'

    if (allowMutations && cust && allowedCustomers.includes(cust) && ad_group_id) {
      try {
        const dev = process.env.GOOGLE_ADS_DEVELOPER_TOKEN!
        const cid = process.env.GOOGLE_ADS_CLIENT_ID!
        const cs = process.env.GOOGLE_ADS_CLIENT_SECRET!
        const rt = process.env.GOOGLE_ADS_REFRESH_TOKEN!
        const login = (process.env.GOOGLE_ADS_LOGIN_CUSTOMER_ID||'').replace(/\D/g,'')
        const access = await getAccessToken(cid, cs, rt)
        const version = 'v19'
        let createAdBody: any
        if (action === 'clone_scale' && source_ad_id) {
          // Fetch source ad texts
          const fetchUrl = `https://googleads.googleapis.com/${version}/customers/${cust}/googleAds:search`
          const query = `SELECT ad.id, ad.responsive_search_ad.headlines, ad.responsive_search_ad.descriptions, ad.final_urls FROM ad WHERE ad.id = ${source_ad_id} LIMIT 1`
          const fh = { 'content-type':'application/json', 'authorization':`Bearer ${access}`, 'developer-token': dev } as any
          if (login) fh['login-customer-id'] = login
          const fRes = await fetch(fetchUrl, { method:'POST', headers: fh, body: JSON.stringify({ query }) })
          const fJson = await fRes.json().catch(()=>({ results: [] }))
          const ad = fJson?.results?.[0]?.ad || {}
          const srcH = (ad?.responsiveSearchAd?.headlines||[]).map((h:any)=> ({ text: h?.text }))
          const srcD = (ad?.responsiveSearchAd?.descriptions||[]).map((d:any)=> ({ text: d?.text }))
          const fin = (ad?.finalUrls||[])
          createAdBody = { operations: [{ create: { adGroup: `customers/${cust}/adGroups/${ad_group_id}`, status: 'PAUSED', ad: { responsiveSearchAd: { headlines: srcH, descriptions: srcD }, finalUrls: fin.length?fin:[final_url||'https://example.com'] } } }] }
        } else {
          const h = (headlines||[]).map((t)=> ({ text: String(t) }))
          const d = (descriptions||[]).map((t)=> ({ text: String(t) }))
          createAdBody = { operations: [{ create: { adGroup: `customers/${cust}/adGroups/${ad_group_id}`, status: 'PAUSED', ad: { responsiveSearchAd: { headlines: h, descriptions: d }, finalUrls: [final_url||'https://example.com'] } } }] }
        }
        const url = `https://googleads.googleapis.com/${version}/customers/${cust}/adGroupAds:mutate${validateOnly?'?validateOnly=true':''}`
        const headers: Record<string,string> = { 'content-type':'application/json', 'authorization': `Bearer ${access}`, 'developer-token': dev }
        if (login) headers['login-customer-id'] = login
        const mRes = await fetch(url, { method:'POST', headers, body: JSON.stringify(createAdBody) })
        const mJson = await mRes.json().catch(()=>({}))
        if (!mRes.ok) throw new Error(mJson?.error?.message || `mutate failed (${mRes.status})`)
        // Extract resource name if not validateOnly
        const resource = mJson?.results?.[0]?.resourceName || null
        ad_id_created = resource ? String(resource).split('/').pop() || null : null
        status = validateOnly ? 'validated' : 'published_paused'
      } catch (e:any) {
        error = e?.message || String(e)
      }
    } else {
      if (!allowMutations) error = 'mutations disabled'
      else if (!cust) error = 'missing customer id'
      else if (!ad_group_id) error = 'missing ad_group_id'
      else error = 'customer not allowlisted'
    }

    // Mark queue item + log
    try { await bq.query({ query: `UPDATE \`${projectId}.${dataset}.creative_publish_queue\` SET status='processed' WHERE run_id=@run`, params: { run: run_id } }) } catch {}
    try {
      const row = { ts: new Date().toISOString(), run_id, platform, platform_ids: JSON.stringify({ ad_id: ad_id_created, campaign_id }), status, policy_topics: JSON.stringify([]), error }
      await bq.dataset(dataset).table('creative_publish_log').insert([row])
    } catch {}

    return NextResponse.json({ ok: !error, run_id, status, ad_id: ad_id_created, validate_only: validateOnly, error })
  } catch (e:any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}
