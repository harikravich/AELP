import React from 'react'
import { TimeSeriesChart } from '../../components/TimeSeriesChart'
import { BanditProposalsTable } from '../../components/BanditProposalsTable'
import ExecClient from './ExecClient'
import dayjs from 'dayjs'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../lib/dataset'
import { fmtWhen } from '../../lib/utils'
import { createBigQueryClient } from '../../lib/bigquery-client'
import MetricTile from '../../components/MetricTile'
import { TrendingUp, TrendingDown, Activity, DollarSign, Target, Zap, AlertCircle, CheckCircle2 } from 'lucide-react'

async function fetchAdsDaily() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  const sql = `
    SELECT DATE(date) AS date,
           SUM(impressions) AS impressions,
           SUM(clicks) AS clicks,
           SUM(cost_micros)/1e6 AS cost,
           SUM(conversions) AS conversions,
           SUM(conversion_value) AS revenue,
           SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
           SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr,
           SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac,
           SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS roas,
           APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS impression_share_p50
    FROM 
      \`${projectId}.${dataset}.ads_campaign_performance\`
    WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    GROUP BY date
    ORDER BY date DESC`
  try {
    const [rows] = await bq.query({ query: sql })
    return { rows }
  } catch {
    return { rows: [] as any[] }
  }
}

async function fetchKpiDaily() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  const sql = `
    SELECT date, conversions, revenue, cost,
           SAFE_DIVIDE(cost, NULLIF(conversions,0)) AS cac,
           SAFE_DIVIDE(revenue, NULLIF(cost,0)) AS roas
    FROM \`${projectId}.${dataset}.ads_kpi_daily\`
    WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    ORDER BY date DESC`
  try {
    const [rows] = await bq.query({ query: sql })
    return { rows }
  } catch {
    return { rows: [] as any[] }
  }
}

async function fetchTopSegments() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  const [rows] = await bq.query({ query: `
    WITH last_day AS (
      SELECT MAX(date) AS d FROM \`${projectId}.${dataset}.segment_scores_daily\`
    )
    SELECT s.date, s.segment, s.score
    FROM \`${projectId}.${dataset}.segment_scores_daily\` s, last_day
    WHERE s.date = last_day.d
    ORDER BY s.score DESC
    LIMIT 10
  `})
  return { rows }
}

async function fetchGa4Summary() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const dataset = (cookies().get(DATASET_COOKIE)?.value === 'prod' ? PROD_DATASET : SANDBOX_DATASET)
  const bq = createBigQueryClient(projectId)
  let last: any = null
  let byDevice: any[] = []
  let byChannel: any[] = []
  try {
    const [r1] = await bq.query({ query: `SELECT MAX(DATE(date)) AS d FROM \`${projectId}.${dataset}.ga4_aggregates\`` })
    last = r1?.[0]?.d || null
  } catch {}
  try {
    const [rows] = await bq.query({ query: `
      SELECT date, device_category, SUM(conversions) AS conv
      FROM \`${projectId}.${dataset}.ga4_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND CURRENT_DATE()
      GROUP BY date, device_category
      ORDER BY date DESC, conv DESC
      LIMIT 9
    `})
    byDevice = rows
  } catch {}
  try {
    const [rows] = await bq.query({ query: `
      SELECT default_channel_group, SUM(conversions) AS conv
      FROM \`${projectId}.${dataset}.ga4_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      GROUP BY default_channel_group
      ORDER BY conv DESC
      LIMIT 6
    `})
    byChannel = rows
  } catch {}
  return { last, byDevice, byChannel }
}

export default async function ExecPage() {
  const ads = await fetchAdsDaily()
  const kpi = await fetchKpiDaily()
  const ga4 = await fetchGa4Summary()
  let segments: any = { rows: [] }
  try { segments = await fetchTopSegments() } catch {}
  
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const dataset = (cookies().get(DATASET_COOKIE)?.value === 'prod' ? PROD_DATASET : SANDBOX_DATASET)
  const bq = createBigQueryClient(projectId)
  
  // Fetch bandit proposals (shadow)
  let banditProps: any[] = []
  try {
    const [rows] = await bq.query({ query: `
      SELECT timestamp, platform, channel, campaign_id, ad_id, action, exploration_pct, reason, shadow, applied
      FROM \`${projectId}.${dataset}.bandit_change_proposals\`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      ORDER BY timestamp DESC
      LIMIT 50
    `})
    banditProps = rows
  } catch (e) {
    banditProps = []
  }

  const kpiRows = (kpi.rows || []).slice().reverse()
  const adsRows = (ads.rows || []).slice().reverse()
  const latest = kpiRows[kpiRows.length - 1] || {}
  const prevLatest = kpiRows[kpiRows.length - 2] || {}
  
  const fmt = (n: any, d=2) => (Number(n ?? 0)).toFixed(d)
  const dateFmt = (s: string) => dayjs(s).format('MM-DD')
  
  const kpiSeries = kpiRows.map((r: any) => ({
    date: dateFmt(r.date),
    cost: Number(r.cost || 0),
    conversions: Number(r.conversions || 0),
    revenue: Number(r.revenue || 0),
    cac: Number(r.cac || 0),
    roas: Number(r.roas || 0),
  }))
  
  const isSeries = adsRows.map((r: any) => ({
    date: dateFmt(r.date),
    impression_share: Number(r.impression_share_p50 || 0),
  }))
  
  const kpiSeriesPlain = JSON.parse(JSON.stringify(kpiSeries))
  const isSeriesPlain = JSON.parse(JSON.stringify(isSeries))
  
  // Normalize bandit proposals for client component (plain JSON only)
  const banditPropsPlain = (banditProps || []).map((r:any)=> ({
    timestamp: fmtWhen(r.timestamp),
    platform: String(r.platform ?? ''),
    channel: String(r.channel ?? ''),
    campaign_id: String(r.campaign_id ?? ''),
    ad_id: String(r.ad_id ?? ''),
    action: String(r.action ?? ''),
    exploration_pct: Number(r.exploration_pct ?? 0),
    reason: String(r.reason ?? ''),
    shadow: Boolean(r.shadow),
    applied: Boolean(r.applied),
  }))

  // Calculate trends
  const spendTrend = latest.cost > prevLatest.cost ? 'up' : latest.cost < prevLatest.cost ? 'down' : 'neutral'
  const convTrend = latest.conversions > prevLatest.conversions ? 'up' : latest.conversions < prevLatest.conversions ? 'down' : 'neutral'
  const cacTrend = latest.cac < prevLatest.cac ? 'up' : latest.cac > prevLatest.cac ? 'down' : 'neutral'
  const roasTrend = latest.roas > prevLatest.roas ? 'up' : latest.roas < prevLatest.roas ? 'down' : 'neutral'

  const spendChange = prevLatest.cost ? ((latest.cost - prevLatest.cost) / prevLatest.cost * 100).toFixed(1) + '%' : ''
  const convChange = prevLatest.conversions ? ((latest.conversions - prevLatest.conversions) / prevLatest.conversions * 100).toFixed(1) + '%' : ''
  const cacChange = prevLatest.cac ? ((prevLatest.cac - latest.cac) / prevLatest.cac * 100).toFixed(1) + '%' : ''
  const roasChange = prevLatest.roas ? ((latest.roas - prevLatest.roas) / prevLatest.roas * 100).toFixed(1) + '%' : ''

  return (
    <ExecClient
      initialData={{ latest, prevLatest, ga4, segments, spendTrend, convTrend, cacTrend, roasTrend, spendChange, convChange, cacChange, roasChange }}
      kpiSeries={kpiSeriesPlain}
      isSeries={isSeriesPlain}
      banditProps={banditPropsPlain}
    />
  )
}
