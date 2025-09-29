import { NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../../lib/bigquery-client'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../lib/dataset'
import { getKpiSourceFromCookie } from '../../../../lib/kpi'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)
    const src = getKpiSourceFromCookie()
    if (src === 'ads') {
      const [rows] = await bq.query({ query: `
        WITH cur AS (
          SELECT SUM(cost) AS cost, SUM(conversions) AS conv, SUM(revenue) AS revenue
          FROM \`${projectId}.${dataset}.ads_kpi_daily\`
          WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
        ), prev AS (
          SELECT SUM(cost) AS cost, SUM(conversions) AS conv, SUM(revenue) AS revenue
          FROM \`${projectId}.${dataset}.ads_kpi_daily\`
          WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)
        )
        SELECT * FROM cur, prev` })
      const r:any = rows?.[0] || {}
      return NextResponse.json({
        source: 'ads',
        mode,
        dataset,
        cost: Number(r.cost||0),
        conv: Number(r.conv||0),
        revenue: Number(r.revenue||0),
        prev_cost: Number(r.cost_1||0),
        prev_conv: Number(r.conv_1||0),
        prev_revenue: Number(r.revenue_1||0)
      })
    }
    if (src === 'pacer') {
      const [rows] = await bq.query({ query: `
        WITH cur AS (
          SELECT SUM(spend) AS cost, SUM(d2c_total_subscribers) AS conv, NULL AS revenue
          FROM \`${projectId}.${dataset}.pacing_pacer_daily\`
          WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
        ), prev AS (
          SELECT SUM(spend) AS cost, SUM(d2c_total_subscribers) AS conv, NULL AS revenue
          FROM \`${projectId}.${dataset}.pacing_pacer_daily\`
          WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)
        )
        SELECT * FROM cur, prev` })
      const r:any = rows?.[0] || {}
      return NextResponse.json({
        source: 'pacer',
        mode,
        dataset,
        cost: Number(r.cost||0),
        conv: Number(r.conv||0),
        revenue: null,
        prev_cost: Number(r.cost_1||0),
        prev_conv: Number(r.conv_1||0),
        prev_revenue: null
      })
    }
    const enrollTable = src==='ga4_google' ? 'ga4_enrollments_google_cpc_session_daily' : 'ga4_enrollments_daily'
    const [rows] = await bq.query({ query: `
      WITH cur AS (
        SELECT 
          (SELECT SUM(cost) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()) AS cost,
          (SELECT SUM(enrollments) FROM \`${projectId}.${dataset}.${enrollTable}\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()) AS conv,
          (SELECT SUM(revenue) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()) AS revenue
      ), prev AS (
        SELECT 
          (SELECT SUM(cost) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)) AS cost,
          (SELECT SUM(enrollments) FROM \`${projectId}.${dataset}.${enrollTable}\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)) AS conv,
          (SELECT SUM(revenue) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)) AS revenue
      )
      SELECT cur.cost, cur.conv, cur.revenue, prev.cost AS prev_cost, prev.conv AS prev_conv, prev.revenue AS prev_revenue FROM cur, prev` })
    const r:any = rows?.[0] || {}
    return NextResponse.json({
      source: src,
      mode,
      dataset,
      cost: Number(r.cost||0),
      conv: Number(r.conv||0),
      revenue: Number(r.revenue||0),
      prev_cost: Number(r.prev_cost||0),
      prev_conv: Number(r.prev_conv||0),
      prev_revenue: Number(r.prev_revenue||0)
    })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message||String(e) }, { status: 500 })
  }
}
