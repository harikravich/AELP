import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
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
    FROM \
      \`${projectId}.${dataset}.ads_campaign_performance\`
    WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    GROUP BY date
    ORDER BY date DESC
  `
  const [rows] = await bq.query({ query: sql })
  return NextResponse.json({ rows })
}
