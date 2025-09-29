import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const sql = `
    SELECT DATE(timestamp) AS date,
           SUM(spend) AS spend,
           SUM(revenue) AS revenue,
           SUM(conversions) AS conversions,
           SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) AS roas,
           SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) AS cac,
           AVG(win_rate) AS avg_win_rate
    FROM \
      \`${projectId}.${dataset}.training_episodes\`
    WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    GROUP BY date
    ORDER BY date DESC
  `
  const [rows] = await bq.query({ query: sql })
  return NextResponse.json({ rows })
}
