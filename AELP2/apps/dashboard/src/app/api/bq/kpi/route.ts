import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const sql = `
    SELECT date, conversions, revenue, cost,
           SAFE_DIVIDE(cost, NULLIF(conversions,0)) AS cac,
           SAFE_DIVIDE(revenue, NULLIF(cost,0)) AS roas
    FROM \`${projectId}.${dataset}.ads_kpi_daily\`
    WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    ORDER BY date DESC
  `
  try {
    const [rows] = await bq.query({ query: sql })
    return NextResponse.json({ rows })
  } catch (e: any) {
    // Graceful fallback if view is missing or permissions prevent query
    const message = e?.message || String(e)
    return NextResponse.json({ rows: [], error: message, dataset }, { status: 200 })
  }
}
