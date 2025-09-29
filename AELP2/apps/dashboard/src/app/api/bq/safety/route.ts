import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const sql = `
    WITH last AS (
      SELECT * FROM \`${projectId}.${dataset}.safety_events\`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
    )
    SELECT 'timeline' AS section, TO_JSON_STRING(x) AS payload FROM (
      SELECT DATE(timestamp) AS date, severity, COUNT(*) AS events
      FROM last GROUP BY date, severity ORDER BY date
    ) x
    UNION ALL
    SELECT 'latest_critical' AS section, TO_JSON_STRING(x) AS payload FROM (
      SELECT * FROM last WHERE severity IN ('HIGH','CRITICAL') ORDER BY timestamp DESC LIMIT 20
    ) x
  `
  try {
    const [rows] = await bq.query({ query: sql })
    return NextResponse.json({ rows })
  } catch (e: any) {
    // Table may not exist yet; return empty shape
    return NextResponse.json({ rows: [], error: e?.message || String(e) }, { status: 200 })
  }
}
