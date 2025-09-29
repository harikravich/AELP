import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const sql = `
    SELECT *
    FROM \`${projectId}.${dataset}.fidelity_evaluations\`
    WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    ORDER BY timestamp DESC
    LIMIT 100
  `
  try {
    const [rows] = await bq.query({ query: sql })
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ rows: [], error: e?.message || String(e) })
  }
}
