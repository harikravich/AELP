import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const sql = `
      SELECT timestamp, flow, rc_map, failures, ok
      FROM \`${projectId}.${dataset}.ops_flow_runs\`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
      ORDER BY timestamp DESC
      LIMIT 100`
    const [rows] = await bq.query({ query: sql })
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || String(e), rows: [] }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
