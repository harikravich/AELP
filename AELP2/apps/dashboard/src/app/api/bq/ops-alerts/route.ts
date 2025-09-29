import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `
      SELECT timestamp, alert, severity, details
      FROM \`${projectId}.${dataset}.ops_alerts\`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
      ORDER BY timestamp DESC LIMIT 50
    `})
    return NextResponse.json({ rows })
  } catch (e:any) {
    return NextResponse.json({ rows: [] })
  }
}
