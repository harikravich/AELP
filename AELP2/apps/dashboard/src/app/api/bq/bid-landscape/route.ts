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
      SELECT campaign_id, cpc, win_rate
      FROM \`${projectId}.${dataset}.bid_landscape_curves\`
      ORDER BY campaign_id, cpc
    `})
    return NextResponse.json({ rows })
  } catch { return NextResponse.json({ rows: [] }) }
}
