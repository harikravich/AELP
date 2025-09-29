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
      SELECT start, end, experiment_id, platform, campaign_id, status, variants
      FROM \`${projectId}.${dataset}.ab_experiments\`
      ORDER BY start DESC LIMIT 50
    `})
    return NextResponse.json({ rows })
  } catch { return NextResponse.json({ rows: [] }) }
}
