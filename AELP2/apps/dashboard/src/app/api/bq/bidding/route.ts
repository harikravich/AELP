import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const sql = `
    SELECT * FROM \`${projectId}.${dataset}.bidding_events_per_minute\`
    WHERE minute >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
    ORDER BY minute ASC
  `
  const [rows] = await bq.query({ query: sql })
  return NextResponse.json({ rows })
}
