import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const url = new URL(req.url)
    const status = (url.searchParams.get('status') || 'queued').toLowerCase()
    const sql = `
      CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.creative_publish_queue\` (
        enqueued_at TIMESTAMP, run_id STRING, platform STRING, type STRING, campaign_id STRING, ad_group_id STRING, asset_group_id STRING, payload JSON, status STRING, requested_by STRING
      ) PARTITION BY DATE(enqueued_at);
      SELECT enqueued_at, run_id, platform, type, campaign_id, ad_group_id, asset_group_id, payload, status, requested_by
      FROM \`${projectId}.${dataset}.creative_publish_queue\`
      WHERE (@status = 'any' OR status = @status)
      ORDER BY enqueued_at DESC
      LIMIT 200`;
    const [rows] = await bq.query({ query: sql, params: { status } })
    return NextResponse.json({ rows })
  } catch (e:any) {
    return NextResponse.json({ rows: [], error: e?.message||String(e) }, { status: 200 })
  }
}
