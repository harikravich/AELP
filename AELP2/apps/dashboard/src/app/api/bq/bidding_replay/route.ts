import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const episodeId = searchParams.get('episode_id')
  const step = searchParams.get('step')
  if (!episodeId || !step) {
    return NextResponse.json({ error: 'episode_id and step are required' }, { status: 400 })
  }
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const sql = `
    SELECT timestamp, user_id, campaign_id, bid_amount, won, price_paid, context, explain
    FROM \`${projectId}.${dataset}.bidding_events\`
    WHERE episode_id = @episodeId AND step = @step
    ORDER BY timestamp DESC
    LIMIT 1
  `
  const [rows] = await bq.query({
    query: sql,
    params: { episodeId, step: parseInt(step, 10) },
  })
  return NextResponse.json({ row: rows[0] || null })
}
