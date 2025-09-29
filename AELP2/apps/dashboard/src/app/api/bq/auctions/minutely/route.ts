import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const url = new URL(req.url)
    const windowMin = Math.max(5, Math.min(24*60, Number(url.searchParams.get('window')||60)))
    const [rows] = await bq.query({ query: `
      SELECT minute, auctions, wins, win_rate, avg_bid, avg_price_paid
      FROM \`${projectId}.${dataset}.bidding_events_per_minute\`
      WHERE minute >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @w MINUTE)
      ORDER BY minute DESC LIMIT 1000
    `, params: { w: windowMin } })
    return NextResponse.json({ rows, window: windowMin })
  } catch { return NextResponse.json({ rows: [] }) }
}
