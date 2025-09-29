import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    try {
      const [rows] = await bq.query({ query: `
        SELECT timestamp, arm, mean, variance
        FROM \`${projectId}.${dataset}.rl_arms_posteriors\`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY timestamp DESC, arm
        LIMIT 500
      `})
      return NextResponse.json({ rows })
    } catch {
      return NextResponse.json({ rows: [] })
    }
  } catch (e:any) {
    return NextResponse.json({ rows: [], error: e?.message || String(e) }, { status: 200 })
  }
}

