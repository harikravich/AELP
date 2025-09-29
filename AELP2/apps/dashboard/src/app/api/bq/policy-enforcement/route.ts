import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `
      SELECT timestamp, rule, target, decision, notes
      FROM \`${projectId}.${dataset}.policy_enforcement\`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
      ORDER BY timestamp DESC
      LIMIT 100
    ` })
    return NextResponse.json({ ok: true, rows })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
