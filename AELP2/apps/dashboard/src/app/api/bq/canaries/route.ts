import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const sql = `
      SELECT timestamp, customer_id, campaign_id, campaign_name, old_budget, new_budget,
             delta_pct, direction, shadow, applied, error, actor, notes
      FROM \`${projectId}.${dataset}.canary_changes\`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      ORDER BY timestamp DESC
      LIMIT 200`;
    const [rows] = await bq.query({ query: sql })
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || String(e), rows: [] }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
