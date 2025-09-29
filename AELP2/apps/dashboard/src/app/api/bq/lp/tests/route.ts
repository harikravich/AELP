import { NextResponse } from 'next/server'
import { getDatasetFromCookie } from '../../../../../lib/dataset'
import { createBigQueryClient } from '../../../../../lib/bigquery-client'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({
      query: `
        SELECT created_at, test_id, lp_a, lp_b, status, traffic_split, primary_metric
        FROM \`${projectId}.${dataset}.lp_tests\`
        ORDER BY created_at DESC
        LIMIT 50
      `
    })
    return NextResponse.json({ ok: true, rows })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}
