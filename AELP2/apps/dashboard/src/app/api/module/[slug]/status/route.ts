import { NextRequest, NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../../lib/bigquery-client'

export async function GET(req: NextRequest, { params }: { params: { slug: string } }) {
  try {
    const url = new URL(req.url)
    const runId = String(url.searchParams.get('run_id')||'')
    if (!runId) return NextResponse.json({ status: 'error', error_code: 'missing_run_id' }, { status: 400 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `SELECT status, elapsed_ms, error_code FROM \`${projectId}.${dataset}.lp_module_runs\` WHERE run_id=@r ORDER BY created_ts DESC LIMIT 1`, params: { r: runId } })
    const r = rows?.[0] || { status: 'unknown' }
    return NextResponse.json(r)
  } catch (e:any) { return NextResponse.json({ status: 'error', error_code: e?.message||String(e) }, { status: 200 }) }
}

export const dynamic = 'force-dynamic'
