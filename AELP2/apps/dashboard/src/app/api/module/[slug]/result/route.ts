import { NextRequest, NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../../lib/bigquery-client'

export async function GET(req: NextRequest, { params }: { params: { slug: string } }) {
  try {
    const url = new URL(req.url)
    const runId = String(url.searchParams.get('run_id')||'')
    if (!runId) return NextResponse.json({ ok:false, error: 'missing_run_id' }, { status: 400 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `SELECT summary_text, result_json FROM \`${projectId}.${dataset}.module_results\` WHERE run_id=@r LIMIT 1`, params: { r: runId } })
    const r = rows?.[0] || null
    return NextResponse.json({ ok: !!r, summary_text: r?.summary_text||null, result_json: r?.result_json||null })
  } catch (e:any) { return NextResponse.json({ ok:false, error: e?.message||String(e) }, { status: 200 }) }
}

export const dynamic = 'force-dynamic'
