import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(()=>({})) as any
    const run_id = String(body?.run_id || '')
    if (!run_id) return NextResponse.json({ ok:false, error:'run_id required' }, { status: 400 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    await bq.query({ query: `UPDATE \`${projectId}.${dataset}.creative_publish_queue\` SET status='rejected' WHERE run_id=@run`, params: { run: run_id } })
    return NextResponse.json({ ok: true, run_id })
  } catch (e:any) { return NextResponse.json({ ok:false, error: e?.message||String(e) }, { status: 200 }) }
}

