import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../lib/dataset'

export async function POST(req: Request) {
  try {
    let body: any = {}
    const ct = req.headers.get('content-type') || ''
    if (ct.includes('application/json')) body = await req.json().catch(()=>({}))
    else if (ct.includes('application/x-www-form-urlencoded') || ct.includes('multipart/form-data')) {
      const fd = await req.formData(); body = Object.fromEntries(Array.from(fd.entries()) as any)
    }
    const action = String(body?.action || '').toLowerCase()
    const experiment_id = String(body?.experiment_id || '')
    const notes = String(body?.notes || '')
    if (!['approve_start','approve_stop'].includes(action) || !experiment_id) {
      return NextResponse.json({ ok: false, error: 'action=approve_start|approve_stop and experiment_id required' }, { status: 200 })
    }
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset, allowed, mode, reason } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })
    const bq = new BigQuery({ projectId })
    const table = `\`${projectId}.${dataset}.ab_approvals\``
    const ensure = `CREATE TABLE IF NOT EXISTS ${table} (timestamp TIMESTAMP, experiment_id STRING, action STRING, notes STRING) PARTITION BY DATE(timestamp)`
    await bq.query({ query: ensure })
    await bq.dataset(dataset).table('ab_approvals').insert([{ timestamp: new Date().toISOString(), experiment_id, action, notes }])
    return NextResponse.json({ ok: true, experiment_id, action })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
