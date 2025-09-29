import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../lib/dataset'

export async function POST(req: Request) {
  try {
    let body: any = {}
    const ct = req.headers.get('content-type') || ''
    if (ct.includes('application/json')) body = await req.json().catch(() => ({}))
    else if (ct.includes('application/x-www-form-urlencoded') || ct.includes('multipart/form-data')) {
      const fd = await req.formData()
      body = Object.fromEntries(Array.from(fd.entries()) as any)
    }
    const action = String(body?.action || '').toLowerCase()
    const objective = String(body?.objective || '')
    const notes = String(body?.notes || '')
    if (!['approve','deny'].includes(action) || !objective) {
      return NextResponse.json({ ok: false, error: 'action=approve|deny and objective required' }, { status: 200 })
    }
    if ((process.env.GATES_ENABLED || '1') === '1' && action === 'approve' && (process.env.AELP2_ALLOW_OPPORTUNITY_APPROVALS || '0') !== '1') {
      return NextResponse.json({ ok: false, error: 'flag_denied: AELP2_ALLOW_OPPORTUNITY_APPROVALS=0' }, { status: 200 })
    }
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset, allowed, mode, reason } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })
    const bq = new BigQuery({ projectId })
    const table = `\`${projectId}.${dataset}.opportunity_approvals\``
    const ensureSql = `CREATE TABLE IF NOT EXISTS ${table} (
      timestamp TIMESTAMP,
      objective STRING,
      action STRING,
      notes STRING
    ) PARTITION BY DATE(timestamp)`
    await bq.query({ query: ensureSql })
    const row = { timestamp: new Date().toISOString(), objective, action, notes }
    await bq.dataset(dataset).table('opportunity_approvals').insert([row])
    return NextResponse.json({ ok: true, row })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
