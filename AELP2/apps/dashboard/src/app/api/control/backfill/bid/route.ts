import { NextResponse } from 'next/server'
import { resolveDatasetForAction } from '../../../../../lib/dataset'
import { BigQuery } from '@google-cloud/bigquery'

export const dynamic = 'force-dynamic'

export async function POST() {
  try {
    const { allowed, reason, dataset, mode } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok:false, error: reason, mode, dataset }, { status: 403 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const bq = new BigQuery({ projectId })
    try { await bq.query({ query: `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.ops_flow_runs\` (id STRING, flow STRING, status STRING, created_at TIMESTAMP, meta JSON) PARTITION BY DATE(created_at)` }) } catch {}
    const id = `${Date.now()}_${Math.random().toString(36).slice(2,8)}`
    await bq.dataset(dataset).table('ops_flow_runs').insert([{ id, flow:'backfill_bid_landscape', status:'queued', created_at: new Date(), meta: { source:'external_ui' } }])
    return NextResponse.json({ ok:true, id })
  } catch (e:any) { return NextResponse.json({ ok:false, error: e?.message||String(e) }, { status: 200 }) }
}

