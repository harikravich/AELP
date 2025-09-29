import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../lib/dataset'

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({})) as any
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset, allowed, mode, reason } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })
    const bq = new BigQuery({ projectId })
    const table = `\`${projectId}.${dataset}.ab_exposures\``
    const ensureSql = `CREATE TABLE IF NOT EXISTS ${table} (
      timestamp TIMESTAMP,
      experiment STRING,
      variant STRING,
      subject_id STRING,
      context JSON
    ) PARTITION BY DATE(timestamp)`
    await bq.query({ query: ensureSql })
    const row = {
      timestamp: new Date().toISOString(),
      experiment: String(body?.experiment || ''),
      variant: String(body?.variant || ''),
      subject_id: String(body?.subject_id || ''),
      context: JSON.stringify(body?.context || {}),
    }
    await bq.dataset(dataset).table('ab_exposures').insert([row])
    return NextResponse.json({ ok: true, row })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
