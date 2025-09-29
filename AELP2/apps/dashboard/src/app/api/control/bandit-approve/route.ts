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
    const table = `\`${projectId}.${dataset}.bandit_change_approvals\``
    const ensureSql = `CREATE TABLE IF NOT EXISTS ${table} (
      timestamp TIMESTAMP,
      campaign_id STRING,
      ad_id STRING,
      action STRING,
      approved BOOL,
      approver STRING,
      reason STRING
    ) PARTITION BY DATE(timestamp)`
    await bq.query({ query: ensureSql })
    const row = {
      timestamp: new Date().toISOString(),
      campaign_id: String(body?.campaign_id || ''),
      ad_id: String(body?.ad_id || ''),
      action: String(body?.action || 'adjust_split'),
      approved: Boolean(body?.approved ?? false),
      approver: String(body?.approver || 'dashboard'),
      reason: String(body?.reason || ''),
    }
    const [job] = await bq.dataset(dataset).table('bandit_change_approvals').insert([row])
    return NextResponse.json({ ok: true, row })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
