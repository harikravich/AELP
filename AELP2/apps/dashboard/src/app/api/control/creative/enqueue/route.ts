import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../../lib/dataset'

export async function POST(req: NextRequest) {
  try {
    const { dataset, allowed, reason } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok:false, error: reason }, { status: 403 })
    const body = await req.json().catch(()=>({})) as any
    const platform = String(body?.platform||'google_ads')
    const type = String(body?.type||'rsa')
    const campaign_id = body?.campaign_id || null
    const ad_group_id = body?.ad_group_id || null
    const asset_group_id = body?.asset_group_id || null
    const payload = JSON.stringify(body?.payload || {})
    const requested_by = body?.requested_by || 'ui'
    const run_id = `${Date.now()}_${Math.random().toString(36).slice(2,8)}`
    const bq = new BigQuery({ projectId: process.env.GOOGLE_CLOUD_PROJECT })
    const ensure = `CREATE TABLE IF NOT EXISTS \`${process.env.GOOGLE_CLOUD_PROJECT}.${dataset}.creative_publish_queue\` (enqueued_at TIMESTAMP, run_id STRING, platform STRING, type STRING, campaign_id STRING, ad_group_id STRING, asset_group_id STRING, payload JSON, status STRING, requested_by STRING) PARTITION BY DATE(enqueued_at)`
    await bq.query({ query: ensure })
    await bq.dataset(dataset).table('creative_publish_queue').insert([{ enqueued_at: new Date().toISOString(), run_id, platform, type, campaign_id, ad_group_id, asset_group_id, payload, status: 'queued', requested_by }])
    return NextResponse.json({ ok:true, run_id })
  } catch (e:any) { return NextResponse.json({ ok:false, error: e?.message||String(e) }, { status: 200 }) }
}

export const dynamic = 'force-dynamic'
