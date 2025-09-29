import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

// Minimal create: enqueue an RSA creation request to approvals queue (HITL)
export async function POST(req: NextRequest) {
  try {
    const { allowed, reason, dataset, mode } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })
    const body = await req.json().catch(()=>({})) as any
    const final_url = String(body?.final_url || '')
    const headlines = (Array.isArray(body?.headlines) ? body.headlines : []).map((s:any)=> String(s)).filter(Boolean)
    const descriptions = (Array.isArray(body?.descriptions) ? body.descriptions : []).map((s:any)=> String(s)).filter(Boolean)
    const campaign_id = body?.campaign_id ? String(body.campaign_id) : null
    const ad_group_id = body?.ad_group_id ? String(body.ad_group_id) : null
    if (!final_url || headlines.length===0 || descriptions.length===0) {
      return NextResponse.json({ ok:false, error: 'final_url, headlines[], descriptions[] required' }, { status: 400 })
    }
    const run_id = `${Date.now()}_${Math.random().toString(36).slice(2,8)}`
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const bq = new BigQuery({ projectId })
    // Ensure queue table exists
    try {
      await bq.query({ query: `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.creative_publish_queue\` (
        run_id STRING, status STRING, type STRING, platform STRING, requested_by STRING, payload JSON, enqueued_at TIMESTAMP
      ) PARTITION BY DATE(enqueued_at)` })
    } catch {}
    const payload = { action:'create', type:'rsa', final_url, headlines, descriptions, campaign_id, ad_group_id }
    await bq.dataset(dataset).table('creative_publish_queue').insert([{ run_id, status:'queued', type:'rsa', platform:'google_ads', requested_by:'creative_studio', payload, enqueued_at: new Date() }])
    return NextResponse.json({ ok:true, run_id })
  } catch (e:any) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 200 })
  }
}

