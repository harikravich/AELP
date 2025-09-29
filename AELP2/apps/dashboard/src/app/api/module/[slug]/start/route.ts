import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import crypto from 'crypto'

export async function POST(req: NextRequest, { params }: { params: { slug: string } }) {
  try {
    const { slug } = params
    const body = await req.json().catch(()=>({})) as any
    const consent = !!body?.consent
    const pageUrl = String(body?.page_url||'')
    if (!consent) return NextResponse.json({ ok:false, error: 'consent required' }, { status: 400 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    const bq = new BigQuery({ projectId })
    const ensureRuns = `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.lp_module_runs\` (run_id STRING, slug STRING, page_url STRING, consent_id STRING, created_ts TIMESTAMP, status STRING, elapsed_ms INT64, error_code STRING) PARTITION BY DATE(created_ts)`
    const ensureConsents = `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.consent_logs\` (consent_id STRING, slug STRING, page_url STRING, consent_text STRING, ip_hash STRING, user_agent STRING, ts TIMESTAMP) PARTITION BY DATE(ts)`
    const ensureResults = `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.module_results\` (run_id STRING, slug STRING, summary_text STRING, result_json JSON, expires_at TIMESTAMP) PARTITION BY DATE(expires_at)`
    await bq.query({ query: ensureRuns }); await bq.query({ query: ensureConsents }); await bq.query({ query: ensureResults })
    const runId = crypto.randomBytes(8).toString('hex')
    const consentId = crypto.randomBytes(6).toString('hex')
    const ua = req.headers.get('user-agent') || ''
    const ip = (req.headers.get('x-forwarded-for') || '').split(',')[0] || ''
    const ipHash = crypto.createHash('sha256').update(ip).digest('hex').slice(0,16)
    await bq.dataset(dataset).table('consent_logs').insert([{ consent_id: consentId, slug, page_url: pageUrl, consent_text: 'accepted', ip_hash: ipHash, user_agent: ua, ts: new Date().toISOString() }])
    await bq.dataset(dataset).table('lp_module_runs').insert([{ run_id: runId, slug, page_url: pageUrl, consent_id: consentId, created_ts: new Date().toISOString(), status: 'running', elapsed_ms: 0, error_code: null }])
    // Demo compute inline
    let summary = 'Preview ready.'
    let result: any = {}
    if (slug === 'insight_preview') {
      summary = 'Late-night activity looks above typical (demo).'
      result = { bars: [{ label: 'Day', v: 42 }, { label: 'Night', v: 88 }], hints: ['Night posts ~2.1x Day (14d)', 'Public bio mentions gaming'] }
    } else if (slug === 'scam_check') {
      summary = 'Link risk: Low (demo) — new domain but no suspicious redirects.'
      result = { risk: 'low', reasons: ['Domain age > 6 months', 'TLS valid', 'Redirects: 0'] }
    } else {
      summary = 'Module not recognized (demo).'
      result = {}
    }
    const expires = new Date(Date.now() + 24*3600*1000).toISOString()
    await bq.dataset(dataset).table('module_results').insert([{ run_id: runId, slug, summary_text: summary, result_json: JSON.stringify(result), expires_at: expires }])
    await bq.dataset(dataset).table('lp_module_runs').insert([{ run_id: runId, slug, page_url: pageUrl, consent_id: consentId, created_ts: new Date().toISOString(), status: 'done', elapsed_ms: 500, error_code: null }])
    return NextResponse.json({ ok:true, run_id: runId, status: 'done' })
  } catch (e:any) { return NextResponse.json({ ok:false, error: e?.message||String(e) }, { status: 200 }) }
}

export const dynamic = 'force-dynamic'
