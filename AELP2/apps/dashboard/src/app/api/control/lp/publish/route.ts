import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

function uid(prefix='lp_') {
  return `${prefix}${Math.random().toString(36).slice(2,10)}`
}

export async function POST(req: NextRequest) {
  try {
    const { dataset, mode, allowed, reason } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok:false, error: reason, mode, dataset }, { status: 403 })

    // Accept form or JSON
    let payload: any = {}
    const contentType = req.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      payload = await req.json().catch(()=> ({}))
    } else {
      const form = await req.formData().catch(()=> null)
      if (form) {
        for (const [k,v] of form.entries()) payload[k] = String(v)
      }
    }

    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const bq = new BigQuery({ projectId })
    const test_id = String(payload.test_id || uid('lp_'))
    const lp_a = String(payload.lp_a || 'https://example.com/lp-a')
    const lp_b = String(payload.lp_b || 'https://example.com/lp-b')
    const traffic_split = Number(payload.traffic_split ?? 50)
    const primary_metric = String(payload.primary_metric || 'conversion_rate')
    const status = 'draft'
    const now = new Date().toISOString()

    // Ensure table
    try {
      await bq.query({ query: `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.lp_tests\` (
        created_at TIMESTAMP, test_id STRING, lp_a STRING, lp_b STRING, status STRING, traffic_split INT64, primary_metric STRING
      ) PARTITION BY DATE(created_at)` })
    } catch {}

    await bq.dataset(dataset).table('lp_tests').insert([{ created_at: now, test_id, lp_a, lp_b, status, traffic_split, primary_metric }])
    return NextResponse.json({ ok:true, test_id, status, lp_a, lp_b, traffic_split, primary_metric })
  } catch (e:any) {
    return NextResponse.json({ ok:false, error: e?.message || String(e) }, { status: 200 })
  }
}

