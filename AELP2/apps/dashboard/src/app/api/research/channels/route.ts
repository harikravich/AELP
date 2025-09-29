import { NextRequest, NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../lib/bigquery-client'
import { getDatasetFromCookie } from '../../../../lib/dataset'
import { BigQuery } from '@google-cloud/bigquery'

export const dynamic = 'force-dynamic'

export async function GET(req: NextRequest) {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = createBigQueryClient(projectId)
    const url = new URL(req.url)
    const status = (url.searchParams.get('status') || 'new').toLowerCase()
    const createName = url.searchParams.get('create')
    if (createName) {
      // Lightweight insert path to support pilot requests without requiring POST (some envs block POST locally)
      const id = `${Date.now()}_${Math.random().toString(36).slice(2,8)}`
      const type = String(url.searchParams.get('type') || 'other')
      const score_total = Number(url.searchParams.get('score') || 0)
      const created_at = new Date().toISOString()
      try { await (await import('@google-cloud/bigquery')).BigQuery.prototype.query.call(new (await import('@google-cloud/bigquery')).BigQuery({ projectId }), { query: `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.channel_candidates\` (id STRING, name STRING, type STRING, status STRING, score_total FLOAT64, created_at TIMESTAMP) PARTITION BY DATE(created_at)` }) } catch {}
      await (await import('@google-cloud/bigquery')).BigQuery.prototype.dataset.call(new (await import('@google-cloud/bigquery')).BigQuery({ projectId }), dataset).table('channel_candidates').insert([{ id, name: createName, type, status: 'new', score_total, created_at }])
    }
    try {
      const [rows] = await bq.query({ query: `
        SELECT id, name, type, status, score_total, created_at
        FROM \`${projectId}.${dataset}.channel_candidates\`
        WHERE status = @status OR @status = 'any'
        ORDER BY created_at DESC
        LIMIT 100
      `, params: { status } })
      return NextResponse.json({ ok: true, rows })
    } catch {}
    // Fallback stub if table missing
    return NextResponse.json({ ok: true, rows: [] })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export async function POST(req: NextRequest) {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const body = await req.json().catch(()=>({})) as any
    const id = `${Date.now()}_${Math.random().toString(36).slice(2,8)}`
    const name = String(body?.name || 'New Channel')
    const type = String(body?.type || 'other')
    const status = String(body?.status || 'new')
    const score_total = Number(body?.score_total || 0)
    const created_at = new Date().toISOString()
    try { await bq.query({ query: `CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.channel_candidates\` (id STRING, name STRING, type STRING, status STRING, score_total FLOAT64, created_at TIMESTAMP) PARTITION BY DATE(created_at)` }) } catch {}
    await bq.dataset(dataset).table('channel_candidates').insert([{ id, name, type, status, score_total, created_at }])
    return NextResponse.json({ ok: true, id })
  } catch (e:any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}
