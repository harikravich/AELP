import { NextRequest, NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../../lib/bigquery-client'
import { resolveDatasetForAction, getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `
      SELECT cell_key, angle, audience, channel, lp, offer, last_seen, spend, clicks, conversions, revenue, cac, value
      FROM \`${projectId}.${dataset}.explore_cells\`
      ORDER BY last_seen DESC
      LIMIT 100
    ` })
    return NextResponse.json({ ok: true, rows })
  } catch (e: any) {
    return NextResponse.json({ ok: false, rows: [], error: e?.message || String(e) }, { status: 200 })
  }
}

export async function POST(req: NextRequest) {
  try {
    const { allowed, reason, dataset } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ ok: false, error: reason }, { status: 403 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const bq = createBigQueryClient(projectId)
    const form = await req.formData().catch(()=>null)
    const angle = String(form?.get('angle') || '')
    const audience = String(form?.get('audience') || '')
    const channel = String(form?.get('channel') || '')
    const lp = String(form?.get('lp') || '')
    const offer = String(form?.get('offer') || '')
    const cell_key = `${angle}:${audience}:${channel}:${lp}:${offer}`
    const row = { cell_key, angle, audience, channel, lp, offer, last_seen: new Date().toISOString(), spend: 0, clicks: 0, conversions: 0, revenue: 0, cac: 0, value: 0 }
    await bq.dataset(dataset).table('explore_cells').insert([row])
    return NextResponse.json({ ok: true, cell_key })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

