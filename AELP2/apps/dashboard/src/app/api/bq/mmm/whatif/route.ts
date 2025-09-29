import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET(req: NextRequest) {
  try {
    const url = new URL(req.url)
    const channel = url.searchParams.get('channel') || 'google_ads'
    const budget = Number(url.searchParams.get('budget') || '0')
    if (!budget || budget <= 0) return NextResponse.json({ error: 'budget>0 required' }, { status: 400 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `
      SELECT spend_grid, conv_grid FROM \`${projectId}.${dataset}.mmm_curves\`
      WHERE channel=@ch ORDER BY timestamp DESC LIMIT 1`, params: { ch: channel } })
    if (!rows?.[0]) return NextResponse.json({ error: `No MMM curve for ${channel}` }, { status: 404 })
    const spend: number[] = JSON.parse(String(rows[0].spend_grid))
    const conv: number[] = JSON.parse(String(rows[0].conv_grid))
    let idx = 0, best = Number.POSITIVE_INFINITY
    for (let i=0;i<spend.length;i++){ const d = Math.abs(spend[i]-budget); if(d<best){best=d;idx=i} }
    const predicted_conversions = Number(conv[idx]||0)
    const cac = budget/Math.max(1,predicted_conversions)
    return NextResponse.json({ ok: true, channel, budget, predicted_conversions, cac })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 200 })
  }
}

