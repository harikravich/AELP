import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const channel = url.searchParams.get('channel') || 'google_ads'
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `
      SELECT timestamp, channel, params, spend_grid, conv_grid, rev_grid, diagnostics
      FROM \`${projectId}.${dataset}.mmm_curves\`
      WHERE channel=@channel
      ORDER BY timestamp DESC LIMIT 1
    `, params: { channel } })
    if (!rows || !rows[0]) return NextResponse.json({ error: 'no rows' }, { status: 404 })
    const r: any = rows[0]
    const parse = (v:any)=> typeof v === 'string' ? JSON.parse(v) : v
    return NextResponse.json({
      timestamp: r.timestamp,
      channel: r.channel,
      params: parse(r.params),
      spend_grid: parse(r.spend_grid),
      conv_grid: parse(r.conv_grid),
      rev_grid: parse(r.rev_grid),
      diagnostics: parse(r.diagnostics),
    })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
