import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie, resolveDatasetForAction } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

function bq() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT!
  return new BigQuery({ projectId })
}

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset } = getDatasetFromCookie()
    await bq().query({ query: `
      CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.value_upload_requests\` (
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        channel STRING, source STRING, date_start DATE, date_end DATE,
        multiplier FLOAT64, notes STRING, user STRING, status STRING
      )
    `})
    const [rows] = await bq().query({ query: `
      SELECT * FROM \`${projectId}.${dataset}.value_upload_requests\`
      ORDER BY created_at DESC LIMIT 50
    `})
    return NextResponse.json({ items: rows })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { channel, date_start, date_end, multiplier, notes, source = 'dashboard', user = 'unknown' } = body || {}
    if (!channel || !date_start || !date_end || typeof multiplier !== 'number') {
      return NextResponse.json({ error: 'channel, date_start, date_end, multiplier required' }, { status: 400 })
    }
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset, allowed, mode, reason } = resolveDatasetForAction('write')
    if (!allowed) return NextResponse.json({ error: reason, mode, dataset }, { status: 403 })
    const client = bq()
    // Ensure table exists then insert
    await client.query({ query: `
      CREATE TABLE IF NOT EXISTS \`${projectId}.${dataset}.value_upload_requests\` (
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        channel STRING, source STRING, date_start DATE, date_end DATE,
        multiplier FLOAT64, notes STRING, user STRING, status STRING
      )
    ` })
    const [job] = await client.dataset(dataset).table('value_upload_requests').insert([{
      channel, source, date_start, date_end, multiplier, notes, user,
      status: (process.env.AELP2_ALLOW_GOOGLE_MUTATIONS === '1') ? 'submitted' : 'audited',
    }])
    const allow = process.env.AELP2_ALLOW_GOOGLE_MUTATIONS === '1'
    return NextResponse.json({ ok: true, status: allow ? 'submitted' : 'audited' })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
