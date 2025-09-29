import { NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../lib/bigquery-client'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `SELECT date, from_channel, to_channel, cannibalization, lift FROM \`${projectId}.${dataset}.channel_interference_scores\` ORDER BY date DESC LIMIT 28` })
    return NextResponse.json({ rows })
  } catch (e:any) { return NextResponse.json({ rows: [], error: e?.message||String(e) }, { status: 200 }) }
}

export const dynamic = 'force-dynamic'
