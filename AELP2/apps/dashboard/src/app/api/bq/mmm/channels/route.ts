import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    // Try to infer channels from mmm_curves table if present; fallback to distinct channel from mmm_allocations
    let channels: string[] = []
    try {
      const [rows] = await bq.query({ query: `
        SELECT DISTINCT channel FROM \`${projectId}.${dataset}.mmm_curves\`
        WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND CURRENT_DATE()
      `})
      channels = rows.map((r:any)=> r.channel).filter(Boolean)
    } catch {}
    if (channels.length === 0) {
      try {
        const [rows] = await bq.query({ query: `
          SELECT DISTINCT channel FROM \`${projectId}.${dataset}.mmm_allocations\`
          WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND CURRENT_DATE()
        `})
        channels = rows.map((r:any)=> r.channel).filter(Boolean)
      } catch {}
    }
    if (channels.length === 0) channels = ['google_ads']
    return NextResponse.json({ channels })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
