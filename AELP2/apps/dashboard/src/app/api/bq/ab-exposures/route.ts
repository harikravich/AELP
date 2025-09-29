import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const sql = `
      SELECT experiment, variant, COUNT(*) AS exposures
      FROM \`${projectId}.${dataset}.ab_exposures\`
      WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
      GROUP BY experiment, variant
      ORDER BY exposures DESC
    `
    const [rows] = await bq.query({ query: sql })
    return NextResponse.json({ ok: true, rows })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
