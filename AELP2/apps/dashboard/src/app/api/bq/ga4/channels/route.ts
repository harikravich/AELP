import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const days = Math.min(180, Math.max(1, Number(url.searchParams.get('days') || '28')))
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `
      SELECT default_channel_group, SUM(conversions) AS conversions
      FROM \`${projectId}.${dataset}.ga4_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()
      GROUP BY default_channel_group
      ORDER BY conversions DESC
      LIMIT 12
    `, params: { days } })
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}

