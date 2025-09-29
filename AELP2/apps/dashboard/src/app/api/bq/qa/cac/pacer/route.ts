import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../../lib/dataset'
import { createBigQueryClient } from '../../../../../../lib/bigquery-client'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const days = Math.min(120, Math.max(1, Number(url.searchParams.get('days') || '30')))
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `
      SELECT date, spend, subscribers, cac
      FROM \`${projectId}.${dataset}.kpi_pacer_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND DATE_ADD(CURRENT_DATE(), INTERVAL 365 DAY)
      ORDER BY date
    `, params: { days } })
    return NextResponse.json({ rows })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message||String(e) }, { status: 500 })
  }
}

