import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../lib/dataset'
import { getKpiSourceFromCookie } from '../../../../lib/kpi'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const days = Math.min(90, Math.max(1, Number(url.searchParams.get('days') || '7')))
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = new BigQuery({ projectId })
    const src = getKpiSourceFromCookie()
    if (src === 'pacer') {
      const [rows] = await bq.query({ query: `
        SELECT date,
               SUM(spend) AS cost,
               SUM(d2c_total_subscribers) AS conversions,
               NULL AS revenue,
               SAFE_DIVIDE(SUM(spend), NULLIF(SUM(d2c_total_subscribers),0)) AS cac
        FROM \`${projectId}.${dataset}.pacing_pacer_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()
        GROUP BY date
        ORDER BY date ASC
      `, params: { days } })
      return NextResponse.json({ rows })
    }
    const [rows] = await bq.query({ query: `
        SELECT date, cost, conversions, revenue,
               SAFE_DIVIDE(cost, NULLIF(conversions,0)) AS cac
        FROM \`${projectId}.${dataset}.ads_kpi_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()
        ORDER BY date ASC
      `, params: { days } })
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
