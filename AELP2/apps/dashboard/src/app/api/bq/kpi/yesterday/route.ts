import { NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../../lib/bigquery-client'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../lib/dataset'
import { getKpiSourceFromCookie } from '../../../../lib/kpi'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)
    const src = getKpiSourceFromCookie()
    if (src === 'pacer') {
      const [rows] = await bq.query({ query: `
        SELECT CAST(DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) AS STRING) AS date,
               SUM(spend) AS cost,
               SUM(d2c_total_subscribers) AS conversions,
               NULL AS revenue,
               SAFE_DIVIDE(SUM(spend), NULLIF(SUM(d2c_total_subscribers),0)) AS cac
        FROM \`${projectId}.${dataset}.pacing_pacer_daily\`
        WHERE date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      `})
      const r:any = rows?.[0] || null
      return NextResponse.json({ row: r })
    }
    const [rows] = await bq.query({ query: `
        SELECT CAST(DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) AS STRING) AS date,
               SUM(cost) AS cost,
               SUM(conversions) AS conversions,
               SUM(revenue) AS revenue,
               SAFE_DIVIDE(SUM(cost), NULLIF(SUM(conversions),0)) AS cac
        FROM \`${projectId}.${dataset}.ads_kpi_daily\`
        WHERE date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      `})
    const r: any = rows?.[0] || null
    return NextResponse.json({ row: r })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
