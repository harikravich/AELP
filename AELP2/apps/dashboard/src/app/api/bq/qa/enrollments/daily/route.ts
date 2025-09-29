import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../../lib/dataset'
import { createBigQueryClient } from '../../../../../../lib/bigquery-client'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const days = Math.min(90, Math.max(1, Number(url.searchParams.get('days') || '30')))
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `
      WITH ga AS (
        SELECT date, SUM(enrollments) AS ga_enrollments
        FROM \`${projectId}.${dataset}.ga4_derived_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()
        GROUP BY date
      ), pacer AS (
        SELECT date, SUM(d2c_total_subscribers) AS pacer_enrollments
        FROM \`${projectId}.${dataset}.pacing_pacer_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()
        GROUP BY date
      )
      SELECT d.date,
             p.pacer_enrollments,
             g.ga_enrollments,
             SAFE_DIVIDE(g.ga_enrollments, NULLIF(p.pacer_enrollments,0)) AS ga_to_pacer_ratio
      FROM (SELECT DISTINCT date FROM ga UNION DISTINCT SELECT date FROM pacer) d
      LEFT JOIN pacer p USING(date)
      LEFT JOIN ga g USING(date)
      ORDER BY d.date
    `, params: { days } })
    return NextResponse.json({ rows })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message||String(e) }, { status: 500 })
  }
}
