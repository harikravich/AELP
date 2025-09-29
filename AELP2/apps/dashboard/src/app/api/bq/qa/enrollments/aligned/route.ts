import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../../lib/dataset'
import { createBigQueryClient } from '../../../../../../lib/bigquery-client'

export const dynamic = 'force-dynamic'

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const days = Math.min(180, Math.max(1, Number(url.searchParams.get('days') || '90')))
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `
      SELECT date,
             ga_enrollments,
             pacer_subs,
             non_ga_delta,
             aligned_enrollments,
             spend,
             SAFE_DIVIDE(spend, NULLIF(aligned_enrollments,0)) AS cac
      FROM \`${projectId}.${dataset}.ga_aligned_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND CURRENT_DATE()
      ORDER BY date
    `, params: { days } })
    return NextResponse.json({ rows })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message||String(e) }, { status: 500 })
  }
}

