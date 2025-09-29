import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../../../../lib/dataset'
import { createBigQueryClient } from '../../../../../../lib/bigquery-client'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)
    const [rows] = await bq.query({ query: `
      WITH g AS (
        SELECT month, enrollments, d2p_starts, post_trial_subs, mobile_subs
        FROM \`${projectId}.${dataset}.ga4_derived_monthly\`
      ), p AS (
        SELECT DATE_TRUNC(date, MONTH) AS month,
               SUM(d2c_total_subscribers) AS pacer_d2c,
               SUM(COALESCE(d2p_starts,0)) AS pacer_d2p,
               SUM(COALESCE(post_trial_subscribers,0)) AS pacer_post,
               SUM(COALESCE(mobile_subscribers,0)) AS pacer_mobile
        FROM \`${projectId}.${dataset}.pacing_pacer_daily\`
        GROUP BY month
      )
      SELECT g.month,
             g.enrollments AS ga_d2c,
             p.pacer_d2c,
             g.d2p_starts AS ga_d2p,
             p.pacer_d2p,
             g.post_trial_subs AS ga_post,
             p.pacer_post,
             g.mobile_subs AS ga_mobile,
             p.pacer_mobile
      FROM g LEFT JOIN p USING(month)
      ORDER BY g.month
    ` })
    return NextResponse.json({ rows })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message||String(e) }, { status: 500 })
  }
}
