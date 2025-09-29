import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const partnerId = searchParams.get('partner_id')
    const days = parseInt(searchParams.get('days') || '60', 10)
    if (!partnerId) return NextResponse.json({ error: 'partner_id required' }, { status: 400 })

    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    if (!projectId || !dataset) throw new Error('Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET')
    const bq = new BigQuery({ projectId })

    const sql = `
      DECLARE lookback INT64 DEFAULT @days;
      WITH p AS (
        SELECT date, payout
        FROM \`${projectId}.${dataset}.impact_partner_daily\`
        WHERE SAFE_CAST(partner_id AS STRING) = @partnerId
          AND date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL lookback DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      ), g AS (
        SELECT date, SUM(enrollments) AS enrollments
        FROM \`${projectId}.${dataset}.ga4_derived_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL lookback DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
        GROUP BY date
      ), lags AS (
        SELECT lag_val AS lag,
               COUNT(*) AS n_days,
               CORR(p.payout, g2.enrollments) AS corr,
               SUM(p.payout) AS payout_sum
        FROM p
        CROSS JOIN UNNEST([0,1,2,3,4,5,6,7]) AS lag_val
        JOIN g g2 ON g2.date = DATE_ADD(p.date, INTERVAL lag_val DAY)
        GROUP BY lag
      )
      SELECT * FROM lags ORDER BY lag
    `
    const [rows] = await bq.query({ query: sql, params: { days, partnerId } })
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 })
  }
}

