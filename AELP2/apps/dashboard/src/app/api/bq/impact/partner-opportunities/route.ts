import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const days = parseInt(searchParams.get('days') || '60', 10)
    const minPayout = parseFloat(searchParams.get('min_payout') || '1000')
    const minDays = parseInt(searchParams.get('min_days') || '14', 10)
    const corrThresh = parseFloat(searchParams.get('corr') || '0.25')
    const limit = parseInt(searchParams.get('limit') || '10', 10)

    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    if (!projectId || !dataset) throw new Error('Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET')
    const bq = new BigQuery({ projectId })

    const sql = `
      DECLARE lookback INT64 DEFAULT @days;
      WITH base AS (
        SELECT date, SAFE_CAST(partner_id AS STRING) AS partner_id, partner, payout
        FROM \`${projectId}.${dataset}.impact_partner_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL lookback DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      ), ga AS (
        SELECT date, SUM(enrollments) AS enrollments
        FROM \`${projectId}.${dataset}.ga4_derived_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL lookback DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
        GROUP BY date
      ), lagged AS (
        SELECT b1.partner_id, ANY_VALUE(b1.partner) AS partner, lag_val AS lag,
               COUNT(*) AS n_days,
               CORR(b1.payout, ga2.enrollments) AS corr,
               SUM(b1.payout) AS payout_sum
        FROM base b1
        JOIN ga ga1 USING(date)
        CROSS JOIN UNNEST([0,1,2,3,4,5,6,7]) AS lag_val
        JOIN ga ga2 ON ga2.date = DATE_ADD(b1.date, INTERVAL lag_val DAY)
        GROUP BY partner_id, lag
      ), totals AS (
        SELECT partner_id, ANY_VALUE(partner) AS partner, SUM(payout_sum) AS payout_total
        FROM lagged
        GROUP BY partner_id
      ), best AS (
        SELECT l.partner_id, ANY_VALUE(l.partner) AS partner,
               ARRAY_AGG(STRUCT(lag, corr, n_days) ORDER BY ABS(corr) DESC LIMIT 1)[OFFSET(0)] AS best
        FROM lagged l
        GROUP BY l.partner_id
      )
      SELECT b.partner_id, b.partner, b.best.lag AS best_lag, b.best.corr AS corr,
             b.best.n_days AS days, t.payout_total AS payout
      FROM best b
      JOIN totals t USING(partner_id)
      WHERE best.n_days >= @minDays AND payout_total >= @minPayout
    `
    const params = { days, minDays, minPayout }
    const [rows] = await bq.query({ query: sql, params })

    const scale = (rows as any[])
      .filter(r => r.corr >= corrThresh)
      .sort((a, b) => (b.corr * b.payout) - (a.corr * a.payout))
      .slice(0, limit)
    const trim = (rows as any[])
      .filter(r => r.corr <= -corrThresh)
      .sort((a, b) => Math.abs(b.corr) * b.payout - Math.abs(a.corr) * a.payout)
      .slice(0, limit)

    const summarySql = `
      WITH base AS (
        SELECT date, payout FROM \`${projectId}.${dataset}.impact_partner_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      )
      SELECT COUNT(DISTINCT date) AS days, SUM(payout) AS payout
      FROM base
    `
    const [sumRows] = await bq.query({ query: summarySql, params: { days } })
    const summary = sumRows && (sumRows as any[])[0] ? (sumRows as any[])[0] : {}

    return NextResponse.json({ summary, scale, trim })
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 })
  }
}
