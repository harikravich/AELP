import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const days = parseInt(searchParams.get('days') || '28', 10)
    const minPayout = parseFloat(searchParams.get('min_payout') || '0')
    const limit = parseInt(searchParams.get('limit') || '20', 10)

    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    if (!projectId || !dataset) {
      return NextResponse.json({ error: 'Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET' }, { status: 500 })
    }
    const bq = new BigQuery({ projectId })

    const sql = `
      WITH base AS (
        SELECT date, partner_id, partner, payout, actions
        FROM \
          \`${projectId}.${dataset}.impact_partner_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      ), agg AS (
        SELECT partner_id, partner,
               SUM(payout) AS payout,
               SUM(actions) AS actions,
               COUNT(DISTINCT date) AS active_days
        FROM base
        GROUP BY partner_id, partner
      )
      SELECT * FROM agg
      WHERE payout >= @minPayout
      ORDER BY payout DESC
      LIMIT @limit
    `
    const options = {
      query: sql,
      params: { days, minPayout, limit },
    }
    const [rows] = await bq.query(options)

    // Summary
    const sumSql = `
      WITH base AS (
        SELECT date, payout, actions
        FROM \`${projectId}.${dataset}.impact_partner_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      )
      SELECT 
        COUNT(DISTINCT date) AS days,
        SUM(payout) AS total_payout,
        SUM(actions) AS total_actions
      FROM base
    `
    const [sumRows] = await bq.query({ query: sumSql, params: { days } })
    const summary = sumRows && sumRows[0] ? sumRows[0] : {}

    return NextResponse.json({ rows, summary })
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 })
  }
}

