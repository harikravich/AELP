import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const days = parseInt(searchParams.get('days') || '28', 10)
    const limit = parseInt(searchParams.get('limit') || '20', 10)
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    if (!projectId || !dataset) throw new Error('Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET')
    const bq = new BigQuery({ projectId })
    const topSql = `
      WITH base AS (
        SELECT date, partner_id, partner, triggered_conversions
        FROM \`${projectId}.${dataset}.ga_affiliate_triggered_by_partner_daily\`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      )
      SELECT partner_id, ANY_VALUE(partner) AS partner, SUM(triggered_conversions) AS triggered
      FROM base GROUP BY partner_id ORDER BY triggered DESC LIMIT @limit
    `
    const [rows] = await bq.query({ query: topSql, params: { days, limit } })

    const sumSql = `
      SELECT COUNT(DISTINCT date) AS days, SUM(triggered_conversions) AS total
      FROM \`${projectId}.${dataset}.ga_affiliate_triggered_by_partner_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    `
    const [sumRows] = await bq.query({ query: sumSql, params: { days } })
    const summary = (sumRows as any[])[0] || {}
    return NextResponse.json({ rows, summary })
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 })
  }
}

