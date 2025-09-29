import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

async function createOrReplaceView(bq: BigQuery, projectId: string, viewFqtn: string, sql: string) {
  // Prefer DDL to avoid table APIs across permissions
  const ddl = `CREATE OR REPLACE VIEW \`${viewFqtn}\` AS\n${sql}`
  await bq.query({ query: ddl })
}

export async function POST(req: Request) {
  // Safety: block writes on prod
  const { dataset, mode, allowed, reason } = resolveDatasetForAction('write')
  if (!allowed) {
    return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })
  }

  const url = new URL(req.url)
  const idsParam = (url.searchParams.get('ids') || '6453292723').trim()
  const idsList = idsParam.split(',').map(s => s.trim()).filter(Boolean)
  if (idsList.length === 0 || !idsList.every(x => /^\d{5,}$/.test(x))) {
    return NextResponse.json({ ok: false, error: 'Provide comma-separated numeric KPI IDs via ?ids=', hint: 'e.g., 6453292723' }, { status: 400 })
  }
  const idsCsv = idsList.map(x => `'${x}'`).join(',')

  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const bq = new BigQuery({ projectId })
  const datasetRef = `${projectId}.${dataset}`

  // Create KPI-only view
  const kpiSql = `
    WITH kpi AS (
      SELECT DATE(s.date) AS date,
             SUM(s.conversions) AS conversions,
             SUM(s.conversion_value) AS revenue
      FROM \`${datasetRef}.ads_conversion_action_stats\` s
      WHERE s.conversion_action_id IN (${idsCsv})
      GROUP BY date
    ),
    cost AS (
      SELECT DATE(date) AS date,
             SUM(cost_micros)/1e6 AS cost
      FROM \`${datasetRef}.ads_campaign_performance\`
      GROUP BY date
    )
    SELECT kpi.date,
           kpi.conversions,
           kpi.revenue,
           cost.cost,
           SAFE_DIVIDE(cost.cost, NULLIF(kpi.conversions,0)) AS cac,
           SAFE_DIVIDE(kpi.revenue, NULLIF(cost.cost,0)) AS roas
    FROM kpi
    LEFT JOIN cost USING(date)
    ORDER BY kpi.date`;

  // Basic episode + ads daily views for convenience
  const trainingSql = `
    SELECT DATE(timestamp) AS date,
           COUNT(*) AS episode_rows,
           SUM(steps) AS steps,
           SUM(auctions) AS auctions,
           SUM(wins) AS wins,
           SUM(spend) AS spend,
           SUM(revenue) AS revenue,
           SUM(conversions) AS conversions,
           SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) AS roas,
           SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) AS cac,
           AVG(win_rate) AS avg_win_rate
    FROM \`${datasetRef}.training_episodes\`
    GROUP BY date
    ORDER BY date`;

  const adsSql = `
    SELECT DATE(date) AS date,
           SUM(impressions) AS impressions,
           SUM(clicks) AS clicks,
           SUM(cost_micros)/1e6 AS cost,
           SUM(conversions) AS conversions,
           SUM(conversion_value) AS revenue,
           SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr,
           SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
           SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac,
           SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS roas,
           APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS impression_share_p50
    FROM \`${datasetRef}.ads_campaign_performance\`
    GROUP BY date
    ORDER BY date`;

  try {
    await createOrReplaceView(bq, projectId, `${datasetRef}.ads_kpi_daily`, kpiSql)
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: `Failed to create ads_kpi_daily: ${e?.message || String(e)}` }, { status: 500 })
  }

  // Best-effort for other views; ignore failures
  try { await createOrReplaceView(bq, projectId, `${datasetRef}.training_episodes_daily`, trainingSql) } catch {}
  try { await createOrReplaceView(bq, projectId, `${datasetRef}.ads_campaign_daily`, adsSql) } catch {}

  return NextResponse.json({ ok: true, kpi_ids: idsList, dataset })
}
