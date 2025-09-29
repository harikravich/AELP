import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { resolveDatasetForAction } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function POST() {
  const { dataset, mode, allowed, reason } = resolveDatasetForAction('write')
  if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })

  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const bq = new BigQuery({ projectId })
  const ds = `${projectId}.${dataset}`

  // Ensure table exists
  try {
    await bq.query({ query: `CREATE TABLE IF NOT EXISTS \`${ds}.fidelity_evaluations\` (
      timestamp TIMESTAMP NOT NULL,
      start_date DATE NOT NULL,
      end_date DATE NOT NULL,
      mape_roas FLOAT64,
      rmse_roas FLOAT64,
      mape_cac FLOAT64,
      rmse_cac FLOAT64,
      ks_winrate_vs_impressionshare FLOAT64,
      passed BOOL NOT NULL,
      details JSON
    ) PARTITION BY DATE(timestamp)` })
  } catch {}

  const sql = `
    WITH bounds AS (
      SELECT DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AS start_date, CURRENT_DATE() AS end_date
    ),
    rl AS (
      SELECT DATE(timestamp) AS date,
             SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) AS roas,
             SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) AS cac
      FROM \`${ds}.training_episodes\`
      WHERE DATE(timestamp) BETWEEN (SELECT start_date FROM bounds) AND (SELECT end_date FROM bounds)
      GROUP BY date
    ),
    ads AS (
      SELECT date, roas, cac
      FROM \`${ds}.ads_kpi_daily\`
      WHERE date BETWEEN (SELECT start_date FROM bounds) AND (SELECT end_date FROM bounds)
    ),
    joined AS (
      SELECT a.date,
             a.roas AS ads_roas, r.roas AS rl_roas,
             a.cac AS ads_cac, r.cac AS rl_cac
      FROM ads a
      JOIN rl r USING(date)
    )
    SELECT (SELECT start_date FROM bounds) AS start_date,
           (SELECT end_date FROM bounds) AS end_date,
           AVG(ABS(SAFE_DIVIDE(ads_roas - rl_roas, NULLIF(ads_roas,0)))) AS mape_roas,
           SQRT(AVG(POW(ads_roas - rl_roas, 2))) AS rmse_roas,
           AVG(ABS(SAFE_DIVIDE(ads_cac - rl_cac, NULLIF(ads_cac,0)))) AS mape_cac,
           SQRT(AVG(POW(ads_cac - rl_cac, 2))) AS rmse_cac
    FROM joined`;

  try {
    const [rows] = await bq.query({ query: sql })
    const r:any = rows?.[0]
    if (!r) return NextResponse.json({ ok: false, error: 'No joined rows. Ensure ads_kpi_daily exists and training_episodes has rows.' }, { status: 400 })
    const start_date = r.start_date
    const end_date = r.end_date
    const mape_roas = Number(r.mape_roas ?? null)
    const rmse_roas = Number(r.rmse_roas ?? null)
    const mape_cac = Number(r.mape_cac ?? null)
    const rmse_cac = Number(r.rmse_cac ?? null)
    const details = { method: 'kpi_only_node', days: 14 }
    await bq.query({
      query: `INSERT \`${ds}.fidelity_evaluations\` (timestamp, start_date, end_date, mape_roas, rmse_roas, mape_cac, rmse_cac, ks_winrate_vs_impressionshare, passed, details)
              VALUES (CURRENT_TIMESTAMP(), @start_date, @end_date, @mape_roas, @rmse_roas, @mape_cac, @rmse_cac, NULL, TRUE, TO_JSON(@details))`,
      params: { start_date, end_date, mape_roas, rmse_roas, mape_cac, rmse_cac, details },
    })
    return NextResponse.json({ ok: true, dataset, metrics: { start_date, end_date, mape_roas, rmse_roas, mape_cac, rmse_cac } })
  } catch (e:any) {
    const msg = e?.message || String(e)
    if (/Not found: Table .*ads_kpi_daily/i.test(msg)) {
      return NextResponse.json({ ok: false, error: 'ads_kpi_daily view not found. Run KPI lock first.' }, { status: 400 })
    }
    return NextResponse.json({ ok: false, error: msg }, { status: 500 })
  }
}
