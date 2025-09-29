import React from 'react'
import { TimeSeriesChart } from '../../components/TimeSeriesChart'
import dayjs from 'dayjs'
import { BigQuery } from '@google-cloud/bigquery'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../lib/dataset'

async function fetchEpisodes() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  const sql = `
    SELECT DATE(timestamp) AS date,
           SUM(spend) AS spend,
           SUM(revenue) AS revenue,
           SUM(conversions) AS conversions,
           SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) AS roas,
           SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) AS cac,
           AVG(win_rate) AS avg_win_rate
    FROM \`${projectId}.${dataset}.training_episodes\`
    WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    GROUP BY date
    ORDER BY date DESC`
  const [rows] = await bq.query({ query: sql })
  return { rows }
}

export default async function TrainingCenter() {
  const data = await fetchEpisodes()
  // Fetch latest fidelity evaluations (best-effort)
  let fidelity: any = { rows: [] }
  let safety: any = { rows: [] }
  let fresh: any = { rows: [] }
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = new BigQuery({ projectId })
    const sql = `SELECT * FROM \`${projectId}.${dataset}.fidelity_evaluations\` ORDER BY timestamp DESC LIMIT 1`
    const [rows] = await bq.query({ query: sql })
    fidelity = { rows }
  } catch {}
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = new BigQuery({ projectId })
    const sql = `
      WITH last AS (
        SELECT * FROM \`${projectId}.${dataset}.safety_events\`
        WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
      )
      SELECT 'timeline' AS section, TO_JSON_STRING(x) AS payload FROM (
        SELECT DATE(timestamp) AS date, severity, COUNT(*) AS events
        FROM last GROUP BY date, severity ORDER BY date
      ) x
      UNION ALL
      SELECT 'latest_critical' AS section, TO_JSON_STRING(x) AS payload FROM (
        SELECT * FROM last WHERE severity IN ('HIGH','CRITICAL') ORDER BY timestamp DESC LIMIT 20
      ) x`
    const [rows] = await bq.query({ query: sql })
    safety = { rows }
  } catch {}
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = new BigQuery({ projectId })
    const sql = `
      SELECT 'training_episodes' AS table_name, MAX(DATE(timestamp)) AS max_date FROM \`${projectId}.${dataset}.training_episodes\`
      UNION ALL
      SELECT 'ads_campaign_performance', MAX(DATE(date)) FROM \`${projectId}.${dataset}.ads_campaign_performance\`
      UNION ALL
      SELECT 'fidelity_evaluations', MAX(DATE(timestamp)) FROM \`${projectId}.${dataset}.fidelity_evaluations\``
    const [rows] = await bq.query({ query: sql })
    fresh = { rows }
  } catch {}
  const rows = (data.rows || []).slice().reverse()
  const dateFmt = (s: string) => dayjs(s).format('MM-DD')
  const series = rows.map((r: any) => ({
    date: dateFmt(r.date),
    spend: Number(r.spend || 0),
    revenue: Number(r.revenue || 0),
    roas: Number(r.roas || 0),
    cac: Number(r.cac || 0),
    win_rate: Number(r.avg_win_rate || 0),
  }))
  const seriesPlain = JSON.parse(JSON.stringify(series))
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded shadow-sm p-4">
          <div className="text-xs text-gray-500">Avg Win Rate (28d)</div>
          <div className="text-xl font-semibold">{(series.reduce((s: number, a: any)=>s + (a.win_rate || 0), 0)/Math.max(series.length,1)).toFixed(3)}</div>
        </div>
        <div className="bg-white rounded shadow-sm p-4">
          <div className="text-xs text-gray-500">Total Spend (28d)</div>
          <div className="text-xl font-semibold">{`$${series.reduce((s: number, a: any)=>s + (a.spend || 0), 0).toFixed(2)}`}</div>
        </div>
        <div className="bg-white rounded shadow-sm p-4">
          <div className="text-xs text-gray-500">Avg CAC</div>
          <div className="text-xl font-semibold">{(series.reduce((s: number, a: any)=>s + (a.cac || 0), 0)/Math.max(series.length,1)).toFixed(2)}</div>
        </div>
        <div className="bg-white rounded shadow-sm p-4">
          <div className="text-xs text-gray-500">Avg ROAS</div>
          <div className="text-xl font-semibold">{(series.reduce((s: number, a: any)=>s + (a.roas || 0), 0)/Math.max(series.length,1)).toFixed(2)}</div>
        </div>
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Training Metrics (Win Rate, CAC, ROAS, Spend)</h2>
        <div className="mt-4">
          <TimeSeriesChart
            data={seriesPlain}
            series={[
              { name: 'Win Rate', dataKey: 'win_rate', color: '#14b8a6', yAxisId: 'left' },
              { name: 'CAC', dataKey: 'cac', color: '#ef4444', yAxisId: 'left' },
              { name: 'ROAS', dataKey: 'roas', color: '#0ea5e9', yAxisId: 'left' },
              { name: 'Spend', dataKey: 'spend', color: '#6366f1', yAxisId: 'right' },
            ]}
          />
        </div>
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Fidelity (latest)</h2>
        <p className="text-sm text-gray-600">Sim vs IRL alignment (KPI-only mode)</p>
        <div className="mt-3 text-sm">
          {fidelity.rows && fidelity.rows.length > 0 ? (
            <pre className="text-xs bg-gray-50 rounded p-3 overflow-x-auto">{JSON.stringify(fidelity.rows[0], null, 2)}</pre>
          ) : (
            <div className="text-gray-500">No recent fidelity rows</div>
          )}
        </div>
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Safety Timeline (14d)</h2>
        <p className="text-sm text-gray-600">Events by severity; latest critical below</p>
        <div className="mt-3 text-sm">
          <pre className="text-xs bg-gray-50 rounded p-3 overflow-x-auto">{JSON.stringify(safety.rows || [], null, 2)}</pre>
        </div>
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Data Freshness</h2>
        <div className="mt-3 text-sm">
          <pre className="text-xs bg-gray-50 rounded p-3 overflow-x-auto">{JSON.stringify(fresh.rows || [], null, 2)}</pre>
        </div>
      </div>
    </div>
  )
}
export const dynamic = 'force-dynamic'
