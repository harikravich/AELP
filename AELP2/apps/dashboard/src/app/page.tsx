import React from 'react'
import { createBigQueryClient } from '../lib/bigquery-client'
import { fmtUSD, fmtFloat } from '../lib/utils'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../lib/dataset'
import { getKpiSourceFromCookie } from './lib/kpi'
import { absoluteUrl } from './lib/url'

export const dynamic = 'force-dynamic'

async function fetchKpis() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  const src = getKpiSourceFromCookie()
  try {
    if (src === 'ads') {
      const [rows] = await bq.query({ query: `
        WITH cur AS (
          SELECT SUM(cost) AS cost, SUM(conversions) AS conv, SUM(revenue) AS revenue
          FROM \`${projectId}.${dataset}.ads_kpi_daily\`
          WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
        ), prev AS (
          SELECT SUM(cost) AS cost, SUM(conversions) AS conv, SUM(revenue) AS revenue
          FROM \`${projectId}.${dataset}.ads_kpi_daily\`
          WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)
        )
        SELECT * FROM cur, prev` })
      const r:any = rows?.[0] || {}
      return { cost: r.cost, conv: r.conv, revenue: r.revenue, prev_cost: r.cost_1, prev_conv: r.conv_1, prev_revenue: r.revenue_1, source: 'ads' }
    }
    const enrollTable = src==='ga4_google' ? 'ga4_enrollments_google_cpc_session_daily' : 'ga4_enrollments_daily'
    const [rows] = await bq.query({ query: `
      WITH cur AS (
        SELECT 
          (SELECT SUM(cost) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()) AS cost,
          (SELECT SUM(enrollments) FROM \`${projectId}.${dataset}.${enrollTable}\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()) AS conv,
          (SELECT SUM(revenue) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()) AS revenue
      ), prev AS (
        SELECT 
          (SELECT SUM(cost) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)) AS cost,
          (SELECT SUM(enrollments) FROM \`${projectId}.${dataset}.${enrollTable}\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)) AS conv,
          (SELECT SUM(revenue) FROM \`${projectId}.${dataset}.ads_campaign_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 56 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 29 DAY)) AS revenue
      )
      SELECT cur.cost, cur.conv, cur.revenue, prev.cost AS prev_cost, prev.conv AS prev_conv, prev.revenue AS prev_revenue FROM cur, prev` })
    const r:any = rows?.[0] || {}
    return { cost: r.cost||0, conv: r.conv||0, revenue: r.revenue||0, prev_cost: r.prev_cost||0, prev_conv: r.prev_conv||0, prev_revenue: r.prev_revenue||0, source: src }
  } catch { return { cost: 0, conv: 0, revenue: 0, source: 'ads' } as any }
}

async function fetchHeadroom() {
  const r = await fetch(absoluteUrl('/api/bq/headroom'), { cache: 'no-store' })
  const j = await r.json().catch(()=>({ rows: [] }))
  return j.rows || []
}

async function fetchApprovals() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
  const bq = createBigQueryClient(projectId)
  try {
    const [rows] = await bq.query({ query: `SELECT run_id, platform, type, enqueued_at FROM \`${projectId}.${dataset}.creative_publish_queue\` WHERE status='queued' ORDER BY enqueued_at DESC LIMIT 5` })
    return rows as any[]
  } catch { return [] }
}

export default async function OverviewPage() {
  const [kpi, headroom, approvals] = await Promise.all([fetchKpis(), fetchHeadroom(), fetchApprovals()])
  const cost = Number(kpi.cost||0)
  const conv = Number(kpi.conv||0)
  const revenue = Number(kpi.revenue||0)
  const cac = conv ? cost/conv : 0
  const roas = cost ? revenue/cost : 0
  const d = (a:number,b:number)=> (b? ((a-b)/Math.abs(b))*100 : 0)
  const costDelta = d(cost, Number((kpi as any).prev_cost||0))
  const convDelta = d(conv, Number((kpi as any).prev_conv||0))
  const cacPrev = Number((kpi as any).prev_cost||0) / Math.max(1, Number((kpi as any).prev_conv||0))
  const cacDelta = d(cac, cacPrev)
  return (
    <div className="space-y-6">
      {/* KPI Source Notice */}
      <div className="glass-card p-4">
        <div className="text-xs text-white/70">
          KPI Source: {kpi.source==='ads' ? 'Ads conversions' : kpi.source==='ga4_all' ? 'GA4 Sitewide Purchases' : 'GA4 Purchases (google / cpc, session)'}
          {kpi.source!=='ads' && ' — CAC uses Google Ads spend / GA4 purchases.'}
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="glass-card p-4"><div className="text-xs text-white/60">New Customers (28d)</div><div className="text-xl font-semibold">{Math.round(conv).toLocaleString('en-US')}</div><div className={`text-[11px] mt-1 ${convDelta>=0?'text-emerald-300':'text-rose-300'}`}>{convDelta>=0?'+':''}{fmtFloat(convDelta,1)}% vs prior 28d</div></div>
        <div className="glass-card p-4"><div className="text-xs text-white/60">Spend (28d)</div><div className="text-xl font-semibold">{fmtUSD(cost,0)}</div><div className={`text-[11px] mt-1 ${costDelta>=0?'text-emerald-300':'text-rose-300'}`}>{costDelta>=0?'+':''}{fmtFloat(costDelta,1)}% vs prior 28d</div></div>
        <div className="glass-card p-4"><div className="text-xs text-white/60">CAC</div><div className="text-xl font-semibold">{fmtUSD(cac,0)}</div><div className={`text-[11px] mt-1 ${cacDelta<=0?'text-emerald-300':'text-rose-300'}`}>{cacDelta>=0?'+':''}{fmtFloat(cacDelta,1)}% vs prior 28d • {kpi.source==='ads' ? 'Ads conversions' : 'GA4 purchases'}</div></div>
        <div className="glass-card p-4"><div className="text-xs text-white/60">ROAS</div><div className="text-xl font-semibold">{fmtFloat(roas,2)}x</div></div>
        <div className="glass-card p-4"><div className="text-xs text-white/60">Revenue (28d)</div><div className="text-xl font-semibold">{fmtUSD(revenue,0)}</div></div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card p-4">
          <div className="text-sm font-semibold mb-2">Where to put the next $5,000/day</div>
          {headroom.length === 0 ? <div className="text-sm text-white/60">No headroom rows yet.</div> : (
            <table className="w-full text-sm">
              <thead><tr className="text-left border-b border-white/10"><th className="py-2">Channel</th><th>Room $/day</th><th>Extra cust/day</th><th>Action</th></tr></thead>
              <tbody>
                {headroom.slice(0,4).map((r:any,i:number)=> (
                  <tr key={i} className="border-b border-white/10">
                    <td className="py-2">{r.channel}</td>
                    <td>${Number(r.room||0).toFixed(0)}</td>
                    <td>{Number(r.extra_per_day||0)}</td>
                    <td><a className="text-indigo-300" href="/spend-planner">Review</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="glass-card p-4">
          <div className="text-sm font-semibold mb-2">Approvals Queue</div>
          {approvals.length === 0 ? <div className="text-sm text-white/60">No pending items.</div> : (
            <ul className="text-sm list-disc pl-5">
              {approvals.map((r:any,i:number)=> (
                <li key={i}>{r.platform}/{r.type} — {r.enqueued_at} <a className="text-indigo-300 ml-2" href="/approvals">Open</a></li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="glass-card p-4">
        <div className="text-sm text-white/70">Tip: Use Chat to ask “Where can I add $5k/day under $80 CAC?” then [Pin] the chart to Canvas.</div>
      </div>
    </div>
  )
}
