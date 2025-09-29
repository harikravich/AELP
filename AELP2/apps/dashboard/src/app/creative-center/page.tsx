import React from 'react'
import { createBigQueryClient } from '../../lib/bigquery-client'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../lib/dataset'
import { fmtWhen } from '../../lib/utils'
import CreativeCenterClient from './CreativeCenterClient'
import MultiModalHub from './EnhancedCreativeCenter'

export const dynamic = 'force-dynamic'

async function fetchCreatives() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  const sql = `
    SELECT 
      DATE(date) AS date,
      campaign_id,
      IFNULL(campaign_name, CONCAT('Campaign ', campaign_id)) AS campaign_name,
      ad_group_id,
      IFNULL(ad_group_name, CONCAT('Ad Group ', ad_group_id)) AS ad_group_name,
      ad_id,
      IFNULL(ad_name, CONCAT('Ad ', ad_id)) AS ad_name,
      ANY_VALUE(customer_id) AS customer_id,
      SUM(impressions) AS impressions,
      SUM(clicks) AS clicks,
      SUM(cost_micros)/1e6 AS cost,
      SUM(conversions) AS conversions,
      SUM(conversion_value) AS revenue,
      SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
      SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr,
      SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac,
      SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS roas
    FROM \`${projectId}.${dataset}.ads_ad_performance\`
    WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    GROUP BY date, campaign_id, campaign_name, ad_group_id, ad_group_name, ad_id, ad_name
    ORDER BY date DESC, revenue DESC
    LIMIT 500`
  try {
    const [rows] = await bq.query({ query: sql })
    return { rows }
  } catch (e: any) {
    return { rows: [], error: e?.message || String(e) }
  }
}

async function fetchDecisions() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  try {
    const [rows] = await bq.query({ query: `
      SELECT timestamp, platform, channel, campaign_id, ad_id, prior_alpha, prior_beta, posterior_alpha, posterior_beta, sample
      FROM \`${projectId}.${dataset}.bandit_decisions\`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
      ORDER BY timestamp DESC
      LIMIT 50
    `})
    return { rows }
  } catch (e: any) {
    return { rows: [], error: e?.message || String(e) }
  }
}

export default async function CreativeCenter() {
  const data = await fetchCreatives()
  const rows = data.rows || []
  const dec = await fetchDecisions()
  const decisions = dec.rows || []
  return (
    <div className="space-y-6">
      {/* Multi-Modal Hub Section */}
      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-xl font-bold mb-4">🚀 AI Creative Generation Hub</h2>
        <MultiModalHub onGenerate={(type, data) => console.log('Generated:', type, data)} />
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Creative Library (Top Ads, 28d)</h2>
        <p className="text-sm text-gray-600">Aggregated from ads_ad_performance</p>
        {data.error && (<div className="text-xs text-red-300 mt-2">{String(data.error)}</div>)}
        <CreativeCenterClient rows={rows as any[]} />
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Bandit Decisions (Last 50)</h2>
        {dec.error && (<div className="text-xs text-red-600 mt-2">{String(dec.error)}</div>)}
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left border-b">
                <th className="py-2 pr-4">When</th>
                <th className="py-2 pr-4">Campaign</th>
                <th className="py-2 pr-4">Ad</th>
                <th className="py-2 pr-4">Posterior Mean</th>
                <th className="py-2 pr-4">95% CI</th>
                <th className="py-2 pr-4">Sample</th>
                <th className="py-2 pr-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((r: any, i: number) => {
                const a = Number(r.posterior_alpha || 1)
                const b = Number(r.posterior_beta || 1)
                const mean = a / Math.max(1, a + b)
                const variance = (a*b)/((a+b)**2*(a+b+1))
                const se = Math.sqrt(variance)
                const lo = Math.max(0, mean - 1.96*se)
                const hi = Math.min(1, mean + 1.96*se)
                return (
                  <tr key={`${r.timestamp}-${i}`} className="border-b last:border-0">
                    <td className="py-2 pr-4">{fmtWhen(r.timestamp)}</td>
                    <td className="py-2 pr-4">{r.campaign_id}</td>
                    <td className="py-2 pr-4">{r.ad_id}</td>
                    <td className="py-2 pr-4">{mean.toFixed(4)}</td>
                    <td className="py-2 pr-4">[{lo.toFixed(4)}, {hi.toFixed(4)}]</td>
                    <td className="py-2 pr-4">{Number(r.sample || 0).toFixed(5)}</td>
                    <td className="py-2 pr-4">
                      <form action="/api/control/bandit-approve" method="post" className="inline-block mr-2">
                        <input type="hidden" name="campaign_id" value={r.campaign_id} />
                        <input type="hidden" name="ad_id" value={r.ad_id} />
                        <input type="hidden" name="action" value="adjust_split" />
                        <input type="hidden" name="approved" value="true" />
                        <input type="hidden" name="approver" value="creative-center" />
                        <button className="px-2 py-1 text-xs rounded bg-emerald-600 text-white">Approve</button>
                      </form>
                      <form action="/api/control/bandit-apply" method="post" className="inline-block">
                        <input type="hidden" name="lookback" value="30" />
                        <button className="px-2 py-1 text-xs rounded bg-blue-600 text-white">Apply (orchestrator)</button>
                      </form>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

