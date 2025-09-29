import React from 'react'
import { BigQuery } from '@google-cloud/bigquery'
import Card from '../../components/Card'
import { fmtWhen } from '../../lib/utils'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../lib/dataset'

async function fetchMMM() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  try {
    const [rows] = await bq.query({ query: `
      SELECT timestamp, channel, proposed_daily_budget, expected_conversions, expected_cac
      FROM \`${projectId}.${dataset}.mmm_allocations\`
      ORDER BY timestamp DESC LIMIT 1
    `})
    return rows?.[0] || null
  } catch {
    return null
  }
}

async function fetchBandits() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  try {
    const [rows] = await bq.query({ query: `
      SELECT timestamp, platform, campaign_id, ad_id, action, exploration_pct, reason
      FROM \`${projectId}.${dataset}.bandit_change_proposals\`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      ORDER BY timestamp DESC LIMIT 20
    `})
    return rows || []
  } catch {
    return []
  }
}

async function fetchOpportunities() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  try {
    const [rows] = await bq.query({ query: `
      SELECT timestamp, platform, campaign_name, objective, daily_budget, status, notes
      FROM \`${projectId}.${dataset}.platform_skeletons\`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      ORDER BY timestamp DESC LIMIT 20
    `})
    return rows || []
  } catch {
    return []
  }
}

export default async function GrowthLab() {
  const mmm = await fetchMMM()
  const bandits = await fetchBandits()
  const opps = await fetchOpportunities()
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card><div className="card-header">MMM Target (Daily)</div><div className="card-value">{mmm ? `$${Number(mmm.proposed_daily_budget||0).toFixed(2)}` : '—'}</div><div className="text-xs text-slate-500 mt-1">Expected CAC: {mmm ? `$${Number(mmm.expected_cac||0).toFixed(2)}` : '—'}</div></Card>
        <Card><div className="card-header">Bandit Suggestions (Last 30d)</div><div className="card-value">{bandits.length}</div></Card>
        <Card><div className="card-header">Growth Ideas</div><div className="card-value">{opps.length}</div></Card>
      </div>

      <Card title="Bandit/Experiment Proposals">
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead><tr className="text-left border-b"><th className="py-2 pr-4">When</th><th className="py-2 pr-4">Campaign</th><th className="py-2 pr-4">Ad</th><th className="py-2 pr-4">Action</th><th className="py-2 pr-4">Exploration%</th><th className="py-2 pr-4">Reason</th></tr></thead>
            <tbody>
              {bandits.map((r: any, i: number)=> (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-2 pr-4">{fmtWhen(r.timestamp)}</td>
                  <td className="py-2 pr-4">{r.campaign_id}</td>
                  <td className="py-2 pr-4">{r.ad_id}</td>
                  <td className="py-2 pr-4">{r.action}</td>
                  <td className="py-2 pr-4">{Number(r.exploration_pct||0).toFixed(2)}</td>
                  <td className="py-2 pr-4">{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="New Growth Ideas (Scanner)">
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead><tr className="text-left border-b"><th className="py-2 pr-4">When</th><th className="py-2 pr-4">Platform</th><th className="py-2 pr-4">Objective</th><th className="py-2 pr-4">Name</th><th className="py-2 pr-4">Notes</th></tr></thead>
            <tbody>
              {opps.map((r:any,i:number)=> (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-2 pr-4">{fmtWhen(r.timestamp)}</td>
                  <td className="py-2 pr-4">{r.platform}</td>
                  <td className="py-2 pr-4">{r.objective}</td>
                  <td className="py-2 pr-4">{r.campaign_name}</td>
                  <td className="py-2 pr-4">{r.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

export const dynamic = 'force-dynamic'
