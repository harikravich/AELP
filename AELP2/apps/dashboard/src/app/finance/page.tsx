import React from 'react'
import { BigQuery } from '@google-cloud/bigquery'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../lib/dataset'
import Card from '../../components/Card'
import MMMSlider from '../../components/MMMSlider'
import PerChannelMMM from '../../components/PerChannelMMM'
import ValueUploads from '../../components/ValueUploads'
import ScenarioModeler from './ScenarioModeler'

async function fetchKpi() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  try {
    const [rows] = await bq.query({ query: `
      SELECT SUM(cost) AS cost, SUM(conversions) AS conv, SUM(revenue) AS revenue
      FROM \`${projectId}.${dataset}.ads_kpi_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    `})
    return rows?.[0] || { cost: 0, conv: 0, revenue: 0 }
  } catch {
    return { cost: 0, conv: 0, revenue: 0 }
  }
}

async function fetchLtv() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  try {
    const [rows] = await bq.query({ query: `
      SELECT date, AVG(ltv_90) AS ltv_90
      FROM \`${projectId}.${dataset}.ltv_priors_daily\`
      GROUP BY date ORDER BY date DESC LIMIT 1
    `})
    return rows?.[0] || null
  } catch { return null }
}

export default async function FinancePage() {
  const kpi = await fetchKpi()
  const ltv = await fetchLtv()
  const cost = Number(kpi.cost || 0)
  const conv = Number(kpi.conv || 0)
  const revenue = Number(kpi.revenue || 0)
  const cac = conv ? cost / conv : 0
  const aov = conv ? revenue / conv : 0
  const ltv90 = Number((ltv?.ltv_90 || 0))
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card><div className="card-header">Spend (28d)</div><div className="card-value">${cost.toFixed(2)}</div></Card>
        <Card><div className="card-header">Signups (28d)</div><div className="card-value">{conv.toFixed(0)}</div></Card>
        <Card><div className="card-header">CAC</div><div className="card-value">${cac.toFixed(2)}</div></Card>
        <Card><div className="card-header">AOV</div><div className="card-value">${aov.toFixed(2)}</div></Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card><div className="card-header">LTV (90d prior)</div><div className="card-value">{ltv90 ? `$${ltv90.toFixed(2)}` : '—'}</div></Card>
        <Card footer="(illustrative; refine with margin + retention)"><div className="card-header">Payback (rough, days)</div><div className="card-value">{(ltv90 && cac ? (cac / (aov || 1) * 30).toFixed(0) : '—')}</div></Card>
      </div>

      <Card title="MMM What‑If (Daily Budget → Conversions)"><MMMSlider /></Card>

      <Card title="MMM What‑If Per Channel" subtitle="Adjust budget by channel to see expected conversions/CAC"><PerChannelMMM /></Card>

      <Card title="Advanced Scenario Modeling" subtitle="Model complex what-if scenarios with ML-powered projections"><ScenarioModeler /></Card>

      <Card title="Value Uploads" subtitle="Create an audited value adjustment request (safely gated)"><ValueUploads /></Card>
    </div>
  )
}

export const dynamic = 'force-dynamic'
