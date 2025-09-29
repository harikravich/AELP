import React from 'react'
import { createBigQueryClient } from '../../lib/bigquery-client'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../lib/dataset'

export const dynamic = 'force-dynamic'

async function fetchLpTests() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  try {
    const [rows] = await bq.query({ query: `
      SELECT created_at, test_id, lp_a, lp_b, status, traffic_split, primary_metric
      FROM \`${projectId}.${dataset}.lp_tests\`
      ORDER BY created_at DESC
      LIMIT 100
    ` })
    return rows as any[]
  } catch { return [] }
}

export default async function LandingPages() {
  const tests = await fetchLpTests()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Landing Pages</h1>
      <div className="glass-card p-4">
        <div className="text-sm font-semibold mb-2">A/B Tests</div>
        {tests.length === 0 ? (
          <div className="text-sm text-white/70">No LP tests yet. Use Control → LP Publish or POST /api/control/lp/publish.</div>
        ) : (
          <table className="w-full text-sm">
            <thead><tr className="text-left border-b border-white/10"><th className="py-2">Created</th><th>Test ID</th><th>LP A</th><th>LP B</th><th>Status</th><th>Split</th><th>Metric</th></tr></thead>
            <tbody>
              {tests.map((r:any,i:number)=> (
                <tr key={i} className="border-b border-white/10">
                  <td className="py-2">{r.created_at}</td>
                  <td>{r.test_id}</td>
                  <td>{r.lp_a}</td>
                  <td>{r.lp_b || '-'}</td>
                  <td>{r.status}</td>
                  <td>{(Number(r.traffic_split||0)*100).toFixed(0)}%</td>
                  <td>{r.primary_metric}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

