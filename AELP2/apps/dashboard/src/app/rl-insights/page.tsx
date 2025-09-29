import React from 'react'
import { createBigQueryClient } from '../../lib/bigquery-client'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../lib/dataset'

export const dynamic = 'force-dynamic'

async function fetchPosteriors() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  try {
    const [rows] = await bq.query({ query: `SELECT ts, cell_key, metric, mean, ci_low, ci_high, samples FROM \`${projectId}.${dataset}.bandit_posteriors\` ORDER BY ts DESC LIMIT 100` })
    return rows as any[]
  } catch { return [] }
}

export default async function RLInsights() {
  const rows = await fetchPosteriors()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">RL Insights</h1>
      <div className="bg-white/5 rounded p-4">
        <table className="w-full text-sm">
          <thead><tr className="text-left border-b border-white/10"><th className="py-2">Cell</th><th>Mean CAC</th><th>CI</th><th>Samples</th><th>ts</th></tr></thead>
          <tbody>
            {rows.filter((r:any)=> r.metric==='cac').map((r:any,i:number)=> (
              <tr key={i} className="border-b border-white/10">
                <td className="py-2">{r.cell_key}</td>
                <td>${Number(r.mean||0).toFixed(2)}</td>
                <td>${Number(r.ci_low||0).toFixed(2)}–${Number(r.ci_high||0).toFixed(2)}</td>
                <td>{Number(r.samples||0)}</td>
                <td>{r.ts}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
