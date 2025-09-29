import React from 'react'
import { createBigQueryClient } from '../../lib/bigquery-client'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../lib/dataset'

export const dynamic = 'force-dynamic'

async function fetchSegments() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = createBigQueryClient(projectId)
  try {
    const [rows] = await bq.query({ query: `SELECT date, segment, score FROM \`${projectId}.${dataset}.segment_scores_daily\` ORDER BY date DESC LIMIT 50` })
    return rows as any[]
  } catch { return [] }
}

export default async function Audiences() {
  const rows = await fetchSegments()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Audiences</h1>
      <div className="bg-white/5 rounded p-4">
        {rows.length === 0 ? <div className="text-sm text-white/70">No segment rows yet.</div> : (
          <table className="w-full text-sm">
            <thead><tr className="text-left border-b border-white/10"><th className="py-2">Segment</th><th>Score</th><th>Action</th></tr></thead>
            <tbody>
              {rows.slice(0,10).map((r:any,i:number)=> (
                <tr key={i} className="border-b border-white/10">
                  <td className="py-2">{r.segment}</td>
                  <td>{Number(r.score||0).toFixed(2)}</td>
                  <td>
                    <form method="post" action="/api/control/audience/sync">
                      <input type="hidden" name="segment" value={r.segment} />
                      <button className="text-indigo-300">Export (shadow)</button>
                    </form>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
