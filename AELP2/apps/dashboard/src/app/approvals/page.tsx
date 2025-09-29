import React from 'react'
import { createBigQueryClient } from '../../lib/bigquery-client'
import { fmtWhen } from '../../lib/utils'

export const dynamic = 'force-dynamic'

async function fetchQueue() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
  const bq = createBigQueryClient(projectId)
  try {
    const [rows] = await bq.query({ query: `SELECT run_id, platform, type, campaign_id, ad_group_id, asset_group_id, status, requested_by, enqueued_at FROM \`${projectId}.${dataset}.creative_publish_queue\` WHERE status='queued' ORDER BY enqueued_at DESC LIMIT 50` })
    const items = rows as any[]
    const ids = Array.from(new Set(items.map(r=> r.campaign_id).filter(Boolean)))
    let nameMap: Record<string,string> = {}
    if (ids.length) {
      const [names] = await bq.query({ query: `
        SELECT campaign_id, ANY_VALUE(campaign_name) AS name
        FROM \`${projectId}.${dataset}.ads_ad_performance\`
        WHERE campaign_id IN (${ids.map((x:string)=>`'${x}'`).join(',')})
        GROUP BY campaign_id` })
      for (const n of names as any[]) nameMap[n.campaign_id] = n.name || n.campaign_id
    }
    return items.map(r=> ({ ...r, campaign_name: nameMap[r.campaign_id]||r.campaign_id }))
  } catch { return [] }
}

export default async function ApprovalsPage() {
  const items = await fetchQueue()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Approvals</h1>
      <div className="bg-white/5 rounded p-4">
        {items.length === 0 ? <div className="text-sm text-white/70">No pending items.</div> : (
          <table className="w-full text-sm">
            <thead><tr className="text-left border-b border-white/10"><th className="py-2">Item</th><th>Type</th><th>Campaign</th><th>Queued</th><th>Action</th></tr></thead>
            <tbody>
              {items.map((r:any,i:number)=> (
                <tr key={i} className="border-b border-white/10">
                  <td className="py-2">Creative Publish</td>
                  <td>{r.platform}/{r.type}</td>
                  <td>{r.campaign_name || r.campaign_id || '-'}</td>
                  <td>{fmtWhen(r.enqueued_at)}</td>
                  <td>
                    <form action={`/api/control/creative/publish`} method="post">
                      <input type="hidden" name="run_id" value={r.run_id} />
                      <button className="text-indigo-300">Approve</button>
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
