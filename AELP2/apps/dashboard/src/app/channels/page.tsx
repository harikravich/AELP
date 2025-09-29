import React from 'react'

export const dynamic = 'force-dynamic'
import { absoluteUrl } from '../lib/url'

async function fetchCandidates() {
  const r = await fetch(absoluteUrl('/api/research/channels?status=new'), { cache: 'no-store' })
  const j = await r.json().catch(()=>({ rows: [] }))
  return j.rows || []
}

export default async function ChannelsPage() {
  const rows = await fetchCandidates()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Channels</h1>
      <div className="bg-white/5 rounded p-4">
        <h2 className="text-lg mb-2">Candidates</h2>
        {rows.length === 0 ? <div className="text-sm text-white/70">No candidates yet.</div> : (
          <table className="w-full text-sm">
            <thead><tr className="text-left border-b border-white/10"><th className="py-2">Name</th><th>Type</th><th>Score</th><th>Action</th></tr></thead>
            <tbody>
              {rows.map((r:any,i:number)=> (
                <tr key={i} className="border-b border-white/10">
                  <td className="py-2">{r.name}</td>
                  <td>{r.type}</td>
                  <td>{Number(r.score_total||0)}</td>
                  <td><form method="post" action="/api/bq/explore/cells"><input type="hidden" name="angle" value="balance_safety" /><input type="hidden" name="audience" value="parents_ios" /><input type="hidden" name="channel" value={r.name?.toLowerCase()||'channel'} /><input type="hidden" name="lp" value="/balance-v2" /><input type="hidden" name="offer" value="trial" /><button className="text-indigo-300">Request Pilot</button></form></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
