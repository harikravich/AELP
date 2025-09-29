import React from 'react'

export const dynamic = 'force-dynamic'
import { absoluteUrl } from '../lib/url'

async function fetchFreshness() {
  const r = await fetch(absoluteUrl('/api/bq/freshness'), { cache: 'no-store' })
  const j = await r.json().catch(()=>({ rows: [] }))
  return j.rows || []
}

export default async function Backstage() {
  const rows = await fetchFreshness()
  const flags = {
    PILOT_MODE: process.env.PILOT_MODE,
    GATES_ENABLED: process.env.GATES_ENABLED,
    AELP2_ALLOW_GOOGLE_MUTATIONS: process.env.AELP2_ALLOW_GOOGLE_MUTATIONS,
    AELP2_ALLOW_BANDIT_MUTATIONS: process.env.AELP2_ALLOW_BANDIT_MUTATIONS,
  }
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Backstage</h1>
      <div className="bg-white/5 rounded p-4">
        <h2 className="text-lg mb-2">Data Freshness</h2>
        <ul className="text-sm">
          {rows.map((r:any,i:number)=> (
            <li key={i}>{r.table_name}: {r.max_date || '—'}</li>
          ))}
        </ul>
      </div>
      <div className="bg-white/5 rounded p-4">
        <h2 className="text-lg mb-2">Flags & Risk</h2>
        <pre className="text-xs">{JSON.stringify(flags, null, 2)}</pre>
      </div>
    </div>
  )
}
