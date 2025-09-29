import React from 'react'

export const dynamic = 'force-dynamic'
import { absoluteUrl } from '../lib/url'

async function fetchHeadroom() {
  const r = await fetch(absoluteUrl('/api/bq/headroom'), { cache: 'no-store' })
  const j = await r.json().catch(()=>({ rows: [] }))
  return j.rows || []
}

export default async function SpendPlanner() {
  const rows = await fetchHeadroom()
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Spend Planner</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {rows.map((r:any,i:number)=> (
          <div key={i} className="glass-card p-4">
            <div className="text-sm font-semibold mb-1">{String(r.channel).replace('_',' ').toUpperCase()}</div>
            <div className="text-xs text-white/60 mb-3">Decision card uses current KPI Source</div>
            <div className="text-sm">
              <div className="mb-1">Room to grow: <strong>{Number(r.room||0).toLocaleString('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0})}/day</strong></div>
              <div className="mb-1">Expected extra customers/day: <strong>{Number(r.extra_per_day||0).toLocaleString('en-US')}</strong></div>
              <div className="mb-3">Estimated CAC: <strong>{Number(r.cac||0).toLocaleString('en-US',{style:'currency','currency':'USD',maximumFractionDigits:0})}</strong></div>
            </div>
            <div className="flex items-center gap-3">
              <form method="post" action="/api/control/status">
                <button className="btn-primary">Approve increase</button>
              </form>
              <form method="post" action="/api/control/reach-planner">
                <button className="btn-glass">Reach estimate</button>
              </form>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
