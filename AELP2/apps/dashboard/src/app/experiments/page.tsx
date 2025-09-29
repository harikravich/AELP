import React from 'react'

export const dynamic = 'force-dynamic'
import { absoluteUrl } from '../lib/url'

async function fetchTests() {
  const r = await fetch(absoluteUrl('/api/bq/lp/tests'), { cache: 'no-store' })
  const j = await r.json().catch(()=>({ rows: [] }))
  return j.rows || []
}

export default async function ExperimentsPage() {
  const rows = await fetchTests()
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Experiments</h2>
        <a className="text-indigo-300 text-sm" href="/">Back to Overview</a>
      </div>

      <div className="glass-card p-4">
        <div className="text-sm font-semibold mb-2">LP Tests</div>
        {rows.length === 0 ? (
          <div className="text-sm text-white/60">No tests yet. Use System Control → LP Publish or POST /api/control/lp/publish.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-white/10">
                <th className="py-2">Created</th>
                <th>Test ID</th>
                <th>LP A</th>
                <th>LP B</th>
                <th>Status</th>
                <th>Split</th>
                <th>Primary Metric</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r: any, i: number) => (
                <tr key={i} className="border-b border-white/10">
                  <td className="py-2">{r.created_at}</td>
                  <td>{r.test_id}</td>
                  <td>{r.lp_a}</td>
                  <td>{r.lp_b || '-'}</td>
                  <td>{r.status}</td>
                  <td>{(Number(r.traffic_split) * 100).toFixed(0)}%</td>
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
