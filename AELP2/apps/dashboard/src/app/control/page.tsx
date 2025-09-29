"use client"
import React, { useState } from 'react'

async function post(url: string): Promise<any> {
  const r = await fetch(url, { method: 'POST' })
  try { return await r.json() } catch { return { ok: false, error: `HTTP ${r.status}` } }
}

export default function ControlPage() {
  const [kpiIds, setKpiIds] = useState('6453292723')
  const [only, setOnly] = useState('')
  const [episodes, setEpisodes] = useState(1)
  const [steps, setSteps] = useState(300)
  const [budget, setBudget] = useState(3000)
  const [log, setLog] = useState<string>('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<Record<string, string | null>>({})
  const [canaryIds, setCanaryIds] = useState('')
  const [canaryDir, setCanaryDir] = useState<'up'|'down'>('down')

  const loadStatus = async () => {
    try {
      const r = await fetch('/api/control/status', { cache: 'no-store' })
      const j = await r.json()
      setStatus(j.status || {})
    } catch {}
  }

  const run = async (fn: () => Promise<any>) => {
    setBusy(true)
    try {
      const res = await fn()
      setLog(JSON.stringify(res, null, 2))
    } catch (e: any) {
      setLog(String(e?.message || e))
    } finally {
      setBusy(false)
      loadStatus()
    }
  }

  React.useEffect(() => { loadStatus() }, [])

  return (
    <div className="space-y-6">
      <div className="bg-white shadow-sm rounded p-4 space-y-4">
        <h2 className="text-lg font-medium">Control Surface</h2>
        <p className="text-xs text-gray-600">All actions target the selected dataset (see header switcher). Writes are blocked on Prod for safety.</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">Apply Canary Budget Change</div>
            <div className="text-xs text-gray-600">One-click apply to allowlisted campaigns (shadow‑only unless flags permit).</div>
            <input value={canaryIds} onChange={e=>setCanaryIds(e.target.value)} className="border rounded px-2 py-1 text-sm w-full" placeholder="Comma-separated campaign IDs" />
            <div className="flex items-center gap-3 text-xs">
              <label className="flex items-center gap-1"><input type="radio" checked={canaryDir==='down'} onChange={()=>setCanaryDir('down')} /> Down</label>
              <label className="flex items-center gap-1"><input type="radio" checked={canaryDir==='up'} onChange={()=>setCanaryDir('up')} /> Up</label>
            </div>
            <button disabled={busy} onClick={()=>run(()=>fetch('/api/control/apply-canary',{method:'POST',body:JSON.stringify({campaign_ids:canaryIds,direction:canaryDir}),headers:{'Content-Type':'application/json'}}).then(r=>r.json()))} className="px-3 py-1 rounded bg-red-600 text-white text-sm disabled:opacity-50">Apply Canary</button>
          </div>
          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">KPI Lock + Refresh Views</div>
            <div className="text-xs text-gray-600">Sets KPI IDs and refreshes KPI/training/ads views in the selected dataset.</div>
            <input value={kpiIds} onChange={e=>setKpiIds(e.target.value)} className="border rounded px-2 py-1 text-sm w-full" placeholder="Comma-separated KPI IDs" />
            <button disabled={busy} onClick={()=>run(()=>post(`/api/control/kpi-lock?ids=${encodeURIComponent(kpiIds)}`))} className="px-3 py-1 rounded bg-emerald-600 text-white text-sm disabled:opacity-50">Lock KPI</button>
            <div className="text-[11px] text-gray-500">Fidelity last: {status['fidelity_last_timestamp'] || '—'}</div>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">Ads Ingestion (last 14d)</div>
            <div className="text-xs text-gray-600">Runs Ads MCC tasks with --only &lt;PERSONAL_CID&gt; (required).</div>
            <input value={only} onChange={e=>setOnly(e.target.value)} className="border rounded px-2 py-1 text-sm w-full" placeholder="<PERSONAL_10_DIGIT_CID>" />
            <button disabled={busy} onClick={()=>run(()=>post(`/api/control/ads-ingest?only=${encodeURIComponent(only)}`))} className="px-3 py-1 rounded bg-sky-600 text-white text-sm disabled:opacity-50">Run Ads Ingest</button>
            <div className="text-[11px] text-gray-500">Ads last: {status['ads_ingest_last_date'] || '—'}</div>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">GA4 Ingestion (last 28d)</div>
            <div className="text-xs text-gray-600">Requires GA4_PROPERTY_ID env (properties/308028264).</div>
            <button disabled={busy} onClick={()=>run(()=>post('/api/control/ga4-ingest'))} className="px-3 py-1 rounded bg-indigo-600 text-white text-sm disabled:opacity-50">Run GA4 Ingest</button>
            <div className="text-[11px] text-gray-500">GA4 last: {status['ga4_ingest_last_date'] || '—'}</div>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">GA4 Lag-aware Attribution</div>
            <div className="text-xs text-gray-600">Computes lagged GA4 credit; writes ga4_lagged_attribution.</div>
            <button disabled={busy} onClick={()=>run(()=>post('/api/control/ga4-attribution'))} className="px-3 py-1 rounded bg-indigo-600 text-white text-sm disabled:opacity-50">Run GA4 Attribution</button>
            <div className="text-[11px] text-gray-500">GA4 lagged last: {status['ga4_attribution_last_date'] || '—'}</div>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">YouTube Reach Planner</div>
            <div className="text-xs text-gray-600">Writes reach estimates (stub if API unavailable).</div>
            <button disabled={busy} onClick={()=>run(()=>post('/api/control/reach-planner'))} className="px-3 py-1 rounded bg-red-600 text-white text-sm disabled:opacity-50">Run Reach Planner</button>
            <button disabled={busy} onClick={async()=>{
              setBusy(true)
              try { const r = await fetch('/api/bq/reach-estimates', { cache: 'no-store' }); const j = await r.json(); setLog(JSON.stringify(j, null, 2)) } catch(e:any) { setLog(String(e?.message||e)) } finally { setBusy(false) }
            }} className="px-3 py-1 rounded bg-gray-700 text-white text-sm disabled:opacity-50">View Estimates</button>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">Canary Rollback</div>
            <div className="text-xs text-gray-600">Revert last N canary changes (shadow intent only).</div>
            <button disabled={busy} onClick={()=>run(()=>fetch('/api/control/canary-rollback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({last_n:5})}).then(r=>r.json()))} className="px-3 py-1 rounded bg-rose-700 text-white text-sm disabled:opacity-50">Rollback Last 5</button>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">Training (episodes/steps/budget)</div>
            <div className="flex gap-2">
              <input type="number" min={1} value={episodes} onChange={e=>setEpisodes(parseInt(e.target.value||'1',10))} className="border rounded px-2 py-1 text-sm w-24" placeholder="Episodes" />
              <input type="number" min={1} value={steps} onChange={e=>setSteps(parseInt(e.target.value||'300',10))} className="border rounded px-2 py-1 text-sm w-24" placeholder="Steps" />
              <input type="number" min={1} value={budget} onChange={e=>setBudget(parseFloat(e.target.value||'3000'))} className="border rounded px-2 py-1 text-sm w-28" placeholder="Budget" />
            </div>
            <button disabled={busy} onClick={()=>run(()=>post(`/api/control/training-run?episodes=${episodes}&steps=${steps}&budget=${budget}`))} className="px-3 py-1 rounded bg-amber-600 text-white text-sm disabled:opacity-50">Run Training</button>
            <div className="text-[11px] text-gray-500">Training last: {status['training_last_timestamp'] || '—'}</div>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">KPI-only Fidelity (14d)</div>
            <div className="text-xs text-gray-600">Computes MAPE/RMSE on ROAS/CAC; inserts into fidelity_evaluations.</div>
            <button disabled={busy} onClick={()=>run(()=>post('/api/control/fidelity-kpi'))} className="px-3 py-1 rounded bg-purple-600 text-white text-sm disabled:opacity-50">Run Fidelity</button>
            <div className="text-[11px] text-gray-500">Fidelity last: {status['fidelity_last_timestamp'] || '—'}</div>
          </div>

          <div className="border rounded p-3 space-y-2">
            <div className="font-medium text-sm">AB Exposures (7d)</div>
            <div className="text-xs text-gray-600">GrowthBook-style exposure logs aggregated by experiment/variant.</div>
            <button disabled={busy} onClick={async()=>{
              setBusy(true)
              try { const r = await fetch('/api/bq/ab-exposures', { cache: 'no-store' }); const j = await r.json(); setLog(JSON.stringify(j, null, 2)) } catch(e:any) { setLog(String(e?.message||e)) } finally { setBusy(false) }
            }} className="px-3 py-1 rounded bg-gray-700 text-white text-sm disabled:opacity-50">Load Exposures</button>
          </div>
        </div>

        <div className="bg-gray-50 rounded p-3 text-xs overflow-x-auto min-h-[120px]">
          <pre>{log}</pre>
        </div>
      </div>
    </div>
  )
}
