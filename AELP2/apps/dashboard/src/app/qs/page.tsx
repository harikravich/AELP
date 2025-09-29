"use client"
import React, { useEffect, useState } from 'react'

type Row = {
  date: string
  campaign_id: string
  name: string | null
  impressions: number
  clicks: number
  spend: number
  conversions: number
  impression_share: number | null
  lost_is_rank: number | null
  lost_is_budget: number | null
  ctr: number
  cvr: number
  cac: number | null
  brand: boolean
  alerts: {type: string; severity: string}[]
  suggestions: string[]
}

export default function QSPage() {
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string|undefined>()
  useEffect(() => {
    fetch('/api/bq/ads/qs-is?days=14').then(r => r.json()).then(d => {
      if (d.error) setError(d.error)
      else setRows(d.rows || [])
    }).catch(e => setError(String(e))).finally(()=>setLoading(false))
  }, [])
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold mb-4">QS / Impression Share (last 14 days)</h1>
      {loading && <div>Loading…</div>}
      {error && <div className="text-red-600">{error}</div>}
      {!loading && !error && (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="py-2">Date</th>
              <th>Campaign</th>
              <th>Spend</th>
              <th>Impr</th>
              <th>CTR</th>
              <th>CVR</th>
              <th>CAC</th>
              <th>IS</th>
              <th>Lost IS (Rank)</th>
              <th>Alerts</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0,200).map((r,i)=> (
              <tr key={i} className="border-b hover:bg-muted/30">
                <td className="py-1">{r.date?.slice(0,10)}</td>
                <td>{r.name || r.campaign_id}{r.brand ? ' · Brand' : ''}</td>
                <td>${r.spend.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
                <td>{r.impressions.toLocaleString()}</td>
                <td>{(r.ctr*100).toFixed(2)}%</td>
                <td>{(r.cvr*100).toFixed(2)}%</td>
                <td>{r.cac!=null ? `$${r.cac.toFixed(0)}` : '—'}</td>
                <td>{r.impression_share!=null ? (r.impression_share*100).toFixed(1)+'%' : '—'}</td>
                <td>{r.lost_is_rank!=null ? (r.lost_is_rank*100).toFixed(1)+'%' : '—'}</td>
                <td>
                  {r.alerts?.map((a,idx)=>(
                    <span key={idx} className={`inline-block px-2 py-0.5 mr-1 rounded ${a.severity==='high'?'bg-red-100 text-red-700':'bg-amber-100 text-amber-800'}`}>{a.type}</span>
                  ))}
                  {r.suggestions?.length ? <div className="text-xs text-muted-foreground mt-1">{r.suggestions[0]}</div> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

