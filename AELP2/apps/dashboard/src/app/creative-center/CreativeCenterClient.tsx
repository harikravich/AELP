"use client"
import React, { useState } from 'react'
import { fmtUSD, fmtInt, fmtFloat, fmtPct } from '../../lib/utils'

type Row = {
  date: string
  campaign_id: string
  campaign_name: string
  ad_group_id: string
  ad_group_name: string
  ad_id: string
  ad_name: string
  customer_id?: string
  impressions: number
  clicks: number
  cost: number
  conversions: number
  revenue: number
  ctr: number
  cvr: number
  cac: number
  roas: number
}

export default function CreativeCenterClient({ rows }: { rows: Row[] }) {
  const [open, setOpen] = useState<any|null>(null)
  return (
    <>
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="py-2 pr-4">Date</th>
              <th className="py-2 pr-4">Campaign</th>
              <th className="py-2 pr-4">Ad Group</th>
              <th className="py-2 pr-4">Ad</th>
              <th className="py-2 pr-4">Impr</th>
              <th className="py-2 pr-4">Clicks</th>
              <th className="py-2 pr-4">Cost</th>
              <th className="py-2 pr-4">Conv</th>
              <th className="py-2 pr-4">Revenue</th>
              <th className="py-2 pr-4">CTR</th>
              <th className="py-2 pr-4">CVR</th>
              <th className="py-2 pr-4">CAC</th>
              <th className="py-2 pr-4">ROAS</th>
              <th className="py-2 pr-4 text-right">Preview</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r,i)=> (
              <tr key={`${r.date}-${r.ad_id}-${i}`} className="border-b last:border-0">
                <td className="py-2 pr-4">{r.date}</td>
                <td className="py-2 pr-4">{r.campaign_name}</td>
                <td className="py-2 pr-4">{r.ad_group_name}</td>
                <td className="py-2 pr-4">{r.ad_name}</td>
                <td className="py-2 pr-4">{fmtInt(r.impressions)}</td>
                <td className="py-2 pr-4">{fmtInt(r.clicks)}</td>
                <td className="py-2 pr-4">{fmtUSD(r.cost,0)}</td>
                <td className="py-2 pr-4">{fmtInt(r.conversions)}</td>
                <td className="py-2 pr-4">{fmtUSD(r.revenue,0)}</td>
                <td className="py-2 pr-4">{fmtPct(r.ctr,1)}</td>
                <td className="py-2 pr-4">{fmtPct(r.cvr,2)}</td>
                <td className="py-2 pr-4">{fmtUSD(r.cac,0)}</td>
                <td className="py-2 pr-4">{fmtFloat(r.roas,2)}x</td>
                <td className="py-2 pr-4 text-right">
                  <button className="px-2 py-1 text-xs rounded bg-indigo-600 text-white" onClick={async()=>{
                    try {
                      const q = new URLSearchParams({ ad_id: String(r.ad_id), campaign_id: String(r.campaign_id), customer_id: String(r.customer_id||'') })
                      const res = await fetch(`/api/ads/creative?${q.toString()}`)
                      const j = await res.json()
                      setOpen({ meta:r, detail:j })
                    } catch (e:any) { setOpen({ meta:r, detail:{ error:String(e) } }) }
                  }}>Preview</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {open && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={()=>setOpen(null)}>
          <div className="bg-white rounded-xl p-4 w-[720px] max-h-[80vh] overflow-y-auto text-slate-900" onClick={e=>e.stopPropagation()}>
            <div className="flex items-start justify-between mb-2">
              <div>
                <div className="text-xs text-slate-500">Campaign</div>
                <div className="font-semibold">{open.meta?.campaign_name}</div>
              </div>
              <button onClick={()=>setOpen(null)} className="text-slate-600">✕</button>
            </div>
            {open.detail?.error ? (
              <div className="text-sm text-rose-600">{open.detail.error}</div>
            ) : (
              <div>
                <div className="text-xs text-slate-500 mb-1">Ad Preview</div>
                <div className="border rounded p-3">
                  <div className="text-xs text-slate-500">Final URL: {open.detail?.final_urls?.[0] || '—'}</div>
                  <div className="mt-2">
                    <div className="text-lg font-semibold">{(open.detail?.headlines||[]).slice(0,3).join(' | ')}</div>
                    <div className="text-sm text-slate-700">{(open.detail?.descriptions||[]).slice(0,2).join(' • ')}</div>
                    <div className="text-xs text-slate-500 mt-1">{open.detail?.path1 || ''}{open.detail?.path2 ? ` / ${open.detail.path2}`:''}</div>
                  </div>
                </div>
                {Array.isArray(open.detail?.images) && open.detail.images.length>0 && (
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    {open.detail.images.slice(0,6).map((u:string,idx:number)=> (
                      <img key={idx} src={u} alt="creative" className="rounded border" />
                    ))}
                  </div>
                )}
                <div className="mt-3 text-right">
                  <a className="btn-glass" target="_blank" rel="noreferrer" href={`https://ads.google.com/aw/ads/search/ads/${open.meta?.customer_id}/ad/detail/${open.meta?.ad_id}`}>Open in Google Ads</a>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}

