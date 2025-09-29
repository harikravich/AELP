"use client"
import React from 'react'
import dynamic from 'next/dynamic'
import { Responsive as RGL, WidthProvider } from 'react-grid-layout'
import Card from '../../components/Card'
import MetricTile from '../../components/MetricTile'
import { TimeSeriesChart } from '../../components/TimeSeriesChart'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import ChatViz, { VizSpec } from '../../components/ChatViz'

const ResponsiveGridLayout = WidthProvider(RGL)

type Widget = { id: string, type: 'metric-cac-yesterday' | 'chart-spend-7d' | 'ga4-top-channels' | 'image' | 'viz', x?:number,y?:number,w?:number,h?:number, data?: any }

export default function CanvasPage(){
  const [widgets, setWidgets] = React.useState<Widget[]>(()=>{
    try { return JSON.parse(localStorage.getItem('canvas-widgets')||'[]') } catch { return [] }
  })
  const [imgUrl, setImgUrl] = React.useState('')
  const [pins, setPins] = React.useState<any[]>([])

  React.useEffect(()=>{ fetch('/api/canvas/list').then(r=>r.json()).then(j=> setPins(j.items||[])) },[])

  React.useEffect(()=>{ localStorage.setItem('canvas-widgets', JSON.stringify(widgets)) }, [widgets])

  const add = (w: Widget)=> setWidgets(prev=> [...prev, { w:3, h:3, x:0, y:Infinity, ...w, id: crypto.randomUUID() }])
  const remove = (id:string)=> setWidgets(prev=> prev.filter(w=>w.id!==id))

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Flow Canvas</h1>
        <div className="flex gap-2 text-sm">
          <button className="btn-primary" onClick={()=>add({ id:'x', type:'metric-cac-yesterday' })}>+ CAC Yesterday</button>
          <button className="btn-primary" onClick={()=>add({ id:'x', type:'chart-spend-7d', w:4, h:3 })}>+ Spend 7d</button>
          <button className="btn-primary" onClick={()=>add({ id:'x', type:'ga4-top-channels', w:4, h:3 })}>+ GA4 Channels</button>
          <div className="flex items-center gap-2">
            <input value={imgUrl} onChange={e=>setImgUrl(e.target.value)} placeholder="Image URL" className="border rounded px-2 py-1" />
            <button className="btn-primary" onClick={()=>{ if(imgUrl) add({ id:'x', type:'image', data:{ url: imgUrl }, w:3, h:3 }); setImgUrl('') }}>+ Image</button>
          </div>
          <div className="flex items-center gap-2">
            <select className="border rounded px-2 py-1" onChange={(e)=>{
              const id = e.target.value; if(!id) return; const pin = pins.find(p=>String(p.id)===id); if(pin){
                try{ const viz:VizSpec = typeof pin.viz==='string'?JSON.parse(pin.viz):pin.viz; add({ id:'x', type:'viz', data:{ viz }, w:4, h:3 }) } catch{}
              }
              e.currentTarget.selectedIndex = 0
            }}>
              <option value="">+ Pinned from Chat…</option>
              {pins.map((p:any)=> (<option key={p.id} value={p.id}>{p.title || p.id}</option>))}
            </select>
          </div>
        </div>
      </div>

      <ResponsiveGridLayout className="layout" rowHeight={90} cols={{ lg: 12, md: 10, sm: 8, xs: 4, xxs: 2 }}
        onLayoutChange={(layout)=>{
          setWidgets(prev=> prev.map(w=>{ const l = layout.find(i=>i.i===w.id); return l ? { ...w, x:l.x, y:l.y, w:l.w, h:l.h } : w }))
        }}
      >
        {widgets.map(w=> (
          <div key={w.id} data-grid={{ i:w.id, x:w.x||0, y:w.y||0, w:w.w||3, h:w.h||3 }} className="">
            <Card actions={<button className="text-xs text-slate-500 hover:text-rose-600" onClick={()=>remove(w.id)}>Remove</button>}>
              {w.type==='metric-cac-yesterday' && <CACYesterday />}
              {w.type==='chart-spend-7d' && <Spend7d />}
              {w.type==='ga4-top-channels' && <GA4TopChannels />}
              {w.type==='image' && <img src={w.data?.url} className="max-w-full max-h-[300px] object-contain rounded" />}
              {w.type==='viz' && w.data?.viz && <ChatViz viz={w.data.viz as VizSpec} />}
            </Card>
          </div>
        ))}
      </ResponsiveGridLayout>
      {widgets.length===0 && (
        <div className="text-sm text-slate-600">Use the buttons above to add widgets. Drag and resize to arrange your canvas.</div>
      )}
    </div>
  )
}

function useFetch<T=any>(url: string){
  const [data, setData] = React.useState<T | null>(null)
  React.useEffect(()=>{ let ok=true; fetch(url).then(r=>r.json()).then(j=>{ if(ok) setData(j as T) }); return ()=>{ok=false} },[url])
  return data
}

function CACYesterday(){
  const j = useFetch<any>('/api/bq/kpi/yesterday')
  const r = j?.row
  const cac = r ? Number(r.cac||0).toFixed(2) : '—'
  const d = r?.date ? String(r.date) : ''
  return <MetricTile label={`CAC (${d||'yesterday'})`} value={r?`$${cac}`:'—'} hint={r?`Spend $${Number(r.cost||0).toFixed(0)} · Conversions ${Number(r.conversions||0).toFixed(0)}`:''} tone="rose" />
}

function Spend7d(){
  const j = useFetch<any>('/api/bq/kpi/daily?days=7')
  const rows = j?.rows || []
  const data = rows.map((r:any)=> ({ date: String(r.date).slice(5), cost: Number(r.cost||0) }))
  return (
    <div>
      <div className="text-sm font-medium mb-2">Spend (Last 7 days)</div>
      <div style={{ width:'100%', height:240 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="cost" fill="#6366f1" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function GA4TopChannels(){
  const j = useFetch<any>('/api/bq/ga4/channels?days=28')
  const rows = (j?.rows || []).slice(0,8)
  const data = rows.map((r:any)=> ({ ch: r.default_channel_group, conv: Number(r.conversions||0) }))
  return (
    <div>
      <div className="text-sm font-medium mb-2">Top GA4 Channels (28d)</div>
      <div style={{ width:'100%', height:240 }}>
        <ResponsiveContainer>
          {/* @ts-ignore */}
          <BarChart data={data} layout="vertical" margin={{ top: 10, right: 10, left: 20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            {/* @ts-ignore */}
            <YAxis type="category" dataKey="ch" width={120} />
            <Tooltip />
            <Bar dataKey="conv" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
