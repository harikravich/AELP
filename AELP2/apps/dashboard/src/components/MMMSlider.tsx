"use client"
import React from 'react'

export default function MMMSlider({ channel = 'google_ads' }: { channel?: string }) {
  const [grid, setGrid] = React.useState<{ spend:number[], conv:number[] }|null>(null)
  const [val, setVal] = React.useState<number>(0)
  const [conv, setConv] = React.useState<number>(0)
  React.useEffect(()=>{
    fetch(`/api/bq/mmm/curves?channel=${encodeURIComponent(channel)}`)
      .then(r=>r.json()).then(j=>{
        if (j.spend_grid && j.conv_grid) {
          const spend = j.spend_grid.map((x:number)=>Number(x))
          const conv = j.conv_grid.map((x:number)=>Number(x))
          setGrid({ spend, conv })
          const mid = Math.floor(spend.length/2)
          setVal(spend[mid]||0)
          setConv(conv[mid]||0)
        }
      })
  },[channel])
  if (!grid) return <div className="text-sm text-gray-600 mt-3">Loading curve…</div>
  const onChange = (e:any)=>{
    const v = Number(e.target.value)
    setVal(v)
    let idx = 0; let best=1e18
    grid.spend.forEach((s,i)=>{ const d=Math.abs(s-v); if(d<best){best=d;idx=i} })
    setConv(grid.conv[idx]||0)
  }
  const min = Math.floor(grid.spend[0])
  const max = Math.ceil(grid.spend[grid.spend.length-1])
  const step = Math.max(1, Math.floor((max-min)/100))
  return (
    <div className="mt-2">
      <input type="range" min={min} max={max} step={step} value={val} onChange={onChange} className="w-full" />
      <div className="mt-2 text-sm flex justify-between">
        <span>Budget: ${val.toFixed(0)}/day</span>
        <span>Expected conversions: {conv.toFixed(0)} · CAC ${(val/Math.max(conv,1)).toFixed(2)}</span>
      </div>
    </div>
  )
}

