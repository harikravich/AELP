"use client"
import React, { useEffect, useState } from 'react'

type Source = 'ads' | 'ga4_all' | 'ga4_google'

export function KpiSourceSwitcher() {
  const [src, setSrc] = useState<Source>('ga4_google')
  const [loading, setLoading] = useState(false)
  useEffect(() => { (async()=>{
    try { const r = await fetch('/api/kpi-source'); const j = await r.json(); setSrc(j.source as Source) } catch {}
  })() }, [])

  const set = async (s: Source) => {
    if (s === src) return
    setLoading(true)
    try {
      await fetch(`/api/kpi-source?source=${s}`, { method: 'POST' })
      setSrc(s)
    } finally {
      setLoading(false)
      window.location.reload()
    }
  }

  const btn = (val: Source, label: string) => (
    <button onClick={()=>set(val)}
      className={`px-3 py-1.5 text-xs rounded-l-none rounded-r-none border border-white/10 ${src===val?'bg-amber-400 text-black':'text-white/70 hover:text-white hover:bg-white/10'}`}
      disabled={loading}
      title={val==='ads' ? 'Ads conversions' : val==='ga4_all' ? 'GA4 sitewide purchases' : 'GA4 purchases (google / cpc)'}
    >{label}</button>
  )

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-white/60">KPI Source</span>
      <div className="inline-flex rounded overflow-hidden">
        {btn('ads','Ads')}
        {btn('ga4_all','GA4 All')}
        {btn('ga4_google','GA4 Google')}
      </div>
    </div>
  )
}

