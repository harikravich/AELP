"use client"
import React, { useEffect, useState } from 'react'
import { Database, RefreshCw, Calendar } from 'lucide-react'

type FreshnessRow = { table_name: string, max_date: string | null }

export function DatasetSwitcher() {
  const [mode, setMode] = useState<'sandbox'|'prod'>('sandbox')
  const [fresh, setFresh] = useState<FreshnessRow[]>([])
  const [loading, setLoading] = useState(false)

  const refresh = async () => {
    try {
      const r = await fetch('/api/bq/freshness', { cache: 'no-store' })
      const j = await r.json()
      setFresh(j.rows || [])
    } catch {
      setFresh([])
    }
  }

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/api/dataset', { cache: 'no-store' })
        const j = await r.json()
        setMode(j.mode === 'prod' ? 'prod' : 'sandbox')
      } catch {}
      refresh()
    })()
  }, [])

  const switchMode = async (m: 'sandbox'|'prod') => {
    if (m === mode) return
    setLoading(true)
    try {
      await fetch(`/api/dataset?mode=${m}`, { method: 'POST' })
      setMode(m)
      await refresh()
    } finally {
      setLoading(false)
      // Force a reload so server components pick up cookie
      window.location.reload()
    }
  }

  const badge = (name: string, label: string) => {
    const row = fresh.find(r => r.table_name === name) as any
    const raw = row?.max_date
    const date = raw == null || raw === '' ? '—' : (typeof raw === 'string' ? raw : (raw?.value ?? String(raw)))
    const hasData = date !== '—'
    
    return (
      <div 
        className={`
          inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs
          backdrop-filter backdrop-blur-sm transition-all duration-200
          ${hasData 
            ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' 
            : 'bg-white/5 border border-white/10 text-white/40'
          }
        `} 
        title={`${name}: ${date}`}
      >
        <Calendar className="w-3 h-3" />
        <span className="font-medium">{label}</span>
        <span className="text-[10px] opacity-75">{date}</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Database className="w-4 h-4 text-white/60" />
        <div className="flex items-center rounded-xl overflow-hidden bg-white/5 border border-white/10">
          <button
            onClick={() => switchMode('sandbox')}
            className={`
              px-4 py-2 text-xs font-medium transition-all duration-300
              ${mode === 'sandbox' 
                ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-lg' 
                : 'text-white/60 hover:text-white hover:bg-white/10'
              }
            `}
            disabled={loading}
          >
            {loading && mode !== 'sandbox' && <RefreshCw className="w-3 h-3 animate-spin inline mr-1" />}
            Sandbox
          </button>
          <button
            onClick={() => switchMode('prod')}
            className={`
              px-4 py-2 text-xs font-medium transition-all duration-300
              ${mode === 'prod' 
                ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg' 
                : 'text-white/60 hover:text-white hover:bg-white/10'
              }
            `}
            disabled={loading}
          >
            {loading && mode !== 'prod' && <RefreshCw className="w-3 h-3 animate-spin inline mr-1" />}
            Production
          </button>
        </div>
      </div>
      
      <div className="hidden xl:flex items-center gap-2">
        {badge('ads_campaign_performance', 'Ads')}
        {badge('training_episodes', 'Training')}
        {badge('ga4_daily', 'GA4')}
      </div>
    </div>
  )
}
