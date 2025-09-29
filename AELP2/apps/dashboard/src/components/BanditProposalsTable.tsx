"use client"
import React, { useState } from 'react'
import { CheckCircle, XCircle, TrendingUp, TrendingDown, Shuffle } from 'lucide-react'

type Proposal = {
  timestamp: string
  platform: string
  channel: string
  campaign_id: string
  ad_id: string
  action: string
  exploration_pct?: number
  shadow?: boolean
  applied?: boolean
  reason?: string
}

export function BanditProposalsTable({ proposals }: { proposals: Proposal[] }) {
  const [rows, setRows] = useState<Proposal[]>(proposals || [])
  const [loading, setLoading] = useState<string | null>(null)

  async function approve(p: Proposal, approved: boolean) {
    const key = `${p.campaign_id}-${p.ad_id}`
    setLoading(key)
    try {
      const res = await fetch('/api/control/bandit-approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          campaign_id: p.campaign_id,
          ad_id: p.ad_id,
          action: p.action,
          approved,
          approver: 'exec-dashboard',
          reason: approved ? 'approved' : 'rejected',
        }),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'failed')
      setRows(prev => prev.map(r => 
        (r.campaign_id === p.campaign_id && r.ad_id === p.ad_id) 
          ? { ...r, reason: (p.reason || '') + ` | ${approved ? 'approved' : 'rejected'}` } 
          : r
      ))
    } catch (e) {
      console.error('approve failed', e)
    } finally {
      setLoading(null)
    }
  }

  const getActionIcon = (action: string) => {
    if (action.toLowerCase().includes('increase')) return <TrendingUp className="w-4 h-4 text-emerald-400" />
    if (action.toLowerCase().includes('decrease')) return <TrendingDown className="w-4 h-4 text-rose-400" />
    return <Shuffle className="w-4 h-4 text-indigo-400" />
  }

  return (
    <div className="overflow-x-auto rounded-lg">
      <table className="table-clean w-full">
        <thead>
          <tr>
            <th>When</th>
            <th>Platform</th>
            <th>Channel</th>
            <th>Campaign</th>
            <th>Ad</th>
            <th>Action</th>
            <th>Explore</th>
            <th>Status</th>
            <th>Reason</th>
            <th className="text-center">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={10} className="text-center py-8 text-white/40">
                No proposals found
              </td>
            </tr>
          ) : (
            rows.map((r, i) => {
              const key = `${r.campaign_id}-${r.ad_id}`
              const isLoading = loading === key
              
              return (
                <tr key={`${r.timestamp}-${i}`}>
                  <td className="text-xs text-white/60">{r.timestamp}</td>
                  <td>
                    <span className="badge">{r.platform}</span>
                  </td>
                  <td className="text-white/80">{r.channel}</td>
                  <td className="font-mono text-xs text-white/70">{r.campaign_id}</td>
                  <td className="font-mono text-xs text-white/70">{r.ad_id}</td>
                  <td>
                    <div className="flex items-center gap-2">
                      {getActionIcon(r.action)}
                      <span className="text-white/90">{r.action}</span>
                    </div>
                  </td>
                  <td>
                    <span className="text-indigo-400 font-semibold">
                      {((r.exploration_pct || 0) * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td>
                    <div className="flex gap-2">
                      {r.shadow && <span className="badge badge-warning text-xs">Shadow</span>}
                      {r.applied && <span className="badge badge-success text-xs">Applied</span>}
                    </div>
                  </td>
                  <td className="text-xs text-white/60 max-w-xs truncate">{r.reason}</td>
                  <td>
                    <div className="flex gap-1 justify-center">
                      <button 
                        className="btn-glass px-3 py-1.5 text-xs flex items-center gap-1 hover:bg-emerald-500/20 hover:border-emerald-500/30 disabled:opacity-50"
                        onClick={() => approve(r, true)}
                        disabled={isLoading}
                      >
                        <CheckCircle className="w-3 h-3" />
                        Approve
                      </button>
                      <button 
                        className="btn-glass px-3 py-1.5 text-xs flex items-center gap-1 hover:bg-rose-500/20 hover:border-rose-500/30 disabled:opacity-50"
                        onClick={() => approve(r, false)}
                        disabled={isLoading}
                      >
                        <XCircle className="w-3 h-3" />
                        Reject
                      </button>
                    </div>
                  </td>
                </tr>
              )
            })
          )}
        </tbody>
      </table>
    </div>
  )
}

