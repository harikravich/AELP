import React, { useMemo } from 'react'
import DashboardLayout from '@/components/layout/DashboardLayout'
import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

function useTopPartners(days: number, minPayout: number, limit: number) {
  return useQuery({
    queryKey: ['impact-top-partners', days, minPayout, limit],
    queryFn: async () => {
      const res = await fetch(`/api/bq/impact/top-partners?days=${days}&min_payout=${minPayout}&limit=${limit}`)
      if (!res.ok) throw new Error('Failed to load Impact partners')
      return res.json()
    }
  })
}

function useSeedHarvest() {
  return useQuery({
    queryKey: ['impact-seed-harvest'],
    queryFn: async () => {
      const res = await fetch('/api/bq/impact/seed-harvest')
      if (!res.ok) throw new Error('Failed to load seed-harvest stats')
      return res.json()
    }
  })
}

function usePartnerOpps(params: { days: number; minPayout: number; minDays: number; corr: number; limit: number }) {
  const { days, minPayout, minDays, corr, limit } = params
  return useQuery({
    queryKey: ['impact-partner-opps', days, minPayout, minDays, corr, limit],
    queryFn: async () => {
      const url = `/api/bq/impact/partner-opportunities?days=${days}&min_payout=${minPayout}&min_days=${minDays}&corr=${corr}&limit=${limit}`
      const res = await fetch(url)
      if (!res.ok) throw new Error('Failed to load partner opportunities')
      return res.json()
    }
  })
}

export default function Affiliates() {
  const [days, setDays] = React.useState(28)
  const [minPayout, setMinPayout] = React.useState(1000)
  const [limit, setLimit] = React.useState(20)
  const [selectedPartner, setSelectedPartner] = React.useState<{id:string,name:string}|null>(null)
  const { data, isLoading } = useTopPartners(days, minPayout, limit)
  const { data: sh } = useSeedHarvest()
  const { data: opps } = usePartnerOpps({ days, minPayout, minDays: 14, corr: 0.2, limit: 10 })
  const { data: partnerLag } = useQuery({
    queryKey: ['impact-partner-seed-harvest', selectedPartner?.id, days],
    queryFn: async () => {
      if (!selectedPartner?.id) return { rows: [] }
      const res = await fetch(`/api/bq/impact/partner-seed-harvest?partner_id=${encodeURIComponent(selectedPartner.id)}&days=${days}`)
      if (!res.ok) throw new Error('Failed to load partner seed-harvest')
      return res.json()
    },
    enabled: !!selectedPartner?.id
  })

  const rows = data?.rows || []
  const summary = data?.summary || {}
  const totalPayout = summary.total_payout || 0
  const totalActions = summary.total_actions || 0

  return (
    <DashboardLayout title="Affiliates" subtitle="Top partners • payout • actions • scale/trim">
      <div className="space-y-4">
        <div className="flex items-end gap-3">
          <div>
            <div className="text-xs text-muted-foreground">Lookback (days)</div>
            <Input type="number" value={days} onChange={(e) => setDays(parseInt(e.target.value || '28', 10))} className="w-24" />
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Min payout ($)</div>
            <Input type="number" value={minPayout} onChange={(e) => setMinPayout(parseInt(e.target.value || '0', 10))} className="w-28" />
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Limit</div>
            <Input type="number" value={limit} onChange={(e) => setLimit(parseInt(e.target.value || '20', 10))} className="w-24" />
          </div>
          <Button onClick={() => void 0}>Refresh</Button>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Card>
            <CardHeader><CardTitle>Total Payout</CardTitle></CardHeader>
            <CardContent><div className="text-2xl font-semibold">${(totalPayout||0).toLocaleString()}</div></CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Total Actions</CardTitle></CardHeader>
            <CardContent><div className="text-2xl font-semibold">{(totalActions||0).toLocaleString()}</div></CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Partners</CardTitle></CardHeader>
            <CardContent><div className="text-2xl font-semibold">{rows.length}</div></CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader><CardTitle>Top Partners</CardTitle></CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-muted-foreground">
                    <th className="p-2">Partner</th>
                    <th className="p-2">Payout</th>
                    <th className="p-2">Actions</th>
                    <th className="p-2">Active Days</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    <tr><td className="p-2" colSpan={4}>Loading…</td></tr>
                  ) : rows.length === 0 ? (
                    <tr><td className="p-2" colSpan={4}>No partners in range</td></tr>
                  ) : (
                    rows.map((r: any) => (
                      <tr key={r.partner_id} className="border-t hover:bg-muted cursor-pointer" onClick={() => setSelectedPartner({id: String(r.partner_id), name: r.partner})}>
                        <td className="p-2">{r.partner}</td>
                        <td className="p-2">${(r.payout||0).toLocaleString()}</td>
                        <td className="p-2">{(r.actions||0).toLocaleString()}</td>
                        <td className="p-2">{r.active_days}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Seed → Harvest (corr by lag)</CardTitle></CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-muted-foreground">
                    <th className="p-2">Lag (days)</th>
                    <th className="p-2">Correlation (Impact cost vs next-day enrollments)</th>
                  </tr>
                </thead>
                <tbody>
                  {(sh?.rows || []).map((r: any) => (
                    <tr key={r.lag} className="border-t">
                      <td className="p-2">{r.lag}</td>
                      <td className="p-2">{(r.corr ?? 0).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Top Opportunities</CardTitle></CardHeader>
          <CardContent>
            {opps?.summary?.days && opps.summary.days < 14 ? (
              <div className="text-sm text-muted-foreground mb-3">
                Daily performance not available yet (only {opps.summary.days} points). Enable a daily API report in Impact to compute correlations.
              </div>
            ) : null}
            <div className="grid grid-cols-2 gap-6">
              <div>
                <div className="font-medium mb-2">Scale candidates</div>
                <table className="w-full text-sm">
                  <thead><tr className="text-left text-muted-foreground"><th className="p-2">Partner</th><th className="p-2">Payout</th><th className="p-2">Best lag</th><th className="p-2">Corr</th><th className="p-2">Signal</th><th className="p-2">Action</th></tr></thead>
                  <tbody>
                    {(opps?.scale || []).length === 0 ? (
                      <tr><td className="p-2" colSpan={6}>No candidates yet</td></tr>
                    ) : (
                      opps.scale.map((r: any) => (
                        <tr key={`s-${r.partner_id}`} className="border-t">
                          <td className="p-2">{r.partner}</td>
                          <td className="p-2">${(r.payout||0).toLocaleString()}</td>
                          <td className="p-2">{r.best_lag}</td>
                          <td className="p-2">{(r.corr ?? 0).toFixed(3)}</td>
                          <td className="p-2 w-40"><Progress value={Math.min(100, Math.max(0, Math.abs(r.corr||0)*100))} /></td>
                          <td className="p-2"><a href="/approvals" className="text-primary underline">Review</a></td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
              <div>
                <div className="font-medium mb-2">Trim candidates</div>
                <table className="w-full text-sm">
                  <thead><tr className="text-left text-muted-foreground"><th className="p-2">Partner</th><th className="p-2">Payout</th><th className="p-2">Best lag</th><th className="p-2">Corr</th><th className="p-2">Signal</th><th className="p-2">Action</th></tr></thead>
                  <tbody>
                  {(opps?.trim || []).length === 0 ? (
                    <tr><td className="p-2" colSpan={6}>No candidates yet</td></tr>
                  ) : (
                    opps.trim.map((r: any) => (
                      <tr key={`t-${r.partner_id}`} className="border-t">
                        <td className="p-2">{r.partner}</td>
                        <td className="p-2">${(r.payout||0).toLocaleString()}</td>
                        <td className="p-2">{r.best_lag}</td>
                        <td className="p-2">{(r.corr ?? 0).toFixed(3)}</td>
                        <td className="p-2 w-40"><Progress value={Math.min(100, Math.max(0, Math.abs(r.corr||0)*100))} /></td>
                        <td className="p-2"><a href="/approvals" className="text-primary underline">Review</a></td>
                      </tr>
                    ))
                  )}
                  </tbody>
                </table>
              </div>
            </div>
          </CardContent>
        </Card>

        {selectedPartner && (
          <Card>
            <CardHeader><CardTitle>Partner Seed → Harvest • {selectedPartner.name}</CardTitle></CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-2 text-sm text-muted-foreground">
                <div>ID: {selectedPartner.id}</div>
                <div>Lookback: {days} days</div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-muted-foreground">
                      <th className="p-2">Lag</th>
                      <th className="p-2">Correlation</th>
                      <th className="p-2">Recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(partnerLag?.rows || []).map((r: any) => {
                      const corr = r.corr ?? 0
                      const rec = corr > 0.25 ? 'Scale (increase placements/budget)' : (corr < -0.25 ? 'Trim (reduce/optimize)' : 'Monitor')
                      return (
                        <tr key={r.lag} className="border-t">
                          <td className="p-2">{r.lag}</td>
                          <td className="p-2">{corr.toFixed(3)}</td>
                          <td className="p-2">{rec}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
              <div className="text-xs text-muted-foreground mt-2">Tip: choose partners with positive corr at lags 1–3d for scaling tests.</div>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  )
}
