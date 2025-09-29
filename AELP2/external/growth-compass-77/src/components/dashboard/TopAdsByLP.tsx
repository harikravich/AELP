import { useEffect, useMemo, useState } from 'react'
import { Card } from '@/components/ui/card'
import { api } from '@/integrations/aelp-api/client'
import { useCreatives } from '@/hooks/useAelp'

type Row = {
  lp: string
  impressions: number
  clicks: number
  cost: number
  conversions: number
  revenue: number
  ads: number
}

export function TopAdsByLP({ pageSize = 10 }:{ pageSize?: number } = {}) {
  const { data, isLoading } = useCreatives()
  const [lpMap, setLpMap] = useState<Record<string, Row>>({})
  const [page, setPage] = useState(0)

  const top = useMemo(()=> (data?.rows||[]).sort((a:any,b:any)=> Number(b.revenue||0) - Number(a.revenue||0)).slice(0,25), [data])

  useEffect(() => {
    let canceled = false
    async function run() {
      const acc: Record<string, Row> = {}
      // First, aggregate with whatever LP info we already have (fall back to unknown)
      for (const r of top) {
        const fallbackLp = String((r.final_url || r.lp_url || '').toString().split('?')[0] || 'unknown')
        if (!acc[fallbackLp]) acc[fallbackLp] = { lp: fallbackLp, impressions:0, clicks:0, cost:0, conversions:0, revenue:0, ads:0 }
        acc[fallbackLp].impressions += Number(r.impressions||0)
        acc[fallbackLp].clicks += Number(r.clicks||0)
        acc[fallbackLp].cost += Number(r.cost||0)
        acc[fallbackLp].conversions += Number(r.conversions||0)
        acc[fallbackLp].revenue += Number(r.revenue||0)
        acc[fallbackLp].ads += 1
      }
      // Then, try to refine with live Ads preview where available
      for (const r of top) {
        try {
          const d = await api.ads.creative(String(r.ad_id), String(r.customer_id||''), String(r.campaign_id||''))
          const lp = String((d?.final_urls?.[0] || '').split('?')[0] || 'unknown')
          if (lp && lp !== 'unknown') {
            // Move r’s aggregates from its fallback bucket to the refined LP bucket
            const fallbackLp = String((r.final_url || r.lp_url || '').toString().split('?')[0] || 'unknown')
            const amt = { impr: Number(r.impressions||0), clk: Number(r.clicks||0), cost: Number(r.cost||0), conv: Number(r.conversions||0), rev: Number(r.revenue||0) }
            // Ensure refined bucket
            if (!acc[lp]) acc[lp] = { lp, impressions:0, clicks:0, cost:0, conversions:0, revenue:0, ads:0 }
            acc[lp].impressions += amt.impr; acc[lp].clicks += amt.clk; acc[lp].cost += amt.cost; acc[lp].conversions += amt.conv; acc[lp].revenue += amt.rev; acc[lp].ads += 1
            // Subtract from fallback if different
            if (acc[fallbackLp] && fallbackLp !== lp) {
              acc[fallbackLp].impressions -= amt.impr; acc[fallbackLp].clicks -= amt.clk; acc[fallbackLp].cost -= amt.cost; acc[fallbackLp].conversions -= amt.conv; acc[fallbackLp].revenue -= amt.rev; acc[fallbackLp].ads = Math.max(0, acc[fallbackLp].ads - 1)
              // Clean empty
              if (acc[fallbackLp].impressions<=0 && acc[fallbackLp].clicks<=0 && acc[fallbackLp].cost<=0 && acc[fallbackLp].conversions<=0 && acc[fallbackLp].revenue<=0) delete acc[fallbackLp]
            }
          }
        } catch {}
      }
      if (!canceled) setLpMap(acc)
    }
    run()
    return () => { canceled = true }
  }, [top])

  const allRows = Object.values(lpMap).sort((a,b)=> b.revenue - a.revenue)
  const pages = Math.max(1, Math.ceil(allRows.length / pageSize))
  const rows = allRows.slice(page*pageSize, page*pageSize + pageSize)

  return (
    <Card className="border shadow-lg">
      <div className="p-4">
        <div className="text-sm font-semibold mb-2">Top Ads by Landing Page</div>
        {isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
        {rows.length === 0 && !isLoading ? (
          <div className="text-sm text-muted-foreground">No data.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-border">
                  <th className="py-2">Landing Page</th>
                  <th>Impr</th>
                  <th>Clicks</th>
                  <th>Cost</th>
                  <th>Conv</th>
                  <th>Revenue</th>
                  <th>CAC</th>
                  <th>ROAS</th>
                  <th>Ads</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r)=> {
                  const cac = r.conversions ? r.cost/r.conversions : 0
                  const roas = r.cost ? r.revenue/r.cost : 0
                  return (
                    <tr key={r.lp} className="border-b border-white/10">
                      <td className="py-2 max-w-[320px] truncate"><a href={r.lp} target="_blank" rel="noreferrer" className="text-primary hover:underline">{r.lp}</a></td>
                      <td>{r.impressions.toLocaleString('en-US')}</td>
                      <td>{r.clicks.toLocaleString('en-US')}</td>
                      <td>${Math.round(r.cost).toLocaleString('en-US')}</td>
                      <td>{r.conversions.toLocaleString('en-US')}</td>
                      <td>${Math.round(r.revenue).toLocaleString('en-US')}</td>
                      <td>${Math.round(cac).toLocaleString('en-US')}</td>
                      <td>{roas.toFixed(2)}x</td>
                      <td>{r.ads}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="flex items-center justify-end gap-2 mt-2">
              <button className="text-xs border rounded px-2 py-1" disabled={page===0} onClick={()=> setPage(p=> Math.max(0, p-1))}>Prev</button>
              <span className="text-xs">Page {page+1} / {pages}</span>
              <button className="text-xs border rounded px-2 py-1" disabled={page+1>=pages} onClick={()=> setPage(p=> Math.min(pages-1, p+1))}>Next</button>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
