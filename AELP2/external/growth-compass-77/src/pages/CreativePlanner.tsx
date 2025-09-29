import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { DashboardLayout } from '@/components/layout/DashboardLayout'

type BudgetKey = '30000'|'50000'

function usePlanner() {
  return useQuery({
    queryKey: ['planner-forecasts'],
    queryFn: async () => {
      const r = await fetch('/api/planner/forecasts'); if (!r.ok) throw new Error('forecasts');
      const j = await r.json();
      const vs = await fetch('/api/planner/vendor-scores').then(r=>r.json()).catch(()=>({items:[]}))
      return { forecasts: j, vendor: vs }
    }
  })
}

export default function CreativePlanner() {
  const { data, isLoading } = usePlanner()
  const budgets: BudgetKey[] = ['30000','50000']
  const sec = data?.forecasts?.security?.items || []
  const bal = data?.forecasts?.balance?.items || []
  const makePkg = (items:any[], B:BudgetKey, k=8)=>{
    const top = [...items].sort((a,b)=> (b.p_win - a.p_win)).slice(0,k)
    const per = Number(B)/k
    const rows = top.map(r=>({ id:r.creative_id, p:r.p_win, su: r?.budgets?.[B]?.signups?.p50||0, cac:r?.budgets?.[B]?.cac?.p50||0, budget: per }))
    const totals = rows.reduce((t,r)=>({ signups:t.signups + r.su, cac: t.cac + (r.cac*(r.budget/Number(B))) }), {signups:0,cac:0})
    return { rows, totals }
  }

  return (
    <DashboardLayout>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold">Creative Planner</h1>
          <p className="text-sm text-muted-foreground">US-only forecasts • packages for $30k / $50k per day</p>
        </div>
        <div className="flex gap-2">
          <Button asChild variant="outline"><a href="/api/planner/forecasts" target="_blank">Download forecasts JSON</a></Button>
          <Button asChild variant="outline"><a href="/api/planner/vendor-scores" target="_blank">Vendor scores</a></Button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {(['Security','Balance'] as const).map(section => {
          const items = section==='Security'?sec:bal
          return (
            <Card key={section}>
              <CardHeader>
                <CardTitle>{section} — Top 8 by p_win</CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading ? 'Loading…' : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-muted-foreground">
                          <th className="p-2">Creative</th>
                          <th className="p-2">p_win</th>
                          {budgets.map(b => (<th key={b} className="p-2">CAC p50 (${b})</th>))}
                          {budgets.map(b => (<th key={b+':s'} className="p-2">Signups p50 ({Number(b)/1000}k)</th>))}
                        </tr>
                      </thead>
                      <tbody>
                        {items.slice(0,8).map((r:any)=> (
                          <tr key={r.creative_id} className="border-t">
                            <td className="p-2 font-mono">
                              <a className="underline" href={`/api/planner/setup/${encodeURIComponent(r.creative_id)}`} target="_blank" rel="noreferrer">{r.creative_id}</a>
                            </td>
                            <td className="p-2">{(r.p_win??0).toFixed(3)}</td>
                            {budgets.map(b => (<td key={b} className="p-2">${(r.budgets?.[b]?.cac?.p50??0).toFixed(0)}</td>))}
                            {budgets.map(b => (<td key={b+':s'} className="p-2">{(r.budgets?.[b]?.signups?.p50??0).toFixed(0)}</td>))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      <div className="grid grid-cols-2 gap-6 mt-6">
        <Card>
          <CardHeader><CardTitle>Security Package — $30k</CardTitle></CardHeader>
          <CardContent>
            {(() => { const p = makePkg(sec,'30000'); return (
              <>
                <div className="text-sm text-muted-foreground mb-2">Signups p50: {p.totals.signups.toFixed(0)} • CAC p50 (weighted): ${p.totals.cac.toFixed(0)}</div>
                <ol className="list-decimal pl-5 space-y-1">
                  {p.rows.map(r=> (<li key={r.id} className="font-mono">{r.id} — budget ${r.budget.toFixed(0)} • CAC ${r.cac.toFixed(0)} • su {r.su.toFixed(0)}</li>))}
                </ol>
              </>
            )})()}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Balance Package — $30k</CardTitle></CardHeader>
          <CardContent>
            {(() => { const p = makePkg(bal,'30000'); return (
              <>
                <div className="text-sm text-muted-foreground mb-2">Signups p50: {p.totals.signups.toFixed(0)} • CAC p50 (weighted): ${p.totals.cac.toFixed(0)}</div>
                <ol className="list-decimal pl-5 space-y-1">
                  {p.rows.map(r=> (<li key={r.id} className="font-mono">{r.id} — budget ${r.budget.toFixed(0)} • CAC ${r.cac.toFixed(0)} • su {r.su.toFixed(0)}</li>))}
                </ol>
              </>
            )})()}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  )
}
