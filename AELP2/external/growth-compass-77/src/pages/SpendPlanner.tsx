import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Target, 
  TrendingUp, 
  DollarSign,
  Users,
  AlertCircle,
  CheckCircle,
  ArrowRight
} from "lucide-react";
import { useHeadroom, useMmmAllocations, useKpiDaily, useDataset } from "@/hooks/useAelp";
import { api } from "@/integrations/aelp-api/client";
import { toast } from "sonner";
import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'

export default function SpendPlanner() {
  const ds = useDataset()
  const headroom = useHeadroom()
  const mmm = useMmmAllocations()
  const kpi = useKpiDaily(28)
  const cost = (kpi.data?.rows||[]).reduce((s,r:any)=> s + Number(r.cost||0), 0)
  const conv = (kpi.data?.rows||[]).reduce((s,r:any)=> s + Number(r.conversions||0), 0)
  const cac = conv ? cost/conv : 0
  const recs = (headroom.data?.rows||[])
  const totalRoom = recs.reduce((s:any,r:any)=> s + Number(r.room||0), 0)
  const totalExtra = recs.reduce((s:any,r:any)=> s + Number(r.extra_per_day||0), 0)
  const deploy = async () => {
    try {
      const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/bandit-apply`, { method: 'POST', credentials: 'include', headers: { 'content-type':'application/json' }, body: JSON.stringify({ lookback: 30 }) })
      const j = await res.json().catch(()=>({}))
      if (j?.ok) toast.success('Bandit apply triggered (shadow)')
      else toast.error(j?.error || 'Failed to trigger')
    } catch (e:any) { toast.error(String(e?.message||e)) }
  }
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-foreground">Spend Planner</h1>
          <p className="text-muted-foreground mt-1">
            Deploy budget to highest-ROAS opportunities with MMM guidance
          </p>
          <Badge variant={ds.data?.mode==='prod'?'destructive':'outline'}>
            Dataset: {ds.isLoading?'…':ds.data?.mode}
          </Badge>
        </div>

        {/* Current State */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="p-6 border">
            <div className="flex items-center gap-3 mb-2">
              <DollarSign className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Current Daily Spend</span>
            </div>
            <div className="text-2xl font-bold">${Math.round(cost/28).toLocaleString('en-US')}</div>
            <p className="text-xs text-muted-foreground mt-1">Across all channels</p>
          </Card>

          <Card className="p-6 border">
            <div className="flex items-center gap-3 mb-2">
              <Target className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Current CAC</span>
            </div>
            <div className="text-2xl font-bold">${Math.round(cac).toLocaleString('en-US')}</div>
            <p className="text-xs text-muted-foreground mt-1">Target: $300</p>
          </Card>

          <Card className="p-6 border">
            <div className="flex items-center gap-3 mb-2">
              <TrendingUp className="w-5 h-5 text-primary" />
              <span className="text-sm text-muted-foreground">Available Headroom</span>
            </div>
            <div className="text-2xl font-bold text-primary">+${Math.round(totalRoom).toLocaleString('en-US')}</div>
            <p className="text-xs text-muted-foreground mt-1">Daily capacity</p>
          </Card>
        </div>

        {/* Recommendations */}
        <Card className="border p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">AI Spending Recommendations</h3>
            <Badge className="bg-primary/20 text-primary border-primary/30">
              MMM + Impression Share Analysis
            </Badge>
          </div>

          <div className="space-y-4">
            {recs.map((r:any, i:number)=> (
              <div key={i} className="border rounded-lg p-4 bg-primary/5">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="w-4 h-4 text-primary" />
                      <span className="font-medium">{r.channel}</span>
                      <Badge variant="secondary" className="text-xs">High Confidence</Badge>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Additional Spend:</span>
                        <div className="font-semibold">+${Math.round(Number(r.room||0)).toLocaleString('en-US')}/day</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Est. New Customers:</span>
                        <div className="font-semibold">+{Number(r.extra_per_day||0).toLocaleString('en-US')}/day</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Projected CAC:</span>
                        <div className="font-semibold">${Math.round(Number(r.cac||0)).toLocaleString('en-US')}</div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">ROAS Impact:</span>
                        <div className="font-semibold text-primary">+</div>
                      </div>
                    </div>
                    <div className="mt-2 text-xs text-muted-foreground">Curves: <LoadCurves channel={r.channel} /></div>
                    {typeof r.impression_share === 'number' && (
                      <div className="mt-1 text-[11px] text-muted-foreground">Capped by impression share ≈ {Math.round(Number(r.impression_share||0)*100)}%</div>
                    )}
                  </div>
                  <div className="flex gap-2 ml-4">
                    <Button size="sm" className="" onClick={async ()=> {
                      try {
                        const obj = `Increase ${r.channel} by $${Math.round(Number(r.room||0))}/day`
                        const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/opportunity-approve`, { method: 'POST', credentials: 'include', headers: { 'content-type':'application/json' }, body: JSON.stringify({ action:'approve', objective: obj, notes: 'queued from external UI' }) })
                        const j = await res.json().catch(()=>({}))
                        if (j?.ok) toast.success('Queued for approval')
                        else toast.error(j?.error || 'Failed to queue')
                      } catch (e:any) { toast.error(String(e?.message||e)) }
                    }}>
                      Queue Plan
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                    <Button size="sm" variant="outline" onClick={deploy}>
                      Deploy (Bandit)
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Impact Summary */}
        <Card className="border p-6">
          <h3 className="text-lg font-semibold mb-4">Total Impact Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">+${Math.round(totalRoom).toLocaleString('en-US')}</div>
              <p className="text-sm text-muted-foreground">Additional Daily Spend</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">+{Math.round(totalExtra).toLocaleString('en-US')}</div>
              <p className="text-sm text-muted-foreground">New Customers/Day</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">$278</div>
              <p className="text-sm text-muted-foreground">Blended CAC</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">0.53x</div>
              <p className="text-sm text-muted-foreground">Projected ROAS</p>
            </div>
          </div>
        </Card>
      </div>
      {(recs.length===0) && (
        <Card className="border p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">No MMM allocations found. You can run MMM upstream or queue backfills in Backstage.</div>
            <a href="/backstage" className="text-sm underline">Open Backstage</a>
          </div>
        </Card>
      )}
    </DashboardLayout>
  );
}

function LoadCurves({ channel }:{ channel:string }){
  const [curves, setCurves] = React.useState<any|null>(null)
  React.useEffect(()=>{ let ok=true; api.mmm.curves(channel).then(setCurves).catch(()=>setCurves(null)); return ()=>{ok=false} },[channel])
  if (!curves) return <>—</>
  try {
    const spend: number[] = Array.isArray(curves.spend_grid) ? curves.spend_grid : JSON.parse(String(curves.spend_grid||'[]'))
    const conv: number[] = Array.isArray(curves.conv_grid) ? curves.conv_grid : JSON.parse(String(curves.conv_grid||'[]'))
    if (!spend.length || !conv.length) return <>—</>
    const data = spend.map((s:number,i:number)=> ({ s, c: Number(conv[i]||0) }))
    return (
      <div className="h-20 w-56 inline-block align-middle">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: 4, right: 4, top: 2, bottom: 2 }}>
            <XAxis dataKey="s" hide={true} />
            <YAxis hide={true} />
            <Tooltip formatter={(v:any)=> Math.round(Number(v)).toLocaleString('en-US')} labelFormatter={(l:any)=> `$${Math.round(Number(l)).toLocaleString('en-US')}`} />
            <Line type="monotone" dataKey="c" stroke="#10b981" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    )
  } catch { return <>—</> }
}
