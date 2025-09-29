import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Zap, 
  TrendingUp, 
  Eye, 
  Target,
  Activity,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  DollarSign
} from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

export default function AuctionsMonitor() {
  const [windowMin, setWindowMin] = useState(60)
  const live = useQuery({
    queryKey: ['auctions','minutely',windowMin],
    queryFn: async ()=> {
      const r = await fetch(`/api/bq/auctions/minutely?window=${windowMin}`, { credentials:'include' })
      return await r.json()
    },
    refetchInterval: 10000,
  })
  const mapRow = (r:any)=> ({
    ts: r.minute || r.ts,
    impressions: r.impressions ?? r.auctions ?? 0,
    clicks: r.clicks ?? r.wins ?? 0,
    cpc: r.cpc ?? r.avg_price_paid ?? 0,
    cpm: r.cpm ?? ((Number(r.avg_price_paid||0))*1000/Math.max(1, Number(r.impressions||r.auctions||1)))
  })
  const latest = live.data?.rows?.[0] ? mapRow(live.data.rows[0]) : null
  const [policy, setPolicy] = useState<any[]>([])
  const [alerts, setAlerts] = useState<any[]>([])
  useEffect(()=>{
    fetch('/api/bq/policy-enforcement').then(r=>r.json()).then(j=> setPolicy(j.rows||[])).catch(()=> setPolicy([]))
    fetch('/api/bq/ops-alerts').then(r=>r.json()).then(j=> setAlerts(j.rows||[])).catch(()=> setAlerts([]))
  },[])
  // Removed static liveMetrics; top KPIs render from latest row only (above)

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Live Auctions Monitor</h1>
            <p className="text-muted-foreground mt-1">
              Real-time auction intelligence • Competitor tracking • Bid optimization alerts
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge className="bg-primary/20 text-primary border-primary/30">
              <Activity className="w-3 h-3 mr-1 animate-pulse" />
              Live Data Stream
            </Badge>
            <select className="text-xs border rounded px-2 py-1" value={windowMin} onChange={(e)=> setWindowMin(Number(e.target.value))}>
              <option value={30}>Last 30m</option>
              <option value={60}>Last 60m</option>
              <option value={240}>Last 4h</option>
            </select>
            <Button variant="outline" size="sm" onClick={()=> live.refetch()}>Refresh</Button>
          </div>
        </div>

        {!latest && policy.length===0 && alerts.length===0 && (
          <div className="p-3 border rounded bg-muted/10 text-sm text-muted-foreground">
            No recent auctions data yet. You can queue GA4 and Ads ingests above to populate live tiles.
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="border p-6 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Impressions/min</span>
            </div>
            <div className="text-2xl font-bold mb-1">{latest ? Number(latest.impressions||0).toLocaleString('en-US') : '—'}</div>
          </Card>
          <Card className="border p-6 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Clicks/min</span>
            </div>
            <div className="text-2xl font-bold mb-1">{latest ? Number(latest.clicks||0).toLocaleString('en-US') : '—'}</div>
          </Card>
          <Card className="border p-6 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Avg CPC</span>
            </div>
            <div className="text-2xl font-bold mb-1">{latest ? `$${Number(latest.cpc||0).toFixed(2)}` : '—'}</div>
          </Card>
          <Card className="border p-6 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Avg CPM</span>
            </div>
            <div className="text-2xl font-bold mb-1">{latest ? `$${Number(latest.cpm||0).toFixed(2)}` : '—'}</div>
          </Card>
        </div>

        <Card className="border p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Auction Health & Ingest</h3>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={async()=>{
                try{
                  const j = await fetch('/api/control/ga4-ingest',{method:'POST'}).then(r=>r.json())
                }catch{}
              }}>GA4 Ingest</Button>
              <Button size="sm" variant="outline" onClick={async()=>{
                try{
                  const cid = prompt('Personal CID (10 digits)')?.trim()||''
                  if(!/^\d{10}$/.test(cid)) return
                  const j = await fetch(`/api/control/ads-ingest?only=${cid}`,{method:'POST'}).then(r=>r.json())
                }catch{}
              }}>Ads Ingest</Button>
            </div>
          </div>
          <div className="text-sm text-muted-foreground">Streaming minutely aggregates; use controls to backfill GA4/Ads if stale. <a className="underline" href="https://support.google.com/google-ads/answer/6167122" target="_blank" rel="noreferrer">Learn about policy enforcement</a>.</div>
          </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="border p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4">Policy Enforcement</h3>
            {policy.length===0 ? (
              <div className="text-sm text-muted-foreground">No policy rows.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm"><thead><tr className="text-left border-b"><th className="py-2">Time</th><th>Rule</th><th>Target</th><th>Decision</th></tr></thead><tbody>
                  {policy.slice(0,10).map((p:any,i:number)=> (
                    <tr key={i} className="border-b"><td className="py-2">{p.timestamp}</td><td>{p.rule}</td><td>{p.target}</td><td>{p.decision}</td></tr>
                  ))}
                </tbody></table>
              </div>
            )}
          </Card>
          <Card className="border p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4">Ops Alerts</h3>
            {alerts.length===0 ? (
              <div className="text-sm text-muted-foreground">No alerts.</div>
            ) : (
              <ul className="text-sm space-y-2">
                {alerts.slice(0,10).map((a:any,i:number)=> (
                  <li key={i} className="flex items-center justify-between border-b pb-1"><span>{a.alert}</span><span className="text-xs text-muted-foreground">{a.severity}</span></li>
                ))}
              </ul>
            )}
          </Card>
        </div>

        <Card className="border p-6 shadow-lg">
          <h3 className="text-lg font-semibold mb-4">Bid Landscape (CPC vs Win Rate)</h3>
          <BidLandscape />
        </Card>
      </div>
    </DashboardLayout>
  );
}

function BidLandscape(){
  const [rows, setRows] = useState<any[]>([])
  useEffect(()=>{ fetch('/api/bq/bid-landscape').then(r=>r.json()).then(j=> setRows(j.rows||[])).catch(()=> setRows([])) },[])
  const data = rows.map((r:any)=> ({ cpc: Number(r.cpc||0), win_rate: Number(r.win_rate||0) }))
  if (data.length===0) return <div className="text-sm text-muted-foreground">No bid landscape rows.</div>
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="cpc" tickFormatter={(v)=> `$${Number(v).toFixed(2)}`} />
          <YAxis dataKey="win_rate" tickFormatter={(v)=> `${Math.round(Number(v)*100)}%`} />
          <Tooltip formatter={(v:any, n:any)=> n==='win_rate' ? `${Math.round(Number(v)*100)}%` : `$${Number(v).toFixed(2)}` } />
          <Line type="monotone" dataKey="win_rate" stroke="#10b981" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
