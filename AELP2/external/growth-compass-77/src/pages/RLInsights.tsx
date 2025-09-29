import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Activity } from "lucide-react";
import { useOffpolicy, useInterference } from "@/hooks/useAelp";
import { useEffect, useMemo, useState } from "react";
import { ResponsiveContainer, LineChart as RLineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { Link } from "react-router-dom";

export default function RLInsights() {
  const off = useOffpolicy()
  const inter = useInterference()
  const offRows = off.data?.rows || []
  const interRows = inter.data?.rows || []
  const counts = useMemo(() => ({ off: offRows.length, inter: interRows.length }), [offRows.length, interRows.length])
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">RL Insights</h1>
            <p className="text-muted-foreground mt-1">Live RL metrics; no static content.</p>
          </div>
          <Badge className="bg-primary/20 text-primary border-primary/30">
            <Activity className="w-3 h-3 mr-1 animate-pulse" />
            Active
          </Badge>
        </div>

        <Tabs defaultValue="decisions" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="decisions">Decision History</TabsTrigger>
            <TabsTrigger value="posteriors">Posteriors</TabsTrigger>
            <TabsTrigger value="summary">Summary</TabsTrigger>
          </TabsList>

          {/* Decisions tab combines off-policy evals + decisions + interference */}
          <TabsContent value="decisions" className="space-y-4">
            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">Recent Off-policy Evaluations</h3>
              {offRows.length===0 ? (
                <div className="text-sm text-muted-foreground">No evaluation rows.</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm"><thead><tr className="text-left border-b"><th className="py-2">Time</th><th>Policy</th><th>Metric</th><th>Value</th><th></th></tr></thead><tbody>
                    {offRows.slice(0,20).map((r:any,i:number)=> (
                      <tr key={i} className="border-b"><td className="py-2">{r.timestamp}</td><td>{r.policy_name}</td><td>{r.eval_metric}</td><td>{Number(r.value||0).toFixed(4)}</td><td className="text-right"><Link to="/approvals" className="text-primary underline">Open Approvals</Link></td></tr>
                    ))}
                  </tbody></table>
                </div>
              )}
            </Card>

            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">Recent AI Decisions</h3>
              <div className="space-y-3">
                {off.isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
                {offRows.length===0 && !off.isLoading && (
                  <div className="text-sm text-muted-foreground">No decisions yet.</div>
                )}
                {offRows.map((r:any, i:number)=> (
                  <div key={i} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <span className="font-medium text-sm">{r.decision || 'Decision'}</span>
                      <p className="text-xs text-muted-foreground">{r.reason || ''}</p>
                    </div>
                    <div className="text-right">
                      <Badge className="bg-primary/20 text-primary">{r.status || 'logged'}</Badge>
                      <p className="text-xs text-muted-foreground mt-1">{r.timestamp || ''}</p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">Interference Scores</h3>
              {inter.isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
              {interRows.length===0 && !inter.isLoading && (
                <div className="text-sm text-muted-foreground">No interference rows.</div>
              )}
              <div className="space-y-2">
                {interRows.slice(0,10).map((r:any,i:number)=> (
                  <div key={i} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <span className="text-sm">{r.from_channel} ↔ {r.to_channel}</span>
                    </div>
                    <div className="text-right text-xs text-muted-foreground">cannibalization: {Number(r.cannibalization||0).toFixed(3)}</div>
                  </div>
                ))}
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="posteriors" className="space-y-4">
            <PosteriorsChart />
          </TabsContent>

          <TabsContent value="summary" className="space-y-4">
            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-2">Live Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Metric label="Off‑policy rows" value={counts.off} />
                <Metric label="Interference rows" value={counts.inter} />
              </div>
              <p className="text-xs text-muted-foreground mt-2">This page renders only live data. If empty, upstream jobs have not produced rows yet.</p>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
}

function PosteriorsChart(){
  const [rows, setRows] = useState<any[]>([])
  useEffect(()=>{ fetch('/api/bq/arms/posteriors').then(r=>r.json()).then(j=> setRows(j.rows||[])).catch(()=> setRows([])) },[])
  if (!rows.length) return <Card className="border p-6"><div className="text-sm text-muted-foreground">No posterior rows.</div></Card>
  const byArm: Record<string, any[]> = {}
  for (const r of rows) { const k = String(r.arm); (byArm[k] ||= []).push({ ts: r.timestamp, mean: Number(r.mean||0) }) }
  return (
    <Card className="border p-6">
      <h3 className="text-lg font-semibold mb-4">Arms Mean Over Time</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(byArm).map(([arm, data])=> (
          <div key={arm} className="h-48">
            <div className="text-sm font-medium mb-1">{arm}</div>
            <ResponsiveContainer width="100%" height="100%">
              <RLineChart data={data as any[]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ts" hide={true} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="mean" stroke="#3b82f6" dot={false} />
              </RLineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </Card>
  )
}

function Metric({ label, value }:{ label:string, value:number }){
  return (
    <div className="p-3 border rounded">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-semibold">{Number(value||0).toLocaleString('en-US')}</div>
    </div>
  )
}
