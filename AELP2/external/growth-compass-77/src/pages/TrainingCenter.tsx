import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { GraduationCap, CheckCircle, AlertTriangle } from "lucide-react";
import { useOpsStatus } from "@/hooks/useAelp";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/integrations/aelp-api/client";
import { toast } from "sonner";

export default function TrainingCenter() {
  const status = useOpsStatus()
  const s = status.data || {}
  const flows = useQuery({ queryKey:['ops','flows'], queryFn: async ()=> fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/ops/flows`, { credentials:'include' }).then(r=>r.json()) })
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Training Center</h1>
          <p className="text-muted-foreground mt-1">
            Episodes over time, safety monitoring, and fidelity tracking
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="border p-6 shadow-lg">
            <div className="flex items-center gap-3 mb-4">
              <GraduationCap className="w-5 h-5 text-accent" />
              <h3 className="text-lg font-semibold">Training Status</h3>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Episodes (last run)</span>
                <span className="font-semibold">{s?.episodes ?? '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Success Rate</span>
                <span className="font-semibold text-primary">{s?.success_rate ? `${Math.round(Number(s.success_rate)*100)}%` : '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Safety Score</span>
                <span className="font-semibold text-primary">{s?.safety_score ?? '—'}</span>
              </div>
            </div>
            <div className="mt-4">
              <button className="px-3 py-1 border rounded text-xs" onClick={async ()=>{
                try { const r = await api.ops.status(); toast.success('Status refreshed') } catch (e:any){ toast.error(String(e?.message||e)) }
              }}>Refresh</button>
            </div>
          </Card>

          <Card className="border p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4">Recent Events</h3>
            <div className="space-y-3">
              {(s?.events||[]).length === 0 && <div className="text-sm text-muted-foreground">No recent events.</div>}
              {(s?.events||[]).map((e:any,i:number)=> (
                <div key={i} className="flex items-center gap-3">
                  {String(e.level||'info').toLowerCase()==='warning' ? <AlertTriangle className="w-4 h-4 text-warning"/> : <CheckCircle className="w-4 h-4 text-primary"/>}
                  <span className="text-sm">{e.message || JSON.stringify(e)}</span>
                </div>
              ))}
            </div>
            <div className="mt-4">
              <button className="px-3 py-1 border rounded text-xs" onClick={async ()=>{
                try { const r = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/training-run`, { method:'POST', credentials:'include' }); const j = await r.json(); if(j?.ok) toast.success('Training run started'); else toast.error(j?.error||'Failed') } catch(e:any){ toast.error(String(e?.message||e)) }
              }}>Start Training Run</button>
            </div>
          </Card>

          <Card className="border p-6 shadow-lg lg:col-span-2">
            <h3 className="text-lg font-semibold mb-4">Recent Flows</h3>
            {(flows.data?.rows||[]).length===0 ? (
              <div className="text-sm text-muted-foreground">No recent flows.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm"><thead><tr className="text-left border-b"><th className="py-2">Time</th><th>Flow</th><th>OK</th><th>Failures</th></tr></thead><tbody>
                  {(flows.data?.rows||[]).slice(0,20).map((r:any,i:number)=> (
                    <tr key={i} className="border-b"><td className="py-2">{r.timestamp}</td><td>{r.flow}</td><td>{String(r.ok)}</td><td>{Number(r.failures||0)}</td></tr>
                  ))}
                </tbody></table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}
