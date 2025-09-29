import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Clock, 
  CheckCircle, 
  XCircle,
  AlertTriangle,
  Users,
  DollarSign,
  Palette,
  Target
} from "lucide-react";
import { useApprovalsQueue, useDataset } from "@/hooks/useAelp";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { toast } from "sonner";

export default function Approvals() {
  const ds = useDataset()
  const q = useApprovalsQueue()
  const [status, setStatus] = useState<'queued'|'processed'|'rejected'|'any'>('queued')
  const [type, setType] = useState<string>('any')
  const filtered = useQuery({
    queryKey: ['approvals','queue',status],
    queryFn: async ()=> {
      const r = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/bq/approvals/queue?status=${status}`, { credentials:'include' })
      return await r.json()
    },
    staleTime: 5000,
    refetchInterval: 10000,
  })
  const rows = (filtered.data?.rows || q.data?.rows || [])
    .filter((r:any)=> type==='any' ? true : String(r.type||'')===type)
  const pendingCount = rows.length
  const approve = async (run_id: string) => {
    try {
      const fd = new FormData()
      fd.append('run_id', run_id)
      const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/creative/publish`, { method: 'POST', credentials: 'include', body: fd })
      const j = await res.json().catch(()=>({}))
      if (j?.ok) toast.success(`Applied ${run_id}`)
      else toast.error(j?.error || 'Failed to apply')
      q.refetch()
    } catch (e:any) { toast.error(String(e?.message||e)) }
  }
  const reject = async (run_id: string) => {
    try {
      const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/bq/approvals/reject`, {
        method: 'POST', credentials: 'include', headers: { 'content-type':'application/json' }, body: JSON.stringify({ run_id })
      })
      const j = await res.json().catch(()=>({}))
      if (j?.ok) toast.success(`Rejected ${run_id}`)
      else toast.error(j?.error || 'Failed to reject')
      q.refetch()
    } catch (e:any) { toast.error(String(e?.message||e)) }
  }
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Human-in-the-Loop Approvals</h1>
            <p className="text-muted-foreground mt-1">
              Safety gate for all AI-proposed changes • Zero auto-publish
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={ds.data?.mode==='prod'?'destructive':'outline'}>
              Dataset: {ds.isLoading?'…':ds.data?.mode}
            </Badge>
            <select className="text-xs border rounded px-2 py-1" value={status} onChange={async (e)=>{
              const v = e.target.value as any
              setStatus(v)
              // Refetch with server-side filter
              try { await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/bq/approvals/queue?status=${v}`, { credentials:'include' }) } catch {}
              q.refetch()
            }}>
              <option value="queued">Queued</option>
              <option value="processed">Processed</option>
              <option value="rejected">Rejected</option>
              <option value="any">Any</option>
            </select>
            <select className="text-xs border rounded px-2 py-1" value={type} onChange={(e)=> setType(e.target.value)}>
              <option value="any">All Types</option>
              <option value="rsa">RSA</option>
              <option value="pmax">PMax</option>
            </select>
            <Button size="sm" variant="outline" onClick={()=> q.refetch()}>Refresh</Button>
            <Badge variant="destructive" className="text-lg px-2 py-1">
              {q.isLoading ? '…' : `${pendingCount} Pending`}
            </Badge>
          </div>
        </div>

        {/* Pending Approvals */}
        <Card className="border p-6">
          <h3 className="text-lg font-semibold mb-4">Pending Review</h3>
          
          <div className="space-y-4">
            {rows.length === 0 && !q.isLoading && (
              <div className="text-sm text-muted-foreground">No pending items.</div>
            )}
            {rows.map((r:any)=> {
              const payload = (()=>{ try { return typeof r.payload==='string' ? JSON.parse(r.payload) : r.payload } catch { return {} } })()
              const source_ad_id = payload?.source_ad_id || payload?.ad_id
              const previewUrl = source_ad_id ? `${(import.meta as any).env.VITE_API_BASE_URL || ''}/creative-studio/preview?ad_id=${encodeURIComponent(source_ad_id)}${r.campaign_id?`&campaign_id=${encodeURIComponent(r.campaign_id)}`:''}` : ''
              const finalUrl = payload?.final_url || ''
              return (
              <div key={r.run_id} className="border rounded-lg p-4 bg-warning/5">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <Palette className="w-5 h-5 text-warning" />
                      <span className="font-medium">{r.platform} {r.type} request</span>
                      <Badge variant="outline" className="text-xs">Queued</Badge>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground">
                        Campaign {r.campaign_id || '—'} • Ad Group {r.ad_group_id || '—'} • Requested by {r.requested_by || '—'}
                      </p>
                      {finalUrl && (
                        <p className="text-xs"><a href={finalUrl} target="_blank" rel="noreferrer" className="text-primary underline">LP: {finalUrl}</a></p>
                      )}
                      {previewUrl && (
                        <p className="text-xs"><a href={previewUrl} target="_blank" rel="noreferrer" className="text-primary underline">Preview source ad</a></p>
                      )}
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        <span>{new Date(r.enqueued_at).toLocaleString()}</span>
                        <span className="ml-2 text-[11px] opacity-75">run: {r.run_id}</span>
                      </div>
                      {payload && Object.keys(payload).length>0 && (
                        <pre className="text-[11px] bg-background/50 p-2 rounded border overflow-auto max-h-32">{JSON.stringify(payload, null, 2)}</pre>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <Button variant="outline" size="sm" onClick={()=> reject(r.run_id)}>
                      <XCircle className="w-4 h-4 mr-2" />
                      Reject
                    </Button>
                    <Button size="sm" className="bg-primary" onClick={()=> approve(r.run_id)}>
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Approve
                    </Button>
                  </div>
                </div>
              </div>)
            })}

            {/* Budget Increase */}
            <div className="border rounded-lg p-4 bg-accent/5">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <DollarSign className="w-5 h-5 text-accent" />
                    <span className="font-medium">Budget Increase Request</span>
                    <Badge className="text-xs bg-accent/20 text-accent">Medium Priority</Badge>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      Increase Google Search daily budget by $5,000
                    </p>
                    <div className="flex items-center gap-4 text-sm">
                      <span>Expected CAC: <strong>$278</strong></span>
                      <span>Est. new customers: <strong className="text-primary">+4,100/day</strong></span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span>Submitted 1h ago by Spend Planner</span>
                    </div>
                  </div>
                </div>
                <div className="flex gap-2 ml-4">
                  <Button variant="outline" size="sm">
                    <XCircle className="w-4 h-4 mr-2" />
                    Reject
                  </Button>
                  <Button size="sm" className="bg-accent">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Approve
                  </Button>
                </div>
              </div>
            </div>

            {/* Audience Targeting */}
            <div className="border rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <Users className="w-5 h-5 text-muted-foreground" />
                    <span className="font-medium">Audience Update</span>
                    <Badge variant="secondary" className="text-xs">Low Priority</Badge>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      Expand similar audiences based on high-LTV customers
                    </p>
                    <div className="flex items-center gap-4 text-sm">
                      <span>Reach expansion: <strong>+2.4M users</strong></span>
                      <span>Confidence: <strong>Medium</strong></span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span>Submitted 45m ago by RL Agent</span>
                    </div>
                  </div>
                </div>
                <div className="flex gap-2 ml-4">
                  <Button variant="outline" size="sm">
                    <XCircle className="w-4 h-4 mr-2" />
                    Reject
                  </Button>
                  <Button variant="outline" size="sm">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Approve
                  </Button>
                </div>
              </div>
            </div>

            {/* Bid Strategy */}
            <div className="border rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <Target className="w-5 h-5 text-muted-foreground" />
                    <span className="font-medium">Bid Strategy Adjustment</span>
                    <Badge variant="secondary" className="text-xs">Low Priority</Badge>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      Switch to Target CPA with $285 target across 5 campaigns
                    </p>
                    <div className="flex items-center gap-4 text-sm">
                      <span>Expected volume: <strong>+15%</strong></span>
                      <span>CAC impact: <strong className="text-primary">-$12</strong></span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span>Submitted 30m ago by Auction Monitor</span>
                    </div>
                  </div>
                </div>
                <div className="flex gap-2 ml-4">
                  <Button variant="outline" size="sm">
                    <XCircle className="w-4 h-4 mr-2" />
                    Reject
                  </Button>
                  <Button variant="outline" size="sm">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Approve
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Recent Activity */}
        <Card className="border p-6">
          <h3 className="text-lg font-semibold mb-4">Recent Decisions</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-primary/5 rounded-lg">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-4 h-4 text-primary" />
                <span className="text-sm">Creative A/B test approved</span>
              </div>
              <span className="text-xs text-muted-foreground">5m ago</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-destructive/5 rounded-lg">
              <div className="flex items-center gap-3">
                <XCircle className="w-4 h-4 text-destructive" />
                <span className="text-sm">Audience expansion rejected</span>
              </div>
              <span className="text-xs text-muted-foreground">1h ago</span>
            </div>
          </div>
        </Card>
      </div>
    </DashboardLayout>
  );
}
