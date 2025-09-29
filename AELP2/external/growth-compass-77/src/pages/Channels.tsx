import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Radio, Plus, TrendingUp } from "lucide-react";
import { useChannelAttribution, useGa4Channels, useMmmAllocations } from "@/hooks/useAelp";
import { toast } from "sonner";

export default function Channels() {
  const attrib = useChannelAttribution()
  const ga4 = useGa4Channels()
  const mmm = useMmmAllocations()
  const rows = attrib.data?.rows || []
  const ga4rows = ga4.data?.rows || []
  const mmmByChannel = (mmm.data?.rows||[]).reduce((acc:any,r:any)=> { acc[r.channel]=r; return acc }, {})
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Channels</h1>
            <p className="text-muted-foreground mt-1">
              New channel candidates and pilot requests
            </p>
          </div>
          <Button className="bg-accent" onClick={async ()=>{
            try {
              const base = (import.meta as any).env.VITE_API_BASE_URL || ''
              const res = await fetch(`${base}/api/research/channels?create=${encodeURIComponent('TikTok Ads Pilot')}&type=tiktok&score=0.8`, { credentials:'include' })
              const j = await res.json().catch(()=>({}))
              if (j?.ok) toast.success('Pilot request submitted')
              else toast.error(j?.error||'Failed to submit')
            } catch(e:any){ toast.error(String(e?.message||e)) }
          }}>
            <Plus className="w-4 h-4 mr-2" />
            Pilot Request
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="border p-6 shadow-lg">
            <div className="flex items-center gap-3 mb-4">
              <Radio className="w-5 h-5 text-accent" />
              <h3 className="text-lg font-semibold">Active Channels</h3>
            </div>
            <div className="space-y-3">
              {rows.length === 0 && !attrib.isLoading && (
                <div className="text-sm text-muted-foreground">No channel rows.</div>
              )}
              {rows.map((r:any)=> {
                const m = mmmByChannel[r.channel] || {}
                return (
                <div key={r.channel} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <span className="font-medium">{r.channel}</span>
                    <p className="text-xs text-muted-foreground">ROAS {Number(r.roas||0).toFixed(2)}x • CAC ${Math.round(Number(r.cac||0))} {m?.expected_cac ? `• MMM CAC ${Math.round(Number(m.expected_cac||0))}` : ''}</p>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">${Math.round(Number(r.cost||0)/28).toLocaleString('en-US')}/day {m?.proposed_daily_budget ? `→ $${Math.round(Number(m.proposed_daily_budget||0)).toLocaleString('en-US')}`:''}</div>
                    <Badge className="bg-primary/20 text-primary">Active</Badge>
                  </div>
                </div>)
              })}
              {ga4rows.length>0 && (
                <div className="p-3 border rounded-lg">
                  <div className="text-sm font-medium mb-2">GA4 Channels (28d)</div>
                  <ul className="text-xs text-muted-foreground grid grid-cols-2 gap-2">
                    {ga4rows.map((g:any,i:number)=> (
                      <li key={i} className="flex justify-between"><span>{g.default_channel_group}</span><span>{Number(g.conversions||0).toLocaleString('en-US')}</span></li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </Card>

          <Card className="border p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4">Pilot Opportunities</h3>
            <div className="space-y-3">
              <div className="p-3 border rounded-lg bg-accent/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">TikTok Ads</span>
                  <Badge variant="secondary">Recommended</Badge>
                </div>
                <div className="text-sm text-muted-foreground">
                  Est. CAC: $245 • Audience overlap: 23%
                </div>
              </div>
              <div className="p-3 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">LinkedIn Ads</span>
                  <Badge variant="outline">Consider</Badge>
                </div>
                <div className="text-sm text-muted-foreground">
                  Est. CAC: $420 • B2B segment potential
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}
