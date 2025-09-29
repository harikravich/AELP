import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export default function Audiences(){
  const sync = async()=>{
    try {
      const base = (import.meta as any).env.VITE_API_BASE_URL || ''
      const r = await fetch(`${base}/api/control/audience/sync`, { method:'POST', credentials:'include' })
      const j = await r.json()
      if (j?.ok) toast.success('Audience sync queued')
      else toast.error(j?.error || 'Failed')
    } catch(e:any){ toast.error(String(e?.message||e)) }
  }
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Audiences</h1>
        <Card className="border p-6">
          <h3 className="text-lg font-semibold mb-2">Sync Audiences</h3>
          <p className="text-sm text-muted-foreground mb-3">Push latest high‑LTV segments to ad platforms (shadow mode).</p>
          <Button onClick={sync}>Sync Now</Button>
        </Card>
      </div>
    </DashboardLayout>
  )
}

