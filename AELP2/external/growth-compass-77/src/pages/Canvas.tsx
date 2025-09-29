import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { toast } from "sonner";

export default function Canvas(){
  const [items, setItems] = useState<any[]>([])
  const load = async ()=>{
    try {
      const base = (import.meta as any).env.VITE_API_BASE_URL || ''
      const j = await fetch(`${base}/api/canvas/list`, { credentials:'include' }).then(r=>r.json())
      setItems(j.items||[])
    } catch { setItems([]) }
  }
  useEffect(()=>{ load() },[])
  const unpin = async(id:string)=>{
    try {
      const base = (import.meta as any).env.VITE_API_BASE_URL || ''
      const fd = new FormData(); fd.append('id', id)
      const j = await fetch(`${base}/api/canvas/unpin`, { method:'POST', credentials:'include', body: fd }).then(r=>r.json())
      if (j?.ok) { toast.success('Unpinned'); load() } else toast.error(j?.error||'Failed')
    } catch(e:any){ toast.error(String(e?.message||e)) }
  }
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Canvas</h1>
        <Card className="border p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {items.length===0 && <div className="text-sm text-muted-foreground">No pins.</div>}
            {items.map((it:any)=> (
              <div key={it.id} className="border rounded p-3">
                <div className="flex items-center justify-between mb-2"><span className="font-medium text-sm">{it.title}</span><Button size="sm" variant="outline" onClick={()=> unpin(it.id)}>Unpin</Button></div>
                <pre className="text-[11px] bg-muted p-2 rounded overflow-auto max-h-40">{JSON.stringify(it.viz, null, 2)}</pre>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </DashboardLayout>
  )
}

