import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";

export default function Backstage(){
  const [fresh, setFresh] = useState<any>({ rows: [] })
  const [conn, setConn] = useState<any>({ checks: {} })
  const [ds, setDs] = useState<any>({ mode: 'sandbox', dataset: '' })
  useEffect(()=>{
    const base = (import.meta as any).env.VITE_API_BASE_URL || ''
    fetch(`${base}/api/bq/freshness`, { credentials:'include' }).then(r=>r.json()).then(setFresh).catch(()=> setFresh({ rows: [] }))
    fetch(`${base}/api/connections/health`, { credentials:'include' }).then(r=>r.json()).then(setConn).catch(()=> setConn({ checks: {} }))
    fetch(`${base}/api/dataset`, { credentials:'include' }).then(r=>r.json()).then(setDs).catch(()=> setDs({}))
  },[])
  const setMode = async(mode:'sandbox'|'prod')=>{
    const base = (import.meta as any).env.VITE_API_BASE_URL || ''
    await fetch(`${base}/api/dataset?mode=${mode}`, { method:'POST', credentials:'include' })
    const j = await fetch(`${base}/api/dataset`, { credentials:'include' }).then(r=>r.json())
    setDs(j)
  }
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Backstage</h1>
            <p className="text-muted-foreground">Freshness, connections, and dataset controls</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">Project: {conn?.checks?.project || '—'}</Badge>
            <Badge variant="outline">Dataset: {ds?.dataset || '—'}</Badge>
            <Badge variant={conn?.checks?.gatesEnabled ? 'default' : 'secondary'}>
              GATES {conn?.checks?.gatesEnabled ? 'ON' : 'OFF'}
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="border p-6">
            <h3 className="text-lg font-semibold mb-4">Dataset Mode</h3>
            <div className="flex items-center gap-2">
              <Button size="sm" variant={ds?.mode==='sandbox'?'default':'outline'} onClick={()=> setMode('sandbox')}>Sandbox</Button>
              <Button size="sm" variant={ds?.mode==='prod'?'default':'outline'} onClick={()=> setMode('prod')}>Prod</Button>
            </div>
          </Card>

          <Card className="border p-6">
            <h3 className="text-lg font-semibold mb-4">Connections</h3>
            <pre className="text-xs bg-muted p-3 rounded overflow-auto max-h-48">{JSON.stringify(conn?.checks || {}, null, 2)}</pre>
          </Card>

          <Card className="border p-6 lg:col-span-2">
            <h3 className="text-lg font-semibold mb-2">Backfills</h3>
            <p className="text-sm text-muted-foreground mb-3">Trigger upstream jobs to populate wired views. Runs only in sandbox.</p>
            <div className="flex flex-wrap gap-2">
              {[
                { label:'Auctions Policy', path:'/api/control/backfill/policy' },
                { label:'Bid Landscape', path:'/api/control/backfill/bid' },
                { label:'RL Metrics', path:'/api/control/backfill/rl' },
                { label:'LTV Priors', path:'/api/control/backfill/ltv' },
              ].map((b)=> (
                <Button key={b.path} size="sm" variant="outline" onClick={async()=>{
                  try {
                    const base = (import.meta as any).env.VITE_API_BASE_URL || ''
                    const j = await fetch(`${base}${b.path}`, { method:'POST', credentials:'include' }).then(r=>r.json())
                    if (j?.ok) alert(`${b.label} queued: ${j.id}`)
                    else alert(j?.error || 'Failed')
                  } catch(e:any){ alert(String(e?.message||e)) }
                }}>{b.label}</Button>
              ))}
            </div>
          </Card>

          <Card className="border p-6 lg:col-span-2">
            <h3 className="text-lg font-semibold mb-4">Freshness</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm"><thead><tr className="text-left border-b"><th className="py-2">Table</th><th>Max Date</th></tr></thead><tbody>
                {(fresh.rows||[]).map((r:any,i:number)=> (
                  <tr key={i} className="border-b"><td className="py-2">{r.table_name}</td><td>{String(r.max_date||'—')}</td></tr>
                ))}
              </tbody></table>
            </div>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  )
}
