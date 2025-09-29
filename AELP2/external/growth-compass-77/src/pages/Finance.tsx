import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { 
  DollarSign, 
  TrendingUp, 
  Users, 
  Target,
  Calculator,
  PieChart,
  BarChart3,
  TrendingDown,
  AlertCircle,
  CheckCircle
} from "lucide-react";
import { useKpiSummary, useDataset } from "@/hooks/useAelp";
import { toast } from "sonner";
import { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'

export default function Finance() {
  const ds = useDataset()
  const kpi = useKpiSummary()
  const cost = Number(kpi.data?.cost||0)
  const conv = Number(kpi.data?.conv||0)
  const revenue = Number(kpi.data?.revenue||0)
  const cac = conv ? cost/conv : 0
  const roas = cost ? revenue/cost : 0
  const [ltv, setLtv] = useState<number|undefined>(undefined)
  const [channels, setChannels] = useState<string[]>([])
  const [whatIf, setWhatIf] = useState<any|null>(null)
  const [alloc, setAlloc] = useState<any[]>([])
  useEffect(()=>{ fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/bq/ltv/summary`).then(r=>r.json()).then(j=> setLtv(Number(j.ltv_90||0))).catch(()=> setLtv(undefined)) },[])
  useEffect(()=>{ fetch(`/api/bq/mmm/channels`).then(r=>r.json()).then(j=> setChannels(j.channels||[])).catch(()=> setChannels([])) },[])
  useEffect(()=>{ fetch(`/api/bq/mmm/allocations`).then(r=>r.json()).then(j=> setAlloc(j.rows||[])).catch(()=> setAlloc([])) },[])
  const cohortData = [
    { period: "Month 1", ltv: "$89", customers: 2847, revenue: "$253k" },
    { period: "Month 3", ltv: "$267", customers: 2847, revenue: "$760k" },
    { period: "Month 6", ltv: "$456", customers: 2847, revenue: "$1.3M" },
    { period: "Month 12", ltv: "$789", customers: 2847, revenue: "$2.2M" },
    { period: "Month 24", ltv: "$1,247", customers: 2847, revenue: "$3.6M" }
  ];

  const scenarios = [
    {
      name: "Conservative Growth",
      dailySpend: "$320k",
      projectedCAC: "$315",
      projectedLTV: "$1,180",
      paybackPeriod: "3.8 months",
      roas: "3.7x",
      confidence: "95%"
    },
    {
      name: "Aggressive Scale", 
      dailySpend: "$450k",
      projectedCAC: "$340",
      projectedLTV: "$1,180",
      paybackPeriod: "4.1 months",
      roas: "3.5x",
      confidence: "78%"
    },
    {
      name: "Current Trajectory",
      dailySpend: "$300k",
      projectedCAC: "$297",
      projectedLTV: "$1,247",
      paybackPeriod: "3.6 months", 
      roas: "4.2x",
      confidence: "97%"
    }
  ];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Finance Intelligence</h1>
            <p className="text-muted-foreground mt-1">
              LTV modeling, payback analysis, and MMM scenario planning
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={ds.data?.mode==='prod'?'destructive':'outline'}>
              Dataset: {ds.isLoading?'…':ds.data?.mode}
            </Badge>
            <Button className="bg-primary">
              <Calculator className="w-4 h-4 mr-2" />
              Run Scenario
            </Button>
          </div>
        </div>

        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="ltv">LTV Analysis</TabsTrigger>
            <TabsTrigger value="scenarios">MMM Scenarios</TabsTrigger>
          </TabsList>

          {/* Overview */}
          <TabsContent value="overview" className="space-y-4">
            {!ltv && (
              <div className="p-3 border rounded bg-muted/10 text-sm text-muted-foreground">
                No LTV rows yet. Once ltv_priors_daily is populated, the tile will show values.
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-2">
                  <DollarSign className="w-5 h-5 text-primary" />
                  <span className="text-sm text-muted-foreground">Customer LTV</span>
                </div>
                <div className="text-2xl font-bold text-primary">{ltv ? `$${Math.round(ltv).toLocaleString('en-US')}` : '—'}</div>
                <p className="text-xs text-muted-foreground">90‑day prior</p>
              </Card>

              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-2">
                  <Target className="w-5 h-5 text-warning" />
                  <span className="text-sm text-muted-foreground">Blended CAC</span>
                </div>
                <div className="text-2xl font-bold">{kpi.isLoading ? '…' : `$${Math.round(cac).toLocaleString('en-US')}`}</div>
                <p className="text-xs text-muted-foreground">All channels weighted</p>
              </Card>

              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-2">
                  <TrendingUp className="w-5 h-5 text-accent" />
                  <span className="text-sm text-muted-foreground">LTV:CAC Ratio</span>
                </div>
                <div className="text-2xl font-bold text-accent">{kpi.isLoading ? '…' : `${roas.toFixed(2)}x`}</div>
                <p className="text-xs text-muted-foreground">Target: 3.0x+</p>
              </Card>

              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-2">
                  <Users className="w-5 h-5 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Gross Margin</span>
                </div>
                <div className="text-2xl font-bold">67%</div>
                <p className="text-xs text-muted-foreground">After COGS</p>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="border p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4">Revenue Attribution</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Google Ads</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-muted rounded-full">
                        <div className="w-16 h-2 bg-primary rounded-full"></div>
                      </div>
                      <span className="text-sm font-semibold">67%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Facebook Ads</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-muted rounded-full">
                        <div className="w-6 h-2 bg-accent rounded-full"></div>
                      </div>
                      <span className="text-sm font-semibold">23%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Organic/Direct</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-muted rounded-full">
                        <div className="w-2 h-2 bg-warning rounded-full"></div>
                      </div>
                      <span className="text-sm font-semibold">10%</span>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="border p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4">Payback (Simple)</h3>
                <div className="space-y-2">
                  <div className="flex justify-between"><span className="text-sm text-muted-foreground">Blended CAC</span><span className="font-semibold">${Math.round(cac).toLocaleString('en-US')}</span></div>
                  <div className="flex justify-between"><span className="text-sm text-muted-foreground">LTV (90‑day prior)</span><span className="font-semibold">{ltv?`$${Math.round(ltv).toLocaleString('en-US')}`:'—'}</span></div>
                  <div className="flex justify-between"><span className="text-sm text-muted-foreground">Payback Ratio</span><span className="font-semibold">{ltv? (ltv/Math.max(1,cac)).toFixed(2)+'x':'—'}</span></div>
                </div>
                <p className="text-[11px] text-muted-foreground mt-2">Labels: CAC from summary; LTV from ltv_priors_daily.</p>
              </Card>

              <Card className="border p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4">Identity Health</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-muted/20 rounded-lg">
                    <div className="text-xs text-muted-foreground">Event Match Quality (Meta)</div>
                    <div className="text-lg font-semibold">N/A</div>
                  </div>
                  <div className="p-3 bg-muted/20 rounded-lg">
                    <div className="text-xs text-muted-foreground">Recovered Audiences</div>
                    <div className="text-lg font-semibold">N/A</div>
                  </div>
                </div>
                <div className="mt-4 flex gap-2">
                  <Button size="sm" variant="outline" onClick={async ()=> {
                    try {
                      const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/value-upload/meta`, { method:'POST', credentials:'include' })
                      const j = await res.json().catch(()=>({}))
                      if (j?.ok) toast.success('Triggered Meta value upload')
                      else toast.error(j?.error || 'Failed to trigger')
                    } catch (e:any) { toast.error(String(e?.message||e)) }
                  }}>Meta Value Upload</Button>
                  <Button size="sm" variant="outline" onClick={async ()=> {
                    try {
                      const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/value-upload/google`, { method:'POST', credentials:'include' })
                      const j = await res.json().catch(()=>({}))
                      if (j?.ok) toast.success('Triggered Google value upload')
                      else toast.error(j?.error || 'Failed to trigger')
                    } catch (e:any) { toast.error(String(e?.message||e)) }
                  }}>Google Value Upload</Button>
                </div>
                <p className="text-[11px] text-muted-foreground mt-2">First-party identifiers are hashed client-side only via server APIs; no raw PII stored in the browser.</p>
              </Card>
            </div>
          </TabsContent>

          {/* LTV Analysis */}
          <TabsContent value="ltv" className="space-y-4">
            <Card className="border p-6 shadow-lg">
              <div className="text-sm text-muted-foreground">No per‑channel LTV endpoint yet. Overview shows LTV 90‑day prior.</div>
            </Card>
          </TabsContent>

          {/* MMM Scenarios */}
          <TabsContent value="scenarios" className="space-y-4">
            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">What‑if Projection</h3>
              <div className="flex items-center gap-2 mb-3">
                <select className="border rounded px-2 py-1 text-sm" id="ch">
                  {channels.map((c)=> <option key={c} value={c}>{c}</option>)}
                </select>
                <input className="border rounded px-2 py-1 text-sm w-32" id="bud" defaultValue={10000} />
                <Button size="sm" onClick={async ()=>{
                  const ch = (document.getElementById('ch') as HTMLSelectElement).value
                  const bud = Number((document.getElementById('bud') as HTMLInputElement).value)
                  try {
                    const j = await fetch(`/api/bq/mmm/whatif?channel=${encodeURIComponent(ch)}&budget=${bud}`).then(r=>r.json())
                    if (j?.error) throw new Error(j.error)
                    setWhatIf(j)
                  } catch(e:any){ toast.error(String(e?.message||e)) }
                }}>Project</Button>
              </div>
              {!whatIf ? (
                <div className="text-sm text-muted-foreground">Enter a budget to project conversions and CAC.</div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-3 border rounded">
                    <div className="text-xs text-muted-foreground">Channel</div>
                    <div className="font-semibold">{whatIf.channel}</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-xs text-muted-foreground">Predicted Conversions</div>
                    <div className="font-semibold text-primary">{Math.round(Number(whatIf.predicted_conversions||0)).toLocaleString('en-US')}</div>
                  </div>
                  <div className="p-3 border rounded">
                    <div className="text-xs text-muted-foreground">Projected CAC</div>
                    <div className="font-semibold">${Math.round(Number(whatIf.cac||0)).toLocaleString('en-US')}</div>
                  </div>
                </div>
              )}
            </Card>
            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">Per‑channel MMM</h3>
              {(alloc||[]).length===0 ? (
                <div className="text-sm text-muted-foreground">No allocations present.</div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {alloc.slice(0,9).map((r:any,i:number)=> (
                    <div key={i} className="p-3 border rounded">
                      <div className="font-medium">{r.channel}</div>
                      <div className="text-xs text-muted-foreground">Proposed Budget</div>
                      <div className="font-semibold">${Math.round(Number(r.proposed_daily_budget||0)).toLocaleString('en-US')}/day</div>
                      <div className="text-xs text-muted-foreground mt-1">Expected CAC</div>
                      <div className="font-semibold">${Math.round(Number(r.expected_cac||0)).toLocaleString('en-US')}</div>
                    </div>
                  ))}
                </div>
              )}
              <p className="text-[11px] text-muted-foreground mt-2">Labels: attribution source noted in header; MMM curves drive these expectations.</p>
            </Card>
          </TabsContent>

          {/* Cohort Analysis */}
          <TabsContent value="cohorts" className="space-y-4">
            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">September 2022 Cohort (2,847 customers)</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Period</th>
                      <th className="text-left p-2">Cumulative LTV</th>
                      <th className="text-left p-2">Active Customers</th>
                      <th className="text-left p-2">Revenue Generated</th>
                      <th className="text-left p-2">Retention Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cohortData.map((row, i) => (
                      <tr key={i} className="border-b">
                        <td className="p-2 font-medium">{row.period}</td>
                        <td className="p-2 font-semibold text-primary">{row.ltv}</td>
                        <td className="p-2">{row.customers}</td>
                        <td className="p-2 font-semibold">{row.revenue}</td>
                        <td className="p-2">
                          <Badge variant="outline">
                            {Math.round((2847 / 2847) * 100 - i * 8)}%
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </TabsContent>

          {/* MMM Scenarios */}
          <TabsContent value="scenarios" className="space-y-4">
            <div className="space-y-4">
              {scenarios.map((scenario, i) => (
                <Card key={i} className="border shadow-lg p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <PieChart className="w-5 h-5 text-accent" />
                        <h3 className="font-semibold text-lg">{scenario.name}</h3>
                        <Badge variant="outline">{scenario.confidence} confidence</Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div>
                          <span className="text-xs text-muted-foreground">Daily Spend</span>
                          <div className="font-semibold text-accent">{scenario.dailySpend}</div>
                        </div>
                        <div>
                          <span className="text-xs text-muted-foreground">Projected CAC</span>
                          <div className="font-semibold">{scenario.projectedCAC}</div>
                        </div>
                        <div>
                          <span className="text-xs text-muted-foreground">Expected LTV</span>
                          <div className="font-semibold text-primary">{scenario.projectedLTV}</div>
                        </div>
                        <div>
                          <span className="text-xs text-muted-foreground">Payback Period</span>
                          <div className="font-semibold">{scenario.paybackPeriod}</div>
                        </div>
                        <div>
                          <span className="text-xs text-muted-foreground">ROAS</span>
                          <div className="font-semibold text-primary">{scenario.roas}</div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex flex-col gap-2 ml-4">
                      <Button size="sm" className="bg-primary">
                        Model Scenario
                      </Button>
                      <Button size="sm" variant="outline">
                        View Details
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>

            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">Custom Scenario Builder</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium">Daily Spend Target</label>
                  <Input placeholder="$400,000" />
                </div>
                <div>
                  <label className="text-sm font-medium">CAC Ceiling</label>
                  <Input placeholder="$350" />
                </div>
                <div>
                  <label className="text-sm font-medium">Time Horizon</label>
                  <Input placeholder="12 months" />
                </div>
              </div>
              <Button className="mt-4 bg-accent">
                Generate Scenario
              </Button>
            </Card>
          </TabsContent>

          {/* Removed static Payback Models tab */}
        </Tabs>
      </div>
    </DashboardLayout>
  );
}
