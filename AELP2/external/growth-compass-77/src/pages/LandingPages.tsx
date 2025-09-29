import React, { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Plus, 
  Eye, 
  Edit, 
  BarChart3, 
  Zap,
  Split,
  Globe,
  Target,
  Smartphone,
  Monitor,
  Tablet
} from 'lucide-react';
import { DynamicPageBuilder } from '@/components/landing-page/DynamicPageBuilder';
import { ABTestManager } from '@/components/landing-page/ABTestManager';
import { APIIntegrationHub } from '@/components/landing-page/APIIntegrationHub';
import { AICampaignManager } from '@/components/landing-page/AICampaignManager';
// (useState, useEffect) imported above
import { toast } from 'sonner'

export default function LandingPages() {
  const [activeTab, setActiveTab] = useState('builder');
  const [tests, setTests] = useState<any[]>([])
  useEffect(()=>{ fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/bq/lp/tests`).then(r=>r.json()).then(j=> setTests(j.rows||[])).catch(()=> setTests([])) },[])

  // Removed mock "pages" list; live LP tests appear below. Live page hosting not in scope here.

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Landing Page Creator</h1>
            <p className="text-muted-foreground">Build, test, and optimize high-converting landing pages with AI-powered insights</p>
          </div>
          <div className="flex gap-2">
            <Badge variant="outline" className="px-3 py-1">
              <Zap className="h-4 w-4 mr-1" />
              AI-Powered
            </Badge>
            <Button onClick={() => setActiveTab('builder')}>
              <Plus className="h-4 w-4 mr-2" />
              New Page
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid grid-cols-6 w-full">
            <TabsTrigger value="ai-campaigns">🧠 AI Campaigns</TabsTrigger>
            <TabsTrigger value="builder">🎨 Builder</TabsTrigger>
            <TabsTrigger value="live-pages">📄 Live Pages</TabsTrigger>
            <TabsTrigger value="ab-tests">🧪 A/B Tests</TabsTrigger>
            <TabsTrigger value="integrations">⚡ API Hub</TabsTrigger>
            <TabsTrigger value="analytics">📊 Analytics</TabsTrigger>
          </TabsList>

          <TabsContent value="ai-campaigns">
            <AICampaignManager />
          </TabsContent>

          <TabsContent value="builder">
            <DynamicPageBuilder />
          </TabsContent>

          <TabsContent value="ab-tests">
            <div className="space-y-3">
              <div className="p-3 border rounded-lg bg-muted/10">
                <div className="text-sm font-medium mb-2">Start A/B Test</div>
                <div className="flex items-center gap-2 flex-wrap">
                  <input id="lpA" placeholder="LP A URL" className="border rounded px-2 py-1 text-sm w-64" />
                  <input id="lpB" placeholder="LP B URL (optional)" className="border rounded px-2 py-1 text-sm w-64" />
                  <input id="split" defaultValue={0.5} className="border rounded px-2 py-1 text-sm w-24" />
                  <select id="metric" className="border rounded px-2 py-1 text-sm"><option value="cac">CAC</option><option value="roas">ROAS</option></select>
                  <Button size="sm" onClick={async ()=>{
                    const lpA = (document.getElementById('lpA') as HTMLInputElement).value.trim()
                    const lpB = (document.getElementById('lpB') as HTMLInputElement).value.trim()
                    const split = Number((document.getElementById('split') as HTMLInputElement).value)
                    const metric = (document.getElementById('metric') as HTMLSelectElement).value
                    if(!lpA) { toast.error('LP A required'); return }
                    try{
                      const base = (import.meta as any).env.VITE_API_BASE_URL || ''
                      const url = new URL(`${base}/api/control/lp/publish`)
                      url.searchParams.set('lp_a', lpA); if(lpB) url.searchParams.set('lp_b', lpB); url.searchParams.set('traffic_split', String(split)); url.searchParams.set('primary_metric', metric)
                      const j = await fetch(url.toString(), { method:'POST', credentials:'include' }).then(r=>r.json())
                      if (j?.ok) { toast.success(`Test ${j.test_id} started`); }
                      else toast.error(j?.error || 'Failed')
                    }catch(e:any){ toast.error(String(e?.message||e)) }
                  }}>Publish Test</Button>
                </div>
              </div>
              {tests.length===0 ? (
                <div className="text-sm text-muted-foreground">No LP tests found.</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="text-left border-b"><th className="py-2">Test</th><th>LP A</th><th>LP B</th><th>Status</th><th>Start</th></tr></thead>
                    <tbody>
                      {tests.map((t:any,i:number)=> (
                        <tr key={i} className="border-b"><td className="py-2">{t.test_id||'—'}</td><td>{t.lp_a||'—'}</td><td>{t.lp_b||'—'}</td><td>{t.status||'—'}</td><td>{t.start_date||'—'}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="integrations">
            <APIIntegrationHub />
          </TabsContent>

          <TabsContent value="live-pages" className="space-y-4">
            <Card className="border p-6 text-sm text-muted-foreground">No live pages listed. Use the Builder to create pages or manage tests under A/B Tests.</Card>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-2">
                    <Monitor className="w-5 h-5 text-primary" />
                    <span className="text-sm text-muted-foreground">Desktop CVR</span>
                  </div>
                  <div className="text-2xl font-bold">11.2%</div>
                  <p className="text-xs text-muted-foreground">+2.3% vs mobile</p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-2">
                    <Smartphone className="w-5 h-5 text-green-600" />
                    <span className="text-sm text-muted-foreground">Mobile CVR</span>
                  </div>
                  <div className="text-2xl font-bold">8.9%</div>
                  <p className="text-xs text-muted-foreground">Optimization needed</p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-2">
                    <Tablet className="w-5 h-5 text-orange-600" />
                    <span className="text-sm text-muted-foreground">Tablet CVR</span>
                  </div>
                  <div className="text-2xl font-bold">9.7%</div>
                  <p className="text-xs text-muted-foreground">Low traffic volume</p>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-2">
                    <BarChart3 className="w-5 h-5 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Overall CVR</span>
                  </div>
                  <div className="text-2xl font-bold">10.1%</div>
                  <p className="text-xs text-muted-foreground">Target: 12%</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Performance Insights</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="w-4 h-4 text-green-600" />
                    <span className="font-medium text-sm">Top Performer</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Trial signup page converting 23% above average. Consider applying learnings to other pages.
                  </p>
                </div>
                <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Smartphone className="w-4 h-4 text-orange-600" />
                    <span className="font-medium text-sm">Mobile Optimization</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Mobile conversion rate 21% below desktop. Test mobile-specific design improvements.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
}
