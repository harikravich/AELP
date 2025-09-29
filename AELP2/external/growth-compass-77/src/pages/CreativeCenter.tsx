import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Palette, 
  Plus, 
  Code,
  Eye,
  Play,
  Pause,
  BarChart3,
  Target,
  TrendingUp,
  Image,
  Video,
  FileText,
  Copy,
  ExternalLink,
  CheckCircle,
  Clock,
  Zap,
  AlertTriangle,
  Bot,
  Wand2
} from "lucide-react";
import { TopAdsByLP } from "@/components/dashboard/TopAdsByLP";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { AdPreview } from "@/components/ads/AdPreview";
import { useCreatives, useDataset } from "@/hooks/useAelp";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/integrations/aelp-api/client";
import { toast } from "sonner";

export default function CreativeCenter() {
  const ds = useDataset()
  const creatives = useCreatives()
  const [tab, setTab] = useState<'performance'|'generate'|'multimodal'|'pipeline'>('performance')
  const rows = creatives.data?.rows || []
  const topAds = useMemo(()=> rows
    .reduce((acc:any, r:any)=> {
      const k = String(r.ad_id)
      if (!acc[k]) acc[k] = { id: k, ad_group_id: r.ad_group_id, customer_id: r.customer_id, campaign_id: r.campaign_id, impressions:0, clicks:0, cost:0, conversions:0, revenue:0 }
      acc[k].impressions += Number(r.impressions||0)
      acc[k].clicks += Number(r.clicks||0)
      acc[k].cost += Number(r.cost||0)
      acc[k].conversions += Number(r.conversions||0)
      acc[k].revenue += Number(r.revenue||0)
      return acc
    }, {} as Record<string, any>), [rows])
  const sorted = useMemo(()=> Object.values(topAds).sort((a:any,b:any)=> b.revenue - a.revenue).slice(0,10), [topAds])
  const [meta, setMeta] = useState<Record<string, any>>({})
  const [suggestions, setSuggestions] = useState<any[]>([])
  const [variants, setVariants] = useState<any[]>([])
  const [quickH1, setQuickH1] = useState('Protect Your Identity Today')
  const [quickH2, setQuickH2] = useState('Aura® All‑in‑One Digital Security')
  const [quickH3, setQuickH3] = useState('Save Up to 68% — Limited Time')
  const [quickD1, setQuickD1] = useState('Stop scams, block fraud, and monitor your credit — in one app.')
  const [quickD2, setQuickD2] = useState('Try Aura risk‑free. 24/7 US‑based support. Fast setup.')
  const [quickURL, setQuickURL] = useState('https://buy.aura.com/?utm_source=google&utm_medium=cpc&utm_campaign=aelp-test&utm_content=creative&utm_term={keyword}')
  const [quickCamp, setQuickCamp] = useState('')
  const [quickAdg, setQuickAdg] = useState('')
  const [images, setImages] = useState<string[]>([])
  const [previewData, setPreviewData] = useState<any|null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  // Pre-fill Quick Create destination from top performers when available
  useEffect(()=>{
    try {
      const first:any = (sorted||[])[0]
      if (first) {
        if (!quickAdg && first.ad_group_id) setQuickAdg(String(first.ad_group_id))
        if (!quickCamp && first.campaign_id) setQuickCamp(String(first.campaign_id))
      }
    } catch {}
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sorted?.length])
  useEffect(()=> {
    let canceled = false
    async function run() {
      const out: Record<string, any> = {}
      for (const r of sorted) {
        try {
          const d = await api.ads.creative(String(r.id), String(r.customer_id||''), String(r.campaign_id||''))
          out[r.id] = d
        } catch {}
      }
      if (!canceled) setMeta(out)
    }
    run()
    return ()=> { canceled = true }
  }, [sorted])

  const clean = (text: string) => {
    try {
      return text
        .replace(/\{([^:{}]+):([^}]+)\}/g, (_:any,_k:string,fb:string)=> fb)
        .replace(/\{LOCATION\([^)]*\):([^}]+)\}/g, (_:any,fb:string)=> fb)
        .replace(/\{CUSTOMIZER\.[^:}]+:([^}]+)\}/g, (_:any,fb:string)=> fb)
        .replace(/\{[^}]+\}/g,'')
        .replace(/\s+/g,' ').trim()
    } catch { return text }
  }

  // Removed static creative ideas; generator pulls from APIs

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Creative Intelligence Center</h1>
            <p className="text-muted-foreground mt-1">
              AI-powered creative optimization • Generate → Test → Scale winners
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={ds.data?.mode==='prod'?'destructive':'outline'}>
              Dataset: {ds.isLoading?'…':ds.data?.mode}
            </Badge>
            {ds.data?.mode==='prod' && (
              <span className="text-xs text-muted-foreground">Writes blocked (HITL)</span>
            )}
          </div>
          <Button className="bg-primary" onClick={()=> setTab('generate')}>
            <Plus className="w-4 h-4 mr-2" />
            Generate Creative
          </Button>
        </div>

        <Tabs value={tab} onValueChange={(v:any)=> setTab(v)} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="performance">Top Performers</TabsTrigger>
            <TabsTrigger value="generate">AI Generator</TabsTrigger>
            <TabsTrigger value="multimodal">Multi-Modal Hub</TabsTrigger>
            <TabsTrigger value="pipeline">Creative Pipeline</TabsTrigger>
          </TabsList>

          {/* Top Performing Ads */}
          <TabsContent value="performance" className="space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <TopAdsByLP />
              {creatives.isLoading && (
                <Card className="border p-6"><div className="text-sm text-muted-foreground">Loading top ads…</div></Card>
              )}
              {sorted.length===0 && !creatives.isLoading && (
                <Card className="border p-6"><div className="text-sm text-muted-foreground">No ad rows.</div></Card>
              )}
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <div>Top ads by revenue</div>
              </div>
              {sorted.slice(0,5).map((ad: any) => {
                const details = meta[ad.id] || {}
                const ctr = ad.impressions ? (ad.clicks/ad.impressions) : 0
                const cvr = ad.clicks ? (ad.conversions/ad.clicks) : 0
                const cac = ad.conversions ? (ad.cost/ad.conversions) : 0
                const roas = ad.cost ? (ad.revenue/ad.cost) : 0
                const headlines: string[] = (details?.headlines||[]).slice(0,3)
                const descriptions: string[] = (details?.descriptions||[]).slice(0,2)
                const finalUrl: string | undefined = details?.final_urls?.[0]
                const onScale = async () => {
                  try {
                    const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/creative/enqueue`, {
                      method: 'POST',
                      credentials: 'include',
                      headers: { 'content-type': 'application/json' },
                      body: JSON.stringify({
                        platform: 'google_ads',
                        type: 'rsa',
                        campaign_id: ad.campaign_id,
                        ad_group_id: ad.ad_group_id,
                        payload: { action: 'clone_scale', source_ad_id: ad.id, scale: 1.2 },
                        requested_by: 'external_ui'
                      })
                    })
                    const j = await res.json().catch(()=>({}))
                    if (j?.ok) toast.success(`Queued scale (run ${j.run_id})`)
                    else toast.error(j?.error || 'Failed to enqueue')
                  } catch (e:any) { toast.error(String(e?.message||e)) }
                }
                return (
                <Card key={ad.id} className="border shadow-lg hover:shadow-xl transition-all duration-300">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <Badge variant={'default'}>
                            RSA
                          </Badge>
                          <h3 className="font-semibold text-lg">Ad {ad.id}</h3>
                          <span className="text-sm text-muted-foreground">Ad Group: {ad.ad_group_id}</span>
                          <Badge className={'bg-primary/20 text-primary'}>
                            active
                          </Badge>
                        </div>
                        
                        {/* Performance Metrics */}
                        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                          <div>
                            <span className="text-xs text-muted-foreground">CTR</span>
                            <div className="font-semibold text-primary">{(ctr*100).toFixed(2)}%</div>
                          </div>
                          <div>
                            <span className="text-xs text-muted-foreground">CVR</span>
                            <div className="font-semibold">{(cvr*100).toFixed(2)}%</div>
                          </div>
                          <div>
                            <span className="text-xs text-muted-foreground">CAC</span>
                            <div className="font-semibold">${Math.round(cac).toLocaleString('en-US')}</div>
                          </div>
                          <div>
                            <span className="text-xs text-muted-foreground">ROAS</span>
                            <div className="font-semibold text-accent">{roas.toFixed(2)}x</div>
                          </div>
                          <div>
                            <span className="text-xs text-muted-foreground">Impressions</span>
                            <div className="font-semibold">{Number(ad.impressions||0).toLocaleString('en-US')}</div>
                          </div>
                          <div>
                            <span className="text-xs text-muted-foreground">Cost</span>
                            <div className="font-semibold">${Math.round(Number(ad.cost||0)).toLocaleString('en-US')}</div>
                          </div>
                        </div>
                        
                        {/* RSA Creative Preview */}
                        <div className="bg-muted/20 p-4 rounded-lg mb-3">
                          <div className="space-y-3">
                            <div>
                              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Headlines:</span>
                          {headlines.length === 0 ? (
                            <div className="text-sm ml-2 mt-1 text-muted-foreground">Loading…</div>
                              ) : headlines.map((headline, i) => (
                                <div key={i} className="text-sm ml-2 mt-1 p-2 bg-background/50 rounded border-l-2 border-primary/30">
                                  {clean(headline)}
                                </div>
                              ))}
                            </div>
                            <div>
                              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Descriptions:</span>
                              {descriptions.length === 0 ? (
                                <div className="text-sm ml-2 mt-1 text-muted-foreground">Loading…</div>
                              ) : descriptions.map((desc, i) => (
                                <div key={i} className="text-sm ml-2 mt-1 p-2 bg-background/50 rounded border-l-2 border-accent/30 text-muted-foreground">
                                  {clean(desc)}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 text-sm">
                          <TrendingUp className="w-4 h-4 text-primary" />
                          <span className="text-muted-foreground">Recent 28d performance</span>
                        </div>
                      </div>
                      
                      <div className="flex flex-col gap-2 ml-6">
                        <Button size="sm" variant="outline" onClick={async()=>{
                          try {
                            const base = (import.meta as any).env.VITE_API_BASE_URL || ''
                            const url = `${base}/api/ads/creative?ad_id=${encodeURIComponent(String(ad.id))}&customer_id=${encodeURIComponent(String(ad.customer_id||''))}&campaign_id=${encodeURIComponent(String(ad.campaign_id||''))}`
                            const res = await fetch(url, { credentials:'include' })
                            const j = await res.json().catch(()=>({}))
                            if (!res.ok || j?.error) throw new Error(j?.error||'Preview failed')
                            setPreviewData(j); setPreviewOpen(true)
                          } catch(e:any){ toast.error(String(e?.message||e)) }
                        }}>
                          <Eye className="w-4 h-4 mr-2" />
                          Live Preview
                        </Button>
                        <Button size="sm" variant="outline" onClick={async()=>{
                          try {
                            const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/creative/enqueue`, {
                              method:'POST', credentials:'include', headers:{'content-type':'application/json'},
                              body: JSON.stringify({ platform:'google_ads', type:'rsa', campaign_id: ad.campaign_id, ad_group_id: ad.ad_group_id, payload:{ action:'clone_scale', source_ad_id: ad.id, scale: 1.0 }, requested_by:'external_ui' })
                            })
                            const j = await res.json().catch(()=>({}))
                            if (j?.ok) toast.success(`Clone queued (run ${j.run_id})`) ; else toast.error(j?.error||'Failed')
                          } catch(e:any){ toast.error(String(e?.message||e)) }
                        }}>
                          <Copy className="w-4 h-4 mr-2" />
                          Clone Creative
                        </Button>
                        <Button size="sm" variant="outline" asChild>
                          <a href={`https://ads.google.com/aw/adgroups?ocid=0&__u=0&__c=0#~2!addetail=${ad.id}`} target="_blank" rel="noreferrer">
                          <ExternalLink className="w-4 h-4 mr-2" />
                          Edit in Google Ads
                          </a>
                        </Button>
                        <Button size="sm" className="bg-primary" onClick={onScale}>Scale Winner</Button>
                      </div>
                    </div>
                  </div>
                </Card>)
              })}
            </div>
          </TabsContent>

          {/* AI Creative Generator */
          }
          <TabsContent value="generate" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-4">
                  <Bot className="w-6 h-6 text-accent" />
                  <h3 className="text-lg font-semibold">AI Creative Generator</h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Campaign Objective</label>
                    <Input placeholder="e.g., Drive SaaS trial signups for SMB market" className="mt-1" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Key Value Props</label>
                    <Textarea 
                      placeholder="e.g., 30-day results, AI-powered, trusted by 10k+ companies" 
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Target Audience</label>
                    <Input placeholder="e.g., B2B decision makers, 25-55, growth-focused" className="mt-1" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Creative Type</label>
                    <div className="flex gap-2 mt-2">
                      <Button variant="outline" size="sm">
                        <FileText className="w-4 h-4 mr-1" />
                        RSA Text
                      </Button>
                      <Button variant="outline" size="sm">
                        <Image className="w-4 h-4 mr-1" />
                        Display Banner
                      </Button>
                      <Button variant="outline" size="sm">
                        <Video className="w-4 h-4 mr-1" />
                        Video Script
                      </Button>
                    </div>
                  </div>
                  <Button className="w-full bg-primary" onClick={async ()=>{
                    try {
                      const [sug, varr] = await Promise.all([api.copySuggestions(), api.creativeVariants()])
                      const srows = (sug as any)?.rows || []
                      const vrows = (varr as any)?.rows || []
                      // Normalize suggestions: map {suggestion,rationale,campaign_id} → {headline,description,campaign_id}
                      const normSug = srows.map((r:any)=>{
                        let headline = ''
                        let description = ''
                        try {
                          // some pipelines store JSON strings
                          const s = typeof r.suggestion === 'string' ? r.suggestion : (r.suggestion?.text || '')
                          const parsed = JSON.parse(s)
                          if (Array.isArray(parsed?.headlines)) headline = parsed.headlines[0]||''
                          if (Array.isArray(parsed?.descriptions)) description = parsed.descriptions[0]||''
                        } catch {
                          headline = r.suggestion || r.title || ''
                          description = r.rationale || r.description || ''
                        }
                        return { headline, description, campaign_id: r.campaign_id || null }
                      }).filter((x:any)=> x.headline || x.description)
                      // Normalize variants: best‑effort parse of free‑form `text`
                      const normVar = vrows.map((r:any)=>{
                        let headlines:string[] = []
                        let descriptions:string[] = []
                        try {
                          const parsed = JSON.parse(r.text)
                          headlines = parsed.headlines || []
                          descriptions = parsed.descriptions || []
                        } catch {
                          if (typeof r.text === 'string' && r.text.trim()) {
                            headlines = [r.text.trim()]
                          }
                        }
                        return { ad_group_id: r.ad_group_id || null, headlines, descriptions }
                      }).filter((v:any)=> (v.headlines?.length||0) > 0)
                      // Smart fallback if nothing came back
                      const fallback = [
                        { headline: 'Identity Theft Protection That Works', description: 'Monitor breaches, alert fast, and block new account fraud.', campaign_id: null },
                        { headline: 'All‑in‑One Digital Security', description: 'VPN + Antivirus + Password Manager + Credit Lock in one app.', campaign_id: null },
                        { headline: 'Stop Scams Before They Happen', description: 'Real‑time alerts and expert help when you need it most.', campaign_id: null },
                        { headline: 'Protect Your Family Online', description: 'Parental controls, safe browsing, and screen time in one place.', campaign_id: null },
                        { headline: 'Save Up to 68% Today', description: 'Limited‑time discount on Aura’s complete protection.', campaign_id: null },
                      ]
                      setSuggestions(normSug.length ? normSug : fallback)
                      setVariants(normVar)
                      toast.success(`Loaded ${normSug.length || fallback.length} suggestions and ${normVar.length} variants`)
                    } catch(e:any){ toast.error(String(e?.message||e)) }
                  }}>
                    <Wand2 className="w-4 h-4 mr-2" />
                    Generate Suggestions
                  </Button>
                </div>
              </Card>

              <Card className="border p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4">AI-Powered Suggestions</h3>
                <div className="space-y-3">
                  {suggestions.length===0 && <div className="text-sm text-muted-foreground">No suggestions yet. Click “Generate Suggestions”.</div>}
                  {suggestions.slice(0,10).map((s:any,i:number)=> (
                    <div key={i} className="p-3 border rounded">
                      <div className="text-sm font-medium">{s.headline || s.title || 'Suggestion'}</div>
                      <div className="text-xs text-muted-foreground">{s.description || ''}</div>
                      <div className="mt-2">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-2">
                          <Input id={`s_url_${i}`} defaultValue={s.final_url || 'https://buy.aura.com/?utm_source=google&utm_medium=cpc&utm_campaign=aelp-test&utm_content=creative&utm_term={keyword}'} placeholder="Final URL (with UTMs)" />
                          <Input id={`s_camp_${i}`} defaultValue={s.campaign_id||String((sorted||[])[0]?.campaign_id||'')} placeholder="Campaign ID" />
                          <Input id={`s_adg_${i}`} defaultValue={s.ad_group_id||String((sorted||[])[0]?.ad_group_id||'')} placeholder="Ad Group ID" />
                        </div>
                        <Button size="sm" variant="outline" onClick={async ()=>{
                          try {
                            const final_url = (document.getElementById(`s_url_${i}`) as HTMLInputElement).value.trim()
                            const campaign_id = (document.getElementById(`s_camp_${i}`) as HTMLInputElement).value.trim()||null
                            const ad_group_id = (document.getElementById(`s_adg_${i}`) as HTMLInputElement).value.trim()||null
                            const base = (import.meta as any).env.VITE_API_BASE_URL || ''
                            const body = { platform:'google_ads', type:'rsa', campaign_id, ad_group_id, payload:{ action:'create', final_url, headlines:[s.headline||s.title||''], descriptions:[s.description||s.rationale||''] }, requested_by:'external_ui' }
                            const res = await fetch(`${base}/api/control/creative/enqueue`, { method:'POST', credentials:'include', headers:{'content-type':'application/json'}, body: JSON.stringify(body) })
                            const j = await res.json().catch(()=>({}))
                            if(j?.ok) toast.success(`Enqueued (run ${j.run_id})`); else toast.error(j?.error||'Failed')
                          }catch(e:any){ toast.error(String(e?.message||e)) }
                        }}>Enqueue</Button>
                        <Button size="sm" variant="outline" className="ml-2" onClick={()=>{
                          const final_url = (document.getElementById(`s_url_${i}`) as HTMLInputElement).value.trim()
                          setPreviewData({ headlines:[s.headline||s.title||''], descriptions:[s.description||''], final_urls:[final_url] }); setPreviewOpen(true)
                        }}>Preview</Button>
                      </div>
                    </div>
                  ))}
                  {variants.length>0 && (
                    <div className="pt-2 border-t">
                      <div className="text-sm font-semibold mb-2">Variants</div>
                      {variants.slice(0,10).map((v:any,i:number)=> (
                        <div key={i} className="p-3 border rounded mb-2">
                          <div className="text-xs text-muted-foreground">Ad group {v.ad_group_id||'—'}</div>
                          <div className="text-sm">{(v.headlines||[]).join(' • ')}</div>
                          <div className="text-xs">{(v.descriptions||[]).join(' | ')}</div>
                          <div className="mt-2">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-2">
                              <Input id={`v_url_${i}`} defaultValue={'https://buy.aura.com/?utm_source=google&utm_medium=cpc&utm_campaign=aelp-test&utm_content=creative&utm_term={keyword}'} placeholder="Final URL (with UTMs)" />
                              <Input id={`v_camp_${i}`} defaultValue={String((sorted||[])[0]?.campaign_id||'')} placeholder="Campaign ID" />
                              <Input id={`v_adg_${i}`} defaultValue={v.ad_group_id||String((sorted||[])[0]?.ad_group_id||'')} placeholder="Ad Group ID" />
                            </div>
                            <Button size="sm" variant="outline" onClick={async ()=>{
                              try {
                                const final_url = (document.getElementById(`v_url_${i}`) as HTMLInputElement).value.trim()
                                const campaign_id = (document.getElementById(`v_camp_${i}`) as HTMLInputElement).value.trim()||null
                                const ad_group_id = (document.getElementById(`v_adg_${i}`) as HTMLInputElement).value.trim()||null
                                const payload = { action:'create', final_url, headlines: v.headlines||[], descriptions: v.descriptions||[] }
                                const res = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/control/creative/enqueue`, { method:'POST', credentials:'include', headers:{'content-type':'application/json'}, body: JSON.stringify({ platform:'google_ads', type:'rsa', campaign_id, ad_group_id, payload, requested_by:'external_ui' }) })
                                const j=await res.json().catch(()=>({})); if(j?.ok) toast.success(`Enqueued (run ${j.run_id})`); else toast.error(j?.error||'Failed')
                              } catch(e:any){ toast.error(String(e?.message||e)) }
                            }}>Enqueue Variant</Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </Card>

              {/* Quick Create RSA (direct enqueue with UTMs + destination) */}
              <Card className="border p-6 shadow-lg lg:col-span-2">
                <div className="flex items-center gap-3 mb-4">
                  <FileText className="w-5 h-5 text-primary" />
                  <h3 className="text-lg font-semibold">Quick Create RSA</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                  <Input placeholder="Final URL (with UTMs)" value={quickURL} onChange={(e)=> setQuickURL(e.target.value)} />
                  <Input placeholder="Campaign ID" value={quickCamp} onChange={(e)=> setQuickCamp(e.target.value)} />
                  <Input placeholder="Ad Group ID" value={quickAdg} onChange={(e)=> setQuickAdg(e.target.value)} />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                  <Input placeholder="Headline 1" value={quickH1} onChange={(e)=> setQuickH1(e.target.value)} />
                  <Input placeholder="Headline 2" value={quickH2} onChange={(e)=> setQuickH2(e.target.value)} />
                  <Input placeholder="Headline 3" value={quickH3} onChange={(e)=> setQuickH3(e.target.value)} />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                  <Input placeholder="Description 1" value={quickD1} onChange={(e)=> setQuickD1(e.target.value)} />
                  <Input placeholder="Description 2" value={quickD2} onChange={(e)=> setQuickD2(e.target.value)} />
                </div>
                <div className="flex items-center gap-2">
                  <Button onClick={async ()=>{
                    try {
                      if (!quickURL || !quickAdg) { toast.error('Final URL and Ad Group ID required'); return }
                      const base = (import.meta as any).env.VITE_API_BASE_URL || ''
                      const payload = {
                        platform:'google_ads', type:'rsa',
                        campaign_id: quickCamp||null, ad_group_id: quickAdg||null,
                        payload: { action:'create', final_url: quickURL, headlines:[quickH1, quickH2, quickH3].filter(Boolean), descriptions:[quickD1, quickD2].filter(Boolean) },
                        requested_by:'external_ui'
                      }
                      const res = await fetch(`${base}/api/control/creative/enqueue`, { method:'POST', credentials:'include', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) })
                      const j = await res.json().catch(()=>({}))
                      if (j?.ok) { toast.success(`Enqueued (run ${j.run_id})`); setTab('pipeline') } else { toast.error(j?.error||'Failed') }
                    } catch(e:any) { toast.error(String(e?.message||e)) }
                  }}>Enqueue</Button>
                  <Button variant="outline" onClick={()=>{ setPreviewData({ headlines:[quickH1,quickH2,quickH3].filter(Boolean), descriptions:[quickD1,quickD2].filter(Boolean), final_urls:[quickURL] }); setPreviewOpen(true) }}>Preview</Button>
                </div>
                <p className="text-[11px] text-muted-foreground mt-2">Publishes PAUSED to your allowlisted Google Ads account upon Approve. Add UTMs here or in Google Ads before enabling.</p>
              </Card>
            </div>
          </TabsContent>

          {/* Multi-Modal Creative Hub */}
          <TabsContent value="multimodal" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-4">
                  <Image className="w-6 h-6 text-accent" />
                  <h3 className="font-semibold">Visual Assets</h3>
                </div>
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <Image className="w-4 h-4 mr-2" />
                    Product Screenshots
                  </Button>
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Social Proof Graphics
                  </Button>
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <Target className="w-4 h-4 mr-2" />
                    Chart/Data Visualizations
                  </Button>
                  <Button className="w-full bg-accent">
                    <Plus className="w-4 h-4 mr-2" />
                    Generate Image Assets
                  </Button>
                </div>
              </Card>

              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-4">
                  <Video className="w-6 h-6 text-primary" />
                  <h3 className="font-semibold">Video Creative</h3>
                </div>
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <Play className="w-4 h-4 mr-2" />
                    Demo Videos
                  </Button>
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Customer Testimonials
                  </Button>
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <Zap className="w-4 h-4 mr-2" />
                    Explainer Animations
                  </Button>
                  <Button className="w-full bg-primary">
                    <Plus className="w-4 h-4 mr-2" />
                    Create Video Script
                  </Button>
                </div>
              </Card>

              <Card className="border p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-4">
                  <FileText className="w-6 h-6 text-warning" />
                  <h3 className="font-semibold">Copy Variants</h3>
                </div>
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <FileText className="w-4 h-4 mr-2" />
                    Headlines (15 chars)
                  </Button>
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <Code className="w-4 h-4 mr-2" />
                    Descriptions (90 chars)
                  </Button>
                  <Button variant="outline" className="w-full justify-start text-sm">
                    <Target className="w-4 h-4 mr-2" />
                    CTAs & Extensions
                  </Button>
                  <Button className="w-full bg-warning">
                    <Plus className="w-4 h-4 mr-2" />
                    Generate Copy Set
                  </Button>
                </div>
              </Card>
            </div>
          </TabsContent>

          {/* Creative Pipeline (live approvals queue) */}
          <TabsContent value="pipeline" className="space-y-6">
            <PipelineView />
          </TabsContent>

          {/* Multi-Modal Hub */}
          <TabsContent value="multimodal" className="space-y-6">
            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4">Generate Image Assets</h3>
              <div className="flex items-center gap-2 mb-3">
                <Input id="imgprompt" placeholder="e.g., clean product hero on gradient background" />
                <Button onClick={async()=>{
                  const prompt = (document.getElementById('imgprompt') as HTMLInputElement).value
                  if (!prompt) return toast.error('Enter a prompt')
                  try {
                    const j = await fetch(`/api/media/generate`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({ prompt }) }).then(r=>r.json())
                    if (!j?.ok) throw new Error(j?.error||'Failed')
                    setImages(j.images||[])
                  } catch(e:any){ toast.error(String(e?.message||e)) }
                }}>Generate</Button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {images.length===0 && <div className="text-sm text-muted-foreground">No images yet.</div>}
                {images.map((u:string,i:number)=> (
                  <div key={i} className="border rounded p-2"><img src={u} alt="gen" className="w-full h-auto rounded" /></div>
                ))}
              </div>
              <p className="text-[11px] text-muted-foreground mt-2">Generated via OpenAI Images API; ensure OPENAI_API_KEY is set on the server.</p>
            </Card>
            <Card className="border p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-2">Other asset generators</h3>
              <div className="text-sm text-muted-foreground">Video/animation/script generators are disabled until endpoints are configured.</div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Ad Preview</DialogTitle>
          </DialogHeader>
          {previewData ? <AdPreview data={previewData} /> : <div className="text-sm text-muted-foreground">Loading…</div>}
        </DialogContent>
      </Dialog>
    </DashboardLayout>
  );
}

function PipelineView(){
  const [data, setData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const load = async()=>{
    try{ const j = await fetch(`/api/bq/approvals/queue`, { credentials:'include' }).then(r=>r.json()); setData(j.rows||[]) }catch{ setData([]) }finally{ setLoading(false) }
  }
  useEffect(()=>{ load(); const t = setInterval(load, 10000); return ()=> clearInterval(t) },[])
  const grouped = data.reduce((acc:any,r:any)=>{ (acc[r.status||'queued'] ||= []).push(r); return acc }, {})
  const approve = async(run_id:string)=>{
    const fd = new FormData(); fd.append('run_id', run_id)
    const res = await fetch(`/api/control/creative/publish`, { method:'POST', credentials:'include', body: fd })
    const j = await res.json().catch(()=>({})); if(j?.ok) toast.success('Published'); else toast.error(j?.error||'Failed'); load()
  }
  const reject = async(run_id:string)=>{
    const res = await fetch(`/api/bq/approvals/reject`, { method:'POST', credentials:'include', headers:{'content-type':'application/json'}, body: JSON.stringify({ run_id }) })
    const j = await res.json().catch(()=>({})); if(j?.ok) toast.success('Rejected'); else toast.error(j?.error||'Failed'); load()
  }
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {['queued','processed','rejected'].map((st)=> (
        <Card key={st} className="border p-6 shadow-lg">
          <h3 className="text-lg font-semibold mb-4">{st.toUpperCase()} ({(grouped[st]||[]).length})</h3>
          {loading ? <div className="text-sm text-muted-foreground">Loading…</div> : (
            <div className="space-y-3">
              {(grouped[st]||[]).slice(0,20).map((r:any,i:number)=> (
                <div key={i} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="min-w-0">
                    <div className="text-sm font-medium">{r.type?.toUpperCase() || 'item'} • {r.platform || 'platform'}</div>
                    <div className="text-[11px] text-muted-foreground truncate">{r.run_id}</div>
                  </div>
                  {st==='queued' ? (
                    <div className="flex gap-2">
                      <Button size="sm" onClick={()=> approve(r.run_id)}>Approve</Button>
                      <Button size="sm" variant="outline" onClick={()=> reject(r.run_id)}>Reject</Button>
                    </div>
                  ) : null}
                </div>
              ))}
              {(grouped[st]||[]).length===0 && <div className="text-sm text-muted-foreground">No items.</div>}
            </div>
          )}
        </Card>
      ))}
    </div>
  )
}
