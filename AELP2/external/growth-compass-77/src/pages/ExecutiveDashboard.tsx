import React from "react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { KPICard } from "@/components/dashboard/KPICard";
import { MetricChart } from "@/components/dashboard/MetricChart";
import { useKpiDaily } from "@/hooks/useAelp";
import { useHeadroom, useGa4Channels, useDataset } from "@/hooks/useAelp";
import { api } from "@/integrations/aelp-api/client";
import { TopAdsByLP } from "@/components/dashboard/TopAdsByLP";
import { 
  DollarSign, 
  Users, 
  Target, 
  TrendingUp,
  Download,
  Calendar,
  Globe,
  Smartphone,
  Monitor
} from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";
import { Skeleton } from "@/components/ui/skeleton";

function useExecutiveData() {
  const [summary, setSummary] = React.useState<any>(null)
  const headroom = useHeadroom()
  const ga4 = useGa4Channels()
  React.useEffect(()=>{ api.kpi.summary().then(setSummary).catch(()=>setSummary(null)) },[])
  return { summary, headroom, ga4 }
}

export default function ExecutiveDashboard() {
  const { summary, headroom, ga4 } = useExecutiveData()
  const ds = useDataset()
  const daily = useKpiDaily(28)
  const cost = Number(summary?.cost||0)
  const conv = Number(summary?.conv||0)
  const revenue = Number(summary?.revenue||0)
  const cac = conv ? cost/conv : 0
  const roas = cost ? revenue/cost : 0
  const prevCost = Number(summary?.prev_cost||0)
  const prevConv = Number(summary?.prev_conv||0)
  const prevRevenue = Number(summary?.prev_revenue||0)
  const d = (a:number,b:number)=> (b? ((a-b)/Math.abs(b))*100 : 0)
  const costDelta = d(cost, prevCost)
  const convDelta = d(conv, prevConv)
  const cacDelta = d(cac, (prevCost/Math.max(1,prevConv)))
  const trendData = (daily.data?.rows||[]).map((r:any)=> ({ name: String(r.date).slice(5), value: Number(r.revenue||0) }))
  const channelRows = (ga4.data?.rows||[]).map((r:any)=> ({ name: r.default_channel_group, value: Number(r.conversions||0) }))
  const channelTotal = channelRows.reduce((s:any,r:any)=> s + Number(r.value||0), 0)
  const channelData = channelRows.map((r:any)=> ({ name: r.name, value: Math.round(100 * Number(r.value||0) / Math.max(1, channelTotal)), color: "#3b82f6" }))
  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-primary-glow to-secondary-glow bg-clip-text text-transparent">
              Executive Dashboard
            </h1>
            <p className="text-muted-foreground mt-2">
              Deep KPI trends, segments, and strategic insights
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant={ds.data?.mode==='prod'?'destructive':'outline'}>
              Dataset: {ds.isLoading?'…':ds.data?.mode}
            </Badge>
            <Button variant="outline" className="flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              Last 28 Days
            </Button>
            <Button className="bg-gradient-primary hover:shadow-glow transition-all duration-300">
              <Download className="w-4 h-4 mr-2" />
              Executive Report
            </Button>
          </div>
        </div>

        {/* Primary KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {summary ? (
            <>
          <KPICard
            title="Total Revenue"
            value={new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(revenue)}
            change={Number.isFinite(costDelta) ? Math.round(costDelta*10)/10 : 0}
            changeLabel="vs prior 28d"
            icon={<DollarSign className="w-5 h-5 text-secondary" />}
            variant="success"
          />
          
          <KPICard
            title="New Customers"
            value={Math.round(conv).toLocaleString('en-US')}
            change={Number.isFinite(convDelta) ? Math.round(convDelta*10)/10 : 0}
            changeLabel="vs prior 28d"
            icon={<Users className="w-5 h-5 text-primary" />}
          />
          
          <KPICard
            title="Blended CAC"
            value={new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(cac)}
            change={Number.isFinite(cacDelta) ? Math.round(cacDelta*10)/10 : 0}
            changeLabel="GA4 purchases"
            icon={<Target className="w-5 h-5 text-warning" />}
            variant="warning"
          />
          
          <KPICard
            title="Blended ROAS"
            value={`${(Math.round(roas*100)/100).toFixed(2)}x`}
            change={0}
            changeLabel="28d attribution"
            icon={<TrendingUp className="w-5 h-5 text-secondary" />}
            variant="success"
          />
          </>
          ) : (
            <>
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
            </>
          )}
        </div>
        <div className="text-xs text-muted-foreground">Source: {summary?.source || '—'} • Dataset: {summary?.dataset || '—'}</div>

        {/* Main Analytics Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Revenue Trend */}
          <div className="lg:col-span-2">
            <MetricChart
              title="Revenue & Spend Trends"
              subtitle="Daily performance with forecasting"
              color="#10b981"
              type="area"
              data={trendData}
              actions={
                <Badge variant="outline">GA4 + Google Ads</Badge>
              }
            />
          </div>

          {/* Channel Distribution */}
          <Card className="bg-gradient-dark border-border shadow-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">
                Channel Mix
              </h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={channelData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {channelData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1e293b', 
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        color: '#f8fafc'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2 mt-4">
                {channelData.map((channel, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: channel.color }} />
                      <span className="text-sm text-muted-foreground">{channel.name}</span>
                    </div>
                    <span className="text-sm font-medium text-foreground">{channel.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>

        {/* Headroom Snapshot */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-gradient-dark border-border shadow-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Headroom</h3>
              <div className="space-y-3">
                {(headroom.data?.rows||[]).slice(0,5).map((r:any,i:number)=> (
                  <div key={i} className="flex items-center justify-between p-3 rounded border border-white/10">
                    <div className="text-sm text-muted-foreground">{r.channel}</div>
                    <div className="text-sm">Room ${Math.round(Number(r.room||0)).toLocaleString('en-US')} • +{Number(r.extra_per_day||0).toLocaleString('en-US')} cust/day @ ${Math.round(Number(r.cac||0)).toLocaleString('en-US')}</div>
                  </div>
                ))}
                {headroom.isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
                {(headroom.data?.rows||[]).length===0 && !headroom.isLoading && <div className="text-sm text-muted-foreground">No headroom rows.</div>}
              </div>
            </div>
          </Card>

          <Card className="bg-gradient-dark border-border shadow-card">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Strategic Insights</h3>
              <div className="space-y-3 text-sm">
                {Number.isFinite(cac) && (
                  <div className="p-3 rounded border border-white/10">
                    <div className="flex items-center gap-2"><Target className="w-4 h-4 text-warning" /><span className="font-medium">CAC</span></div>
                    <div className="text-muted-foreground mt-1">Current ${Math.round(cac).toLocaleString('en-US')} vs prior ${Math.round(prevCost/Math.max(1,prevConv)).toLocaleString('en-US')} ({Math.round(cacDelta)}%).</div>
                  </div>
                )}
                <div className="p-3 rounded border border-white/10">
                  <div className="flex items-center gap-2"><DollarSign className="w-4 h-4 text-primary" /><span className="font-medium">Revenue 28d</span></div>
                  <div className="text-muted-foreground mt-1">${Math.round(revenue).toLocaleString('en-US')} ({Math.round(d(revenue, prevRevenue))}% vs prior).</div>
                </div>
                {channelData.length>0 ? (
                  <div className="p-3 rounded border border-white/10">
                    <div className="flex items-center gap-2"><TrendingUp className="w-4 h-4 text-secondary" /><span className="font-medium">Top GA4 channels</span></div>
                    <div className="text-muted-foreground mt-1">{channelData.slice(0,3).map(c=>c.name).join(', ')}</div>
                  </div>
                ) : (
                  <div className="text-muted-foreground">No GA4 channel rows.</div>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* Top Ads by LP */}
        <TopAdsByLP />
      </div>
    </DashboardLayout>
  );
}
