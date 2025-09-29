import { 
  DollarSign, 
  Target, 
  TrendingUp, 
  Users, 
  CheckCircle,
  AlertTriangle,
  Filter,
  Calendar
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { KPICard } from "@/components/dashboard/KPICard";
import { MetricChart } from "@/components/dashboard/MetricChart";
import React from "react";
import { useKpiSummary, useKpiDaily } from "@/hooks/useAelp";

export function Overview() {
  const [days, setDays] = React.useState(28)
  const [liveAgo, setLiveAgo] = React.useState('30s ago')
  const kpi = useKpiSummary()
  const daily = useKpiDaily(days)
  const cost = Number(kpi.data?.cost||0)
  const conv = Number(kpi.data?.conv||0)
  const revenue = Number(kpi.data?.revenue||0)
  const cac = conv ? cost/conv : 0
  const roas = cost ? revenue/cost : 0
  const prevCost = Number(kpi.data?.prev_cost||0)
  const prevConv = Number(kpi.data?.prev_conv||0)
  const d = (a:number,b:number)=> (b? ((a-b)/Math.abs(b))*100 : 0)
  const convDelta = d(conv, prevConv)
  const cacDelta = d(cac, (prevCost/Math.max(1,prevConv)))
  React.useEffect(()=>{ const t = setInterval(()=> setLiveAgo('just now'), 30000); return ()=> clearInterval(t) },[])
  const setKpiSource = async(source:'ga4'|'ads')=>{ try { await fetch(`/api/kpi-source?source=${source}`, { method:'POST', credentials:'include' }) } catch {} }
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">
            Performance Intelligence
          </h1>
          <p className="text-muted-foreground mt-1">
            AI-powered growth optimization • Test → Learn → Scale
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={()=> setKpiSource('ga4')}>
            <Filter className="w-4 h-4 mr-2" />
            GA4 Source
          </Button>
          <div className="flex items-center gap-1">
            <Button variant={days===7?'default':'outline'} size="sm" onClick={()=> setDays(7)}>
              <Calendar className="w-4 h-4 mr-2" /> 7 Days
            </Button>
            <Button variant={days===28?'default':'outline'} size="sm" onClick={()=> setDays(28)}>
              <Calendar className="w-4 h-4 mr-2" /> 28 Days
            </Button>
          </div>
          <Badge variant="secondary" className="flex items-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full"></div>
            Live • {liveAgo}
          </Badge>
        </div>
      </div>

      {/* Core KPIs - Match the screenshot layout */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* CAC - Green */}
        <Card className="bg-primary/20 border-primary/30 p-6">
          <div className="flex items-center gap-3 mb-2">
            <DollarSign className="w-5 h-5 text-primary" />
            <span className="text-sm text-muted-foreground">Customer Acquisition Cost</span>
          </div>
          <div className="space-y-1">
            <div className="text-3xl font-bold text-foreground">${Math.round(cac).toLocaleString('en-US')}</div>
            <div className="flex items-center gap-2">
              <Badge variant={cacDelta<=0?'default':'destructive'} className="text-xs px-2 py-0">{Math.round(cacDelta)}%</Badge>
              <span className="text-xs text-muted-foreground">vs prior {days}d • Target: $300</span>
            </div>
          </div>
        </Card>

        {/* ROAS - Yellow/Warning */}
        <Card className="bg-warning/20 border-warning/30 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Target className="w-5 h-5 text-warning" />
            <span className="text-sm text-muted-foreground">Return on Ad Spend</span>
          </div>
          <div className="space-y-1">
            <div className="text-3xl font-bold text-foreground">{(roas||0).toFixed(2)}x</div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">{days}d attribution • Target: 0.50x</span>
            </div>
          </div>
        </Card>

        {/* Daily Spend - Neutral */}
        <Card className="bg-card border p-6">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-accent" />
            <span className="text-sm text-muted-foreground">Daily Spend</span>
          </div>
          <div className="space-y-1">
            <div className="text-3xl font-bold text-foreground">${Math.round(cost/Math.max(1,days)).toLocaleString('en-US')}</div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Spend/day</span>
            </div>
          </div>
        </Card>

        {/* New Customers - Neutral */}
        <Card className="bg-card border p-6">
          <div className="flex items-center gap-3 mb-2">
            <Users className="w-5 h-5 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">New Customers</span>
          </div>
          <div className="space-y-1">
            <div className="text-3xl font-bold text-foreground">{Math.round(conv).toLocaleString('en-US')}</div>
            <div className="flex items-center gap-2">
              <Badge variant={convDelta>=0?'default':'destructive'} className="text-xs px-2 py-0">{Math.round(convDelta)}%</Badge>
              <span className="text-xs text-muted-foreground">{days}d period</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Chart and AI Queue */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <MetricChart
            title="CAC Performance Tracking"
            subtitle="Daily customer acquisition cost vs $300 target line"
            showTarget={true}
            color="hsl(142 71% 45%)"
            type="area"
            actions={
              <Badge className="bg-primary/20 text-primary border-primary/30">
                <div className="w-2 h-2 bg-primary rounded-full mr-2"></div>
                Active
              </Badge>
            }
          />
        </div>
        
        <Card className="border p-6">
          <h3 className="text-lg font-semibold mb-4">AI Optimization Queue</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 bg-primary/10 border border-primary/20 rounded-lg">
              <CheckCircle className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium">Scale high-performer</p>
                <p className="text-xs text-muted-foreground">Campaign 19665 • 84% better CTR</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3 p-3 bg-warning/10 border border-warning/20 rounded-lg">
              <AlertTriangle className="w-4 h-4 text-warning mt-0.5 flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium">Budget reallocation</p>
                <p className="text-xs text-muted-foreground">Shift $5k/day to ROAS leaders</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3 p-3 bg-accent/10 border border-accent/20 rounded-lg">
              <CheckCircle className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium">Auction health optimal</p>
                <p className="text-xs text-muted-foreground">Win rate: 40% (target: 30-60%)</p>
              </div>
            </div>
          </div>
          
          <Button className="w-full mt-4 bg-accent hover:bg-accent/90" asChild>
            <a href="/approvals">Review All Recommendations</a>
          </Button>
        </Card>
      </div>
    </div>
  );
}
