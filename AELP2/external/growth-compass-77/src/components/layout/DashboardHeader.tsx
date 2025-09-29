import { Bell, Search, User, ChevronDown, Activity, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";
import { useDataset, useKpiDaily } from "@/hooks/useAelp";

export function DashboardHeader() {
  const ds = useDataset()
  const kpi = useKpiDaily(28)
  const rows = kpi.data?.rows || []
  const cost = rows.reduce((s,r:any)=> s + Number(r.cost||0), 0)
  const conv = rows.reduce((s,r:any)=> s + Number(r.conversions||0), 0)
  const revenue = rows.reduce((s,r:any)=> s + Number(r.revenue||0), 0)
  const cac = conv ? cost/conv : 0
  const roas = cost ? revenue/cost : 0
  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <SidebarTrigger />
        
        {/* Quick Status Indicators */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-secondary rounded-full"></div>
            <span className="text-sm text-muted-foreground">Live</span>
          </div>
          <Badge variant="outline" className="text-xs">
            <Activity className="w-3 h-3 mr-1" />
            CAC: {kpi.isLoading? '…' : `$${Math.round(cac).toLocaleString('en-US')}`}
          </Badge>
          <Badge variant="outline" className="text-xs">
            ROAS: {kpi.isLoading? '…' : `${roas.toFixed(2)}x`}
          </Badge>
          <Badge variant="outline" className="text-xs">
            <Database className="w-3 h-3 mr-1" />
            {ds.isLoading ? 'Dataset: …' : `Dataset: ${ds.data?.mode}`}
          </Badge>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
          <Input 
            placeholder="Search campaigns, creatives..." 
            className="pl-10 w-80 bg-background/50"
          />
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="w-5 h-5" />
          <Badge 
            variant="destructive" 
            className="absolute -top-1 -right-1 w-5 h-5 text-xs p-0 flex items-center justify-center"
          >
            3
          </Badge>
        </Button>

        {/* User Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-primary rounded-full flex items-center justify-center">
                <User className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm font-medium">Marketing Team</span>
              <ChevronDown className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuItem>Profile Settings</DropdownMenuItem>
            <DropdownMenuItem>Team Management</DropdownMenuItem>
            <DropdownMenuItem>Data Sources</DropdownMenuItem>
            <DropdownMenuItem>Sign out</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
