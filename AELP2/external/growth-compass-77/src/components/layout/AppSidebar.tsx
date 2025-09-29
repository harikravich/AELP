import { NavLink, useLocation } from "react-router-dom";
import {
  BarChart3,
  Brain,
  Palette,
  FileText,
  DollarSign,
  Radio,
  Shield,
  Target,
  Users,
  MessageSquare,
  TestTube,
  Globe,
  Settings,
  TrendingUp,
  Eye,
  Zap,
  CheckSquare,
  Layers
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";

const navigationItems = [
  {
    title: "EXECUTIVE",
    items: [
      { title: "Overview", url: "/", icon: BarChart3 },
      { title: "Executive Dashboard", url: "/executive", icon: TrendingUp },
      { title: "Finance", url: "/finance", icon: DollarSign },
    ]
  },
  {
    title: "OPERATIONS",
    items: [
      { title: "QS/IS", url: "/qs", icon: Shield },
      { title: "Affiliates", url: "/affiliates", icon: Users },
      { title: "Creative Center", url: "/creative-center", icon: Palette },
      { title: "Creative Planner", url: "/creative-planner", icon: FileText },
      { title: "Spend Planner", url: "/spend-planner", icon: Target },
      { title: "Approvals", url: "/approvals", icon: CheckSquare },
      { title: "Auctions Monitor", url: "/auctions", icon: Eye },
    ]
  },
  {
    title: "INTELLIGENCE",
    items: [
      { title: "RL Insights", url: "/rl-insights", icon: Brain },
      { title: "Training Center", url: "/training", icon: Zap },
      { title: "Ops Chat", url: "/chat", icon: MessageSquare },
      { title: "Canvas", url: "/canvas", icon: Layers },
    ]
  },
  {
    title: "GROWTH",
    items: [
      { title: "Channels", url: "/channels", icon: Radio },
      { title: "Experiments", url: "/experiments", icon: TestTube },
      { title: "Landing Pages", url: "/landing-pages", icon: Globe },
      { title: "Backstage", url: "/backstage", icon: Settings },
      { title: "Audiences", url: "/audiences", icon: Users },
    ]
  }
];

export function AppSidebar() {
  const { state } = useSidebar();
  const location = useLocation();
  const currentPath = location.pathname;
  const isCollapsed = state === "collapsed";

  const isActive = (path: string) => {
    if (path === "/" && currentPath === "/") return true;
    if (path !== "/" && currentPath.startsWith(path)) return true;
    return false;
  };

  return (
    <Sidebar className="border-r border-border bg-background w-64">
      <SidebarContent className="bg-background">
        {/* Logo */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
              <Brain className="w-4 h-4 text-accent-foreground" />
            </div>
            <div>
              <h2 className="font-semibold text-foreground">AELP2</h2>
              <p className="text-xs text-muted-foreground">Marketing Intelligence Platform</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        {navigationItems.map((section) => (
          <SidebarGroup key={section.title} className="py-2">
            <SidebarGroupLabel className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-2">
              {section.title}
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu className="space-y-1 px-2">
                {section.items.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink
                        to={item.url}
                        className={({ isActive: navActive }) =>
                          `flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-colors ${
                            isActive(item.url) || navActive
                              ? "bg-accent text-accent-foreground font-medium"
                              : "text-muted-foreground hover:text-foreground hover:bg-muted"
                          }`
                        }
                      >
                        <item.icon className="w-4 h-4 flex-shrink-0" />
                        <span>{item.title}</span>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}

        {/* System Status */}
        <div className="mt-auto p-4 border-t border-border">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 bg-primary rounded-full"></div>
            <span className="text-xs text-muted-foreground">System Health</span>
          </div>
          <p className="text-xs text-muted-foreground">Data refreshed 2m ago</p>
        </div>
      </SidebarContent>
    </Sidebar>
  );
}
