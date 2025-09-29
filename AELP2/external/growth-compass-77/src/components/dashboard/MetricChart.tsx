import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MoreHorizontal, TrendingUp, Eye } from "lucide-react";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip, Area, AreaChart } from "recharts";

interface MetricChartProps {
  title: string;
  subtitle?: string;
  data?: Array<{ name: string; value: number; target?: number }>;
  type?: "line" | "area";
  color?: string;
  showTarget?: boolean;
  actions?: React.ReactNode;
}

const sampleData = [
  { name: "Sep 1", value: 340, target: 300 },
  { name: "Sep 3", value: 324, target: 300 },
  { name: "Sep 5", value: 289, target: 300 },
  { name: "Sep 7", value: 312, target: 300 },
  { name: "Sep 9", value: 298, target: 300 },
  { name: "Sep 11", value: 285, target: 300 },
  { name: "Sep 12", value: 297, target: 300 },
];

export function MetricChart({ 
  title, 
  subtitle, 
  data = sampleData, 
  type = "area", 
  color = "#3b82f6", 
  showTarget = false,
  actions
}: MetricChartProps) {
  return (
    <Card className="border shadow-lg hover:shadow-xl transition-all duration-300 group bg-card/50 backdrop-blur-sm">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-foreground group-hover:text-primary transition-colors">
              {title}
            </h3>
            {subtitle && (
              <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>
            )}
          </div>
          <div className="flex items-center gap-2">
            {actions}
            <Button variant="ghost" size="icon" className="opacity-0 group-hover:opacity-100 transition-opacity">
              <Eye className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" className="opacity-0 group-hover:opacity-100 transition-opacity">
              <MoreHorizontal className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            {type === "area" ? (
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={color} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="name" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--card))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                    color: 'hsl(var(--foreground))'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorValue)"
                />
                {showTarget && (
                  <Area
                    type="monotone"
                    dataKey="target"
                    stroke="hsl(var(--destructive))"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    fillOpacity={0}
                    fill="transparent"
                  />
                )}
              </AreaChart>
            ) : (
              <LineChart data={data}>
                <XAxis 
                  dataKey="name" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--card))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                    color: 'hsl(var(--foreground))'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  strokeWidth={3}
                  dot={{ fill: color, strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: color, strokeWidth: 2 }}
                />
                {showTarget && (
                  <Line
                    type="monotone"
                    dataKey="target"
                    stroke="hsl(var(--destructive))"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                )}
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-between mt-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground">
              Trending up 12% vs last period
            </span>
          </div>
          <Badge variant="outline">Live</Badge>
        </div>
      </div>
    </Card>
  );
}