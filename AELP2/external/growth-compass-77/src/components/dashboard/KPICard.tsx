import { ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown } from "lucide-react";

interface KPICardProps {
  title: string;
  value: string;
  change: number;
  changeLabel: string;
  icon: ReactNode;
  variant?: "default" | "success" | "warning" | "destructive";
  children?: ReactNode;
}

export function KPICard({ 
  title, 
  value, 
  change, 
  changeLabel, 
  icon, 
  variant = "default",
  children 
}: KPICardProps) {
  const isPositive = change > 0;
  
  const variantStyles = {
    default: "bg-card border",
    success: "bg-primary/10 border-primary/20",
    warning: "bg-warning/10 border-warning/20", 
    destructive: "bg-destructive/10 border-destructive/20"
  };

  return (
    <Card className={`${variantStyles[variant]} shadow-card hover:shadow-executive transition-all duration-300 group`}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${
              variant === "success" ? "bg-secondary/20" :
              variant === "warning" ? "bg-warning/20" :
              variant === "destructive" ? "bg-destructive/20" :
              "bg-primary/20"
            }`}>
              {icon}
            </div>
            <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          </div>
          <Badge 
            variant={isPositive ? "default" : "destructive"} 
            className="flex items-center gap-1"
          >
            {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {Math.abs(change)}%
          </Badge>
        </div>
        
        <div className="space-y-2">
          <p className="text-3xl font-bold text-foreground group-hover:text-primary transition-colors">
            {value}
          </p>
          <p className="text-xs text-muted-foreground">{changeLabel}</p>
        </div>

        {children && (
          <div className="mt-4 pt-4 border-t border-border/50">
            {children}
          </div>
        )}
      </div>
    </Card>
  );
}