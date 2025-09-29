import React from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricTileProps {
  label: string
  value: string
  hint?: string
  trend?: 'up' | 'down' | 'neutral'
  change?: string
  tone?: 'indigo' | 'emerald' | 'rose' | 'sky'
  icon?: React.ReactNode
  className?: string
}

export default function MetricTile({ 
  label, 
  value, 
  hint, 
  trend,
  change,
  tone = 'indigo',
  icon,
  className,
}: MetricTileProps) {
  const gradients: Record<string, string> = {
    indigo: 'from-indigo-500 to-purple-600',
    emerald: 'from-emerald-500 to-teal-600',
    rose: 'from-rose-500 to-pink-600',
    sky: 'from-sky-500 to-cyan-600',
  }

  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus

  return (
    <div className={`glass-card group hover:scale-[1.02] transition-all duration-300 ${className || ''}`}>
      <div className="relative">
        {/* Gradient accent line */}
        <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${gradients[tone]} rounded-t-xl opacity-80`} />
        
        <div className="pt-2">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="text-xs font-medium text-white/60 uppercase tracking-wider mb-2 flex items-center gap-2">
                {icon && <span className="text-white/70">{icon}</span>}
                <span>{label}</span>
              </p>
              <p className={`text-3xl font-bold bg-gradient-to-r ${gradients[tone]} bg-clip-text text-transparent`}>
                {value}
              </p>
              {hint && (
                <p className="text-xs text-white/40 mt-2">
                  {hint}
                </p>
              )}
            </div>
            
            {trend && (
              <div className={`
                flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium
                ${trend === 'up' 
                  ? 'bg-emerald-500/10 text-emerald-400' 
                  : trend === 'down'
                  ? 'bg-rose-500/10 text-rose-400'
                  : 'bg-white/5 text-white/50'
                }
              `}>
                <TrendIcon className="w-3 h-3" />
                {change && <span>{change}</span>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
