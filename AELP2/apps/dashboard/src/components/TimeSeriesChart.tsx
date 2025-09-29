"use client"
import React from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'

export type Series = {
  name: string
  dataKey: string
  color?: string
  yAxisId?: string
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 text-xs">
        <p className="text-white/90 font-medium mb-1">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-white/70">
            <span style={{ color: entry.color }}>{entry.name}:</span> {entry.value?.toFixed(2)}
          </p>
        ))}
      </div>
    )
  }
  return null
}

export function TimeSeriesChart({ data, series, height = 280 }: { data: any[]; series: Series[]; height?: number }) {
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 11, fill: 'rgba(255,255,255,0.6)' }} 
            stroke="rgba(255,255,255,0.2)"
          />
          <YAxis 
            yAxisId="left" 
            tick={{ fontSize: 11, fill: 'rgba(255,255,255,0.6)' }} 
            stroke="rgba(255,255,255,0.2)"
          />
          <YAxis 
            yAxisId="right" 
            orientation="right" 
            tick={{ fontSize: 11, fill: 'rgba(255,255,255,0.6)' }} 
            stroke="rgba(255,255,255,0.2)"
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ color: 'rgba(255,255,255,0.8)' }}
            iconType="line"
          />
          {series.map((s) => (
            <Line
              key={s.name}
              yAxisId={s.yAxisId || 'left'}
              type="monotone"
              dataKey={s.dataKey}
              name={s.name}
              stroke={s.color || '#667eea'}
              dot={false}
              strokeWidth={2.5}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

