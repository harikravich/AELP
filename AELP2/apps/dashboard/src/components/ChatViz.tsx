"use client"
import React from 'react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar, Legend } from 'recharts'

export type VizSpec = {
  type: 'line' | 'bar'
  title?: string
  xKey: string
  yKey: string
  data: any[]
}

export default function ChatViz({ viz }: { viz: VizSpec }) {
  const height = 240
  return (
    <div className="mt-2 bg-white border border-slate-200 rounded p-2">
      {viz.title && <div className="text-sm font-medium mb-1">{viz.title}</div>}
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          {viz.type === 'line' ? (
            <LineChart data={viz.data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={viz.xKey} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line dataKey={viz.yKey} stroke="#6366f1" dot={false} strokeWidth={2} />
            </LineChart>
          ) : (
            <BarChart data={viz.data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={viz.xKey} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey={viz.yKey} fill="#10b981" />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  )
}

