"use client"
import React from 'react'
import { ResponsiveContainer, Sankey, Tooltip } from 'recharts'

export default function JourneysSankey(){
  const [data, setData] = React.useState<{nodes:any[], links:any[]}|null>(null)
  React.useEffect(()=>{
    fetch('/api/bq/journeys/sankey').then(r=>r.json()).then(j=>{
      if (j.nodes && j.links) setData(j)
    })
  },[])
  if (!data) return <div className="text-sm text-slate-600">Loading paths…</div>
  if (data.nodes.length === 0 || data.links.length === 0) return <div className="text-sm text-slate-600">No path data.</div>
  return (
    <div style={{ width: '100%', height: 360 }}>
      <ResponsiveContainer>
        {/* @ts-ignore */}
        <Sankey data={data} nodePadding={24} nodeWidth={14} linkCurvature={0.5} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
          {/* @ts-ignore */}
          <Tooltip />
        </Sankey>
      </ResponsiveContainer>
    </div>
  )
}

