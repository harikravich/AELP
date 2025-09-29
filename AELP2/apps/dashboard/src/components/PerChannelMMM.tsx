"use client"
import React from 'react'
import MMMSlider from './MMMSlider'

export default function PerChannelMMM(){
  const [channels, setChannels] = React.useState<string[]>([])
  React.useEffect(()=>{
    fetch('/api/bq/mmm/channels').then(r=>r.json()).then(j=> setChannels(j.channels||[]))
  },[])
  if (channels.length === 0) return <div className="text-sm text-slate-600">No channels found.</div>
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {channels.map(ch=> (
        <div key={ch} className="p-2 border border-slate-200 rounded">
          <div className="text-sm font-medium mb-2">{ch}</div>
          <MMMSlider channel={ch} />
        </div>
      ))}
    </div>
  )
}

