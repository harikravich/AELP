"use client"
import React, { useEffect, useState } from 'react'

export default function CreativePreview() {
  const [data, setData] = useState<any>(null)
  const [error, setError] = useState<string>('')
  useEffect(()=>{
    const u = new URL(window.location.href)
    const ad_id = u.searchParams.get('ad_id')
    const campaign_id = u.searchParams.get('campaign_id')
    if (!ad_id) { setError('Missing ad_id'); return }
    fetch(`/api/ads/creative?ad_id=${encodeURIComponent(ad_id)}${campaign_id?`&campaign_id=${encodeURIComponent(campaign_id)}`:''}`).then(r=>r.json()).then(j=>{
      if (j.error) setError(j.error); else setData(j)
    }).catch(e=> setError(String(e)))
  },[])
  if (error) return <div className="p-4 text-sm text-red-600">{error}</div>
  if (!data) return <div className="p-4 text-sm text-gray-600">Loading…</div>
  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <h1 className="text-xl font-semibold">Ad Preview</h1>
      <div className="bg-white rounded shadow-sm p-4">
        <div className="text-xs text-gray-500">Ad ID</div>
        <div className="text-sm">{data.ad_id}</div>
      </div>
      <div className="bg-white rounded shadow-sm p-4">
        <div className="text-xs text-gray-500">Final URL</div>
        <a className="text-sm text-indigo-600 hover:underline" href={data.final_urls?.[0]} target="_blank" rel="noreferrer">{data.final_urls?.[0] || '—'}</a>
      </div>
      <div className="bg-white rounded shadow-sm p-4">
        <div className="text-xs text-gray-500">Headlines</div>
        <ul className="list-disc pl-5 text-sm">
          {(data.headlines||[]).map((h:string,i:number)=> (<li key={i}>{h}</li>))}
        </ul>
      </div>
      <div className="bg-white rounded shadow-sm p-4">
        <div className="text-xs text-gray-500">Descriptions</div>
        <ul className="list-disc pl-5 text-sm">
          {(data.descriptions||[]).map((d:string,i:number)=> (<li key={i}>{d}</li>))}
        </ul>
      </div>
    </div>
  )
}

