"use client"
import React, { useState } from 'react'

export default function CreateAd() {
  const [campaignId, setCampaignId] = useState('')
  const [adGroupId, setAdGroupId] = useState('')
  const [finalUrl, setFinalUrl] = useState('https://buy.aura.com')
  const [headlines, setHeadlines] = useState<string>('Aura® Digital Security\nIdentity Theft Protection\n24/7 Customer Support')
  const [descriptions, setDescriptions] = useState<string>('Protect your digital life today.\n#1 rated identity protection.')
  const [msg, setMsg] = useState<string>('')
  const [runId, setRunId] = useState<string>('')
  const [created, setCreated] = useState<{ ad_id?: string, headlines?: string[], descriptions?: string[], final_url?: string } | null>(null)
  const submit = async () => {
    setMsg('')
    const payload = {
      campaign_id: campaignId.trim(),
      ad_group_id: adGroupId.trim(),
      final_url: finalUrl.trim(),
      headlines: headlines.split('\n').map(s=>s.trim()).filter(Boolean),
      descriptions: descriptions.split('\n').map(s=>s.trim()).filter(Boolean),
    }
    if (!payload.final_url || payload.headlines.length===0 || payload.descriptions.length===0) { setMsg('Please fill final URL, at least one headline and one description.'); return }
    const r = await fetch('/api/ads/google/create', { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(payload) })
    const j = await r.json()
    setMsg(j.error ? `Error: ${j.error}` : `Queued for approval (run ${j.run_id})`)
    if (j?.run_id) setRunId(String(j.run_id))
    setCreated({ headlines: payload.headlines, descriptions: payload.descriptions, final_url: payload.final_url })
  }
  return (
    <div className="max-w-3xl mx-auto space-y-4">
      <h1 className="text-xl font-semibold">Create Google RSA</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-gray-600">Campaign ID</label>
          <input className="w-full border rounded px-3 py-2" value={campaignId} onChange={e=>setCampaignId(e.target.value)} placeholder="e.g., 19665933250" />
        </div>
        <div>
          <label className="text-xs text-gray-600">Ad Group ID</label>
          <input className="w-full border rounded px-3 py-2" value={adGroupId} onChange={e=>setAdGroupId(e.target.value)} placeholder="e.g., 144595646583" />
        </div>
        <div className="md:col-span-2">
          <label className="text-xs text-gray-600">Final URL</label>
          <input className="w-full border rounded px-3 py-2" value={finalUrl} onChange={e=>setFinalUrl(e.target.value)} />
        </div>
        <div>
          <label className="text-xs text-gray-600">Headlines (one per line)</label>
          <textarea className="w-full border rounded px-3 py-2 h-32" value={headlines} onChange={e=>setHeadlines(e.target.value)} />
        </div>
        <div>
          <label className="text-xs text-gray-600">Descriptions (one per line)</label>
          <textarea className="w-full border rounded px-3 py-2 h-32" value={descriptions} onChange={e=>setDescriptions(e.target.value)} />
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button onClick={submit} className="px-4 py-2 bg-indigo-600 text-white rounded">Create</button>
        <button onClick={async ()=>{
          const payload = {
            platform:'google_ads', type:'rsa', campaign_id: campaignId.trim()||null, ad_group_id: adGroupId.trim()||null,
            payload: { final_url: finalUrl.trim(), headlines: headlines.split('\n').map(s=>s.trim()).filter(Boolean), descriptions: descriptions.split('\n').map(s=>s.trim()).filter(Boolean), draft:true }, requested_by:'creative_studio'
          }
          const r = await fetch('/api/control/creative/enqueue', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) })
          const j = await r.json(); setMsg(j?.ok?`Draft queued (run ${j.run_id})`:(j?.error||'Failed'))
        }} className="px-3 py-2 border rounded">Save as Draft</button>
      </div>
      {msg && <div className="text-sm mt-3">{msg}</div>}
      {runId && <div className="text-xs mt-1">Approval run: {runId}</div>}
      {created && (
        <div className="mt-4 border rounded p-3">
          <div className="text-sm font-medium mb-2">Preview (Draft RSA)</div>
          <div className="text-xs text-gray-600 mb-1">Final URL: {created.final_url}</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <div className="text-[11px] text-gray-500">Headlines</div>
              <ul className="list-disc pl-4 text-sm">
                {(created.headlines||[]).map((h,i)=> <li key={i}>{h}</li>)}
              </ul>
            </div>
            <div>
              <div className="text-[11px] text-gray-500">Descriptions</div>
              <ul className="list-disc pl-4 text-sm">
                {(created.descriptions||[]).map((d,i)=> <li key={i}>{d}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
