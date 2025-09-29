"use client"
import React from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { fmtWhen } from '../lib/utils'

export default function ValueUploads(){
  const [channels, setChannels] = React.useState<string[]>([])
  const [form, setForm] = React.useState({ channel:'', start:'', end:'', multiplier:'1.00', notes:'' })
  const [rows, setRows] = React.useState<any[]>([])
  const [msg, setMsg] = React.useState<string>('')
  const allowMut = (process.env.NEXT_PUBLIC_AELP2_ALLOW_GOOGLE_MUTATIONS || '') === '1'

  const load = ()=> fetch('/api/bq/value-uploads').then(r=>r.json()).then(j=> setRows(j.items||[]))
  React.useEffect(()=>{ fetch('/api/bq/mmm/channels').then(r=>r.json()).then(j=> setChannels(j.channels||[])); load() },[])

  const submit = async (e:any)=>{
    e.preventDefault()
    setMsg('Submitting…')
    const res = await fetch('/api/bq/value-uploads', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({
      channel: form.channel, date_start: form.start, date_end: form.end, multiplier: Number(form.multiplier), notes: form.notes,
    }) })
    const j = await res.json()
    if (!res.ok) { const e = j.error||res.statusText; setMsg(`Error: ${e}`); try{(await import('sonner')).toast.error(e)}catch{}; return }
    setMsg('Submitted'); try{(await import('sonner')).toast.success('Value upload request created')}catch{}
    setForm({ ...form, notes:'' })
    load()
  }

  return (
    <div>
      {!allowMut && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2 mb-3">Mutations are disabled (AELP2_ALLOW_GOOGLE_MUTATIONS=0). Submissions are audited only.</div>
      )}
      <form onSubmit={submit} className="grid grid-cols-1 md:grid-cols-6 gap-2 items-end">
        <label className="text-sm">Channel
          <select className="w-full border border-slate-200 rounded-md px-3 py-2 text-sm bg-white" value={form.channel} onChange={e=>setForm({...form, channel:e.target.value})} required>
            <option value="" disabled>Select</option>
            {channels.map(ch=> <option key={ch} value={ch}>{ch}</option>)}
          </select>
        </label>
        <label className="text-sm">Start
          <Input type="date" value={form.start} onChange={e=>setForm({...form, start:e.target.value})} required />
        </label>
        <label className="text-sm">End
          <Input type="date" value={form.end} onChange={e=>setForm({...form, end:e.target.value})} required />
        </label>
        <label className="text-sm">Multiplier
          <Input type="number" step="0.01" value={form.multiplier} onChange={e=>setForm({...form, multiplier:e.target.value})} required />
        </label>
        <label className="text-sm md:col-span-2">Notes
          <Input type="text" placeholder="reason, ticket, etc" value={form.notes} onChange={e=>setForm({...form, notes:e.target.value})} />
        </label>
        <Button type="submit">Create Request</Button>
      </form>
      {msg && <div className="text-xs text-slate-600 mt-2">{msg}</div>}
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full text-sm table-clean">
          <thead><tr><th className="py-2 pr-4">When</th><th className="py-2 pr-4">Channel</th><th className="py-2 pr-4">Range</th><th className="py-2 pr-4">Multiplier</th><th className="py-2 pr-4">Status</th><th className="py-2 pr-4">Notes</th></tr></thead>
          <tbody>
            {rows.map((r:any,i:number)=> (
              <tr key={i}>
                <td className="py-2 pr-4">{fmtWhen(r.created_at||r.timestamp)}</td>
                <td className="py-2 pr-4">{r.channel}</td>
                <td className="py-2 pr-4">{r.date_start} → {r.date_end}</td>
                <td className="py-2 pr-4">{Number(r.multiplier||1).toFixed(2)}</td>
                <td className="py-2 pr-4">{r.status||'pending'}</td>
                <td className="py-2 pr-4">{r.notes||''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
