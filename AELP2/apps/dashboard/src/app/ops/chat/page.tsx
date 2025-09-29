"use client"
import React, { useState } from 'react'
import { Button } from '../../../components/ui/button'
import { Input } from '../../../components/ui/input'
import ChatViz, { VizSpec } from '../../../components/ChatViz'

type Msg = { role: 'user'|'assistant', content: string, viz?: VizSpec }

export default function ChatOps() {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const send = async () => {
    if (!input.trim()) return
    const m: Msg = { role: 'user', content: input }
    setMsgs(prev => [...prev, m])
    setInput('')
    setBusy(true)
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify({ messages: [...msgs, m] }) })
      const j = await r.json()
      const reply = j.reply || j.error || 'Error'
      const viz = j.viz as VizSpec | undefined
      setMsgs(prev => [...prev, { role: 'assistant', content: reply, viz }])
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="max-w-3xl mx-auto space-y-4">
      <h1 className="text-xl font-semibold">ChatOps</h1>
      <div className="border rounded h-[26rem] overflow-y-auto bg-white p-3 text-slate-900">
        {msgs.length === 0 && <div className="text-sm text-gray-500">Ask about CAC, budgets, GA4 trends, MMM targets, or type /help.</div>}
        {msgs.map((m,i)=> (
          <div key={i} className={`text-sm my-2 ${m.role==='assistant' ? 'bg-slate-50 border border-slate-200' : 'bg-indigo-50 border border-indigo-200'} p-2 rounded text-slate-900` }>
            <div><strong className="mr-1">{m.role==='user'?'You':'Assistant'}:</strong> {m.content}</div>
            {m.viz && (
              <div className="mt-2">
                <ChatViz viz={m.viz} />
                <form method="post" action="/api/canvas/pin" className="mt-2">
                  <input type="hidden" name="title" value={m.viz.title || 'Chart'} />
                  <input type="hidden" name="viz" value={JSON.stringify(m.viz)} />
                  <button className="px-2 py-1 text-xs rounded bg-indigo-600 text-white">Pin to Canvas</button>
                </form>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <Input className="flex-1" value={input} onChange={e=>setInput(e.target.value)} placeholder="Ask a question..." />
        <Button onClick={send} disabled={busy}>Send</Button>
      </div>
      <div className="text-xs text-slate-500">Backend: {process.env.OPENAI_BASE_URL ? 'OpenAI‑compatible ('+process.env.OPENAI_BASE_URL+')' : 'OpenAI API'} · Commands: /help, /mmm, /creative, /run, /sql</div>
    </div>
  )
}
