import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { MessageSquare, Send, BarChart3, TrendingUp, Pin, Bot, User } from "lucide-react";
import { useState, useEffect } from "react";
import { toast } from "sonner";

const chatHistory = [
  {
    id: 1,
    type: "assistant",
    message: "Hi! Ask me about CAC, ROAS, channels, MMM what‑ifs (/mmm <channel> <budget>), or type /help.",
    timestamp: "now"
  }
];

export default function OpsChat() {
  const [message, setMessage] = useState("");
  const [kpiMeta, setKpiMeta] = useState<any>(null)
  useEffect(()=>{ fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/bq/kpi/summary`).then(r=>r.json()).then(setKpiMeta).catch(()=> setKpiMeta(null)) },[])
  const [gates, setGates] = useState<boolean|undefined>(undefined)
  useEffect(()=>{ fetch('/api/connections/health').then(r=>r.json()).then(j=> setGates(!!j?.checks?.gatesEnabled)).catch(()=> setGates(undefined)) },[])
  const [log, setLog] = useState<any[]>(chatHistory)
  const send = async () => {
    if (!message.trim()) return
    const next = [...log, { id: Date.now(), type:'user', message, timestamp:'now' }]
    setLog(next)
    setMessage('')
    try {
      const base = (import.meta as any).env.VITE_API_BASE_URL || ''
      const endpoint = gates ? '/api/chat' : '/api/chat/stream'
      const url = `${base}${endpoint}`
      if (endpoint.endsWith('/stream')) {
        const res = await fetch(url, { method:'POST', credentials:'include', headers:{'content-type':'application/json'}, body: JSON.stringify({ messages: [{ role:'user', content: message }] }) })
        const reader = res.body?.getReader(); const decoder = new TextDecoder(); let acc = ''
        if (reader) {
          while(true){ const { value, done } = await reader.read(); if(done) break; acc += decoder.decode(value); }
          setLog(prev=> [...prev, { id: Date.now()+2, type:'assistant', message: acc, timestamp:'now' }])
        } else {
          const j = await res.json().catch(()=>({})); setLog(prev=> [...prev, { id: Date.now()+2, type:'assistant', message: String(j?.reply||''), timestamp:'now' }])
        }
      } else {
        const res = await fetch(url, { method:'POST', credentials:'include', headers:{'content-type':'application/json'}, body: JSON.stringify({ messages: [{ role:'user', content: message }] }) })
        const j = await res.json().catch(()=>({}))
        const reply = (j?.reply || j?.text || JSON.stringify(j))
        setLog(prev=> [...prev, { id: Date.now()+1, type:'assistant', message: reply, timestamp:'now' }])
      }
    } catch(e:any) { toast.error(String(e?.message||e)) }
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Marketing Intelligence Chat</h1>
            <p className="text-muted-foreground mt-1">
              Ask questions, get instant analytics, pin charts to your dashboard
            </p>
          </div>
          <Badge className="bg-primary/20 text-primary border-primary/30">
            <Bot className="w-3 h-3 mr-1" />
            AI Assistant Online
          </Badge>
          <Badge variant={gates ? 'default' : 'secondary'}>{gates===undefined?'Flags…':(gates?'Gates ON':'Gates OFF')}</Badge>
        </div>

        <Card className="border shadow-lg">
          <div className="flex flex-col h-96">
            {/* Chat Header */}
            <div className="p-4 border-b bg-muted/20">
              <div className="flex items-center gap-3">
                <Avatar className="w-8 h-8 bg-accent">
                  <AvatarFallback>AI</AvatarFallback>
                </Avatar>
                <div>
                  <h3 className="font-medium">Marketing Intelligence Assistant</h3>
                  <p className="text-xs text-muted-foreground">Source: {kpiMeta?.source || '—'} • Dataset: {kpiMeta?.dataset || '—'}</p>
                </div>
              </div>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 p-4 space-y-4 overflow-y-auto">
              {log.map((chat) => (
                <div key={chat.id} className={`flex ${chat.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`flex gap-3 max-w-[80%] ${chat.type === 'user' ? 'flex-row-reverse' : ''}`}>
                    <Avatar className="w-6 h-6 flex-shrink-0">
                      <AvatarFallback className={chat.type === 'user' ? 'bg-accent text-accent-foreground' : 'bg-primary text-primary-foreground'}>
                        {chat.type === 'user' ? <User className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
                      </AvatarFallback>
                    </Avatar>
                    <div className={`rounded-lg p-3 ${
                      chat.type === 'user' 
                        ? 'bg-accent text-accent-foreground' 
                        : 'bg-muted border'
                    }`}>
                      <p className="text-sm">{chat.message}</p>
                      
                      {/* Chart Visualization */}
                      {chat.hasChart && (
                        <div className="mt-3 p-3 bg-background/50 rounded border border-dashed">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium">CAC Trend (14 days)</span>
                            <Button size="sm" variant="ghost" className="h-6 px-2" onClick={async ()=>{
                              try{
                                await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || ''}/api/canvas/pin`, { method:'POST', credentials:'include', headers:{'content-type':'application/json'}, body: JSON.stringify({ title:'CAC Trend', type:'chart', payload:{ metric:'cac_14d' } }) })
                                toast.success('Pinned to Canvas')
                              }catch(e:any){toast.error(String(e?.message||e))}
                            }}>
                              <Pin className="w-3 h-3" />
                            </Button>
                          </div>
                          <div className="h-24 bg-primary/10 rounded flex items-center justify-center">
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <BarChart3 className="w-4 h-4" />
                              Interactive chart showing CAC $340 → $297
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Action Buttons */}
                      {chat.hasActions && (
                        <div className="mt-3 flex gap-2">
                          <Button size="sm" variant="outline" className="text-xs">
                            Show Breakdown
                          </Button>
                          <Button size="sm" variant="outline" className="text-xs">
                            Suggest Reallocation
                          </Button>
                          <Button size="sm" variant="ghost" className="text-xs">
                            <Pin className="w-3 h-3 mr-1" />
                            Pin
                          </Button>
                        </div>
                      )}
                      
                      <p className="text-xs text-muted-foreground mt-2">{chat.timestamp}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Input Area */}
            <div className="p-4 border-t">
              <div className="flex gap-2">
                <Input 
                  placeholder="Ask about CAC, ROAS, campaigns, or request specific charts..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  className="flex-1"
                />
                <Button size="icon" className="bg-accent" onClick={send}>
                  <Send className="w-4 h-4" />
                </Button>
              </div>
              <div className="flex gap-2 mt-2">
                <Button variant="outline" size="sm" className="text-xs">
                  Show ROAS by channel
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  Campaign performance this week
                </Button>
                <Button variant="outline" size="sm" className="text-xs">
                  Top converting keywords
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* Quick Analytics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="border p-4 shadow-lg">
            <div className="flex items-center gap-3 mb-2">
              <TrendingUp className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">Recent Insights</span>
            </div>
            <p className="text-xs text-muted-foreground">CAC trending down 6.9% • ROAS stable at 0.45x</p>
          </Card>

          <Card className="border p-4 shadow-lg">
            <div className="flex items-center gap-3 mb-2">
              <BarChart3 className="w-4 h-4 text-accent" />
              <span className="text-sm font-medium">Live Data</span>
            </div>
            <p className="text-xs text-muted-foreground">Connected to Google Ads + GA4 • Updated 30s ago</p>
          </Card>

          <Card className="border p-4 shadow-lg">
            <div className="flex items-center gap-3 mb-2">
              <MessageSquare className="w-4 h-4 text-warning" />
              <span className="text-sm font-medium">Suggestions</span>
            </div>
            <p className="text-xs text-muted-foreground">Try: "Show conversion funnel" or "Compare last 2 weeks"</p>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}
