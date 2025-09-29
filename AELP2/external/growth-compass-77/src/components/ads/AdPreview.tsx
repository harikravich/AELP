import React from 'react'

function clean(text:string){
  try{
    // Replace {KeyWord:...} style with fallback text after ':'
    text = text.replace(/\{([^:{}]+):([^}]+)\}/g, (_,key,fb) => fb)
    // Replace {LOCATION(...):fallback} -> fallback
    text = text.replace(/\{LOCATION\([^)]*\):([^}]+)\}/g, (_,fb) => fb)
    // Replace {CUSTOMIZER.[^:]+:fallback} -> fallback
    text = text.replace(/\{CUSTOMIZER\.[^:}]+:([^}]+)\}/g, (_,fb) => fb)
    // Strip any remaining {TOKEN}
    text = text.replace(/\{[^}]+\}/g,'')
    // Collapse whitespace
    text = text.replace(/\s+/g,' ').trim()
    return text
  }catch{ return text }
}

export function AdPreview({ data }:{ data:any }){
  const headlines: string[] = (data?.headlines||[]).slice(0,3)
  const descriptions: string[] = (data?.descriptions||[]).slice(0,2)
  const url: string = (data?.final_urls||[])[0] || ''
  const path1 = data?.path1 || ''
  const path2 = data?.path2 || ''
  const host = (()=>{ try { return url ? new URL(url).host : 'example.com' } catch { return 'example.com' } })()
  const displayPath = [path1,path2].filter(Boolean).join('/')
  return (
    <div className="border rounded-lg p-4 bg-card text-card-foreground space-y-2">
      <div className="text-xs text-muted-foreground">Ad • {host}{displayPath?`/${displayPath}`:''}</div>
      <div className="space-y-1">
        {headlines.map((h,i)=> (
          <div key={i} className="text-lg font-semibold text-primary leading-tight">{clean(h)}</div>
        ))}
      </div>
      <div className="space-y-1">
        {descriptions.map((d,i)=> (
          <div key={i} className="text-sm text-muted-foreground leading-snug">{clean(d)}</div>
        ))}
      </div>
    </div>
  )
}
