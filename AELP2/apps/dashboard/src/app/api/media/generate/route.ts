import { NextRequest, NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(()=>({})) as any
    const prompt = String(body?.prompt || '')
    if (!prompt) return NextResponse.json({ ok:false, error:'prompt required' }, { status: 400 })
    const key = process.env.OPENAI_API_KEY as string
    if (!key) {
      // Fallback: return simple SVG placeholder as data URL
      const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='1024' height='1024'>
        <rect width='100%' height='100%' fill='#0ea5e9'/>
        <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-size='48' fill='white' font-family='sans-serif'>${(prompt||'Creative').slice(0,40)}</text>
      </svg>`
      const url = `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`
      return NextResponse.json({ ok:true, images:[url], fallback:true })
    }
    const sizeReq = String(body?.size || '1024x1024')
    const allowed = new Set(['1024x1024','1024x1536','1536x1024','auto'])
    const size = allowed.has(sizeReq) ? sizeReq : '1024x1024'
    const r = await fetch('https://api.openai.com/v1/images/generations', {
      method:'POST',
      headers:{ 'content-type':'application/json', 'authorization':`Bearer ${key}` },
      body: JSON.stringify({ model: 'gpt-image-1', prompt, size, n: 2 })
    })
    const j = await r.json().catch(()=>({}))
    if (!r.ok) {
      const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='1024' height='1024'>
        <rect width='100%' height='100%' fill='#f97316'/>
        <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-size='42' fill='white' font-family='sans-serif'>Preview Unavailable</text>
      </svg>`
      const url = `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`
      return NextResponse.json({ ok:true, images:[url], fallback:true, error: j?.error?.message || `image gen failed (${r.status})` })
    }
    return NextResponse.json({ ok:true, images: (j.data||[]).map((d:any)=> d.url || d.b64_json) })
  } catch (e:any) { return NextResponse.json({ ok:false, error: e?.message||String(e) }, { status: 200 }) }
}
