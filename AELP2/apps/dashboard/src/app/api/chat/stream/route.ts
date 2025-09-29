import { NextRequest } from 'next/server'

export const dynamic = 'force-dynamic'

export async function POST(req: NextRequest) {
  const { absoluteUrl } = await import('../../../lib/url')
  try {
    const body = await req.json().catch(()=>({}))
    const messages = Array.isArray(body?.messages) ? body.messages : []
    // call existing /api/chat for a full reply, then stream it in chunks
    const r = await fetch(absoluteUrl('/api/chat'), { method:'POST', headers: { 'content-type':'application/json' }, body: JSON.stringify({ messages }) })
    const j = await r.json().catch(()=>({}))
    const reply = String(j?.reply || '')
    const encoder = new TextEncoder()
    return new Response(new ReadableStream({
      async start(controller) {
        const chunkSize = 64
        for (let i=0; i<reply.length; i+=chunkSize) {
          controller.enqueue(encoder.encode(reply.slice(i, i+chunkSize)))
          await new Promise(res=> setTimeout(res, 20))
        }
        controller.close()
      }
    }), { headers: { 'content-type':'text/plain; charset=utf-8' } })
  } catch (e:any) {
    return new Response(`error: ${e?.message || String(e)}`, { status: 500 })
  }
}
