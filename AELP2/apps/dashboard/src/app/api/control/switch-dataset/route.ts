import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  const body = await req.json().catch(()=>({})) as any
  const dataset = String(body?.dataset || '')
  if (!dataset) return NextResponse.json({ ok: false, error: 'dataset required' }, { status: 200 })
  // This endpoint only acknowledges intent; callers must restart app with env.
  return NextResponse.json({ ok: true, message: `Switch to dataset ${dataset} acknowledged (restart required).` })
}

export const dynamic = 'force-dynamic'

