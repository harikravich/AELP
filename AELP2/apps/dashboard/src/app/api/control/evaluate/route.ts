import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(()=>({})) as any
    const changePct = Number(body?.change_pct||0)
    const todayPct = Number(body?.today_pct||0)
    const perChangeCap = Number(process.env.AELP2_PER_CHANGE_CAP || '5')
    const dailyCap = Number(process.env.AELP2_DAILY_CAP || '10')
    const ok = (changePct <= perChangeCap) && ((todayPct + changePct) <= dailyCap)
    const reason = ok ? '' : `Cap breach: per-change <= ${perChangeCap}%, daily <= ${dailyCap}%`
    return NextResponse.json({ ok, reason, limits: { perChangeCap, dailyCap } })
  } catch (e:any) { return NextResponse.json({ ok:false, error: e?.message||String(e) }, { status: 200 }) }
}

export const dynamic = 'force-dynamic'
