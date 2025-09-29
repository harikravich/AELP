import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { KPI_COOKIE, KpiSource } from '../../lib/kpi'

export const dynamic = 'force-dynamic'

export async function GET() {
  const c = cookies().get(KPI_COOKIE)?.value || 'ga4_google'
  return NextResponse.json({ source: c })
}

export async function POST(req: Request) {
  const url = new URL(req.url)
  const v = (url.searchParams.get('source') || '').trim() as KpiSource
  const valid: KpiSource[] = ['ads','ga4_all','ga4_google','pacer']
  if (!valid.includes(v)) return NextResponse.json({ ok:false, error:'invalid source' }, { status: 400 })
  const res = NextResponse.json({ ok:true, source: v })
  res.cookies.set(KPI_COOKIE, v, { path: '/', httpOnly: false })
  return res
}
