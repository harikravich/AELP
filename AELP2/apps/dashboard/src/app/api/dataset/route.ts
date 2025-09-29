import { cookies } from 'next/headers'
import { NextResponse } from 'next/server'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const c = cookies()
  const mode = c.get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  return NextResponse.json({ mode, dataset })
}

export async function POST(req: Request) {
  const url = new URL(req.url)
  const mode = url.searchParams.get('mode') === 'prod' ? 'prod' : 'sandbox'
  const res = NextResponse.json({ ok: true, mode, dataset: mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET })
  res.cookies.set({ name: DATASET_COOKIE, value: mode, httpOnly: false, sameSite: 'lax', path: '/' })
  return res
}
