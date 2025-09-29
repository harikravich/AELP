import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({})) as any
    const ids = String(body?.campaign_ids || process.env.AELP2_GOOGLE_CANARY_CAMPAIGN_IDS || '')
    const direction = String(body?.direction || process.env.AELP2_CANARY_BUDGET_DIRECTION || 'down')
    if (!ids) return NextResponse.json({ ok: false, error: 'campaign_ids required' }, { status: 400 })
    if ((process.env.GATES_ENABLED || '1') === '1' && (process.env.AELP2_ALLOW_GOOGLE_MUTATIONS || '0') !== '1') {
      return NextResponse.json({ ok: false, error: 'flag_denied: AELP2_ALLOW_GOOGLE_MUTATIONS=0' }, { status: 200 })
    }

    const env = { ...process.env,
      AELP2_GOOGLE_CANARY_CAMPAIGN_IDS: ids,
      AELP2_CANARY_BUDGET_DIRECTION: direction,
      // Keep shadow unless gates allow otherwise; script enforces
      AELP2_SHADOW_MODE: process.env.AELP2_SHADOW_MODE || '1',
    }
    const py = process.env.PYTHON_BIN || 'python3'
    const mod = 'AELP2/scripts/apply_google_canary.py'
    const child = spawn(py, [mod], { env })
    let out = ''
    let err = ''
    child.stdout.on('data', (d) => out += d.toString())
    child.stderr.on('data', (d) => err += d.toString())
    const code: number = await new Promise((resolve) => child.on('close', resolve)) as number
    return NextResponse.json({ ok: code === 0, code, out, err })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'
