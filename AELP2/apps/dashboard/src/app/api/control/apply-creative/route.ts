import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(()=> ({})) as any
    const change = JSON.stringify(body?.change || {})
    if ((process.env.GATES_ENABLED || '1') === '1' && (process.env.AELP2_ALLOW_BANDIT_MUTATIONS || '0') !== '1') {
      return NextResponse.json({ ok: false, error: 'flag_denied: AELP2_ALLOW_BANDIT_MUTATIONS=0' }, { status: 200 })
    }
    const py = process.env.PYTHON_BIN || 'python3'
    const env = { ...process.env, AELP2_CREATIVE_CHANGE_JSON: change, AELP2_SHADOW_MODE: process.env.AELP2_SHADOW_MODE || '1' }
    const child = spawn(py, ['AELP2/scripts/apply_google_creatives.py'], { env })
    let out = ''; let err = ''
    child.stdout.on('data', (d)=> out += d.toString())
    child.stderr.on('data', (d)=> err += d.toString())
    const code: number = await new Promise((resolve)=> child.on('close', resolve)) as number
    return NextResponse.json({ ok: code === 0, code, out, err })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'

