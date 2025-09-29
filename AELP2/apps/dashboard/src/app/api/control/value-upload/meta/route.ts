import { NextResponse } from 'next/server'
import { spawn } from 'node:child_process'

export async function POST() {
  try {
    if ((process.env.GATES_ENABLED || '1') === '1' && (process.env.AELP2_ALLOW_VALUE_UPLOADS || '0') !== '1') {
      return NextResponse.json({ ok: false, error: 'flag_denied: AELP2_ALLOW_VALUE_UPLOADS=0' }, { status: 200 })
    }
    const py = process.env.PYTHON_BIN || 'python3'
    const child = spawn(py, ['-m', 'AELP2.pipelines.upload_meta_capi_conversions'], { env: { ...process.env } })
    let out = ''; let err = ''
    child.stdout.on('data', d => out += d.toString())
    child.stderr.on('data', d => err += d.toString())
    const code: number = await new Promise(resolve => child.on('close', resolve)) as number
    return NextResponse.json({ ok: code === 0, code, out, err })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'

