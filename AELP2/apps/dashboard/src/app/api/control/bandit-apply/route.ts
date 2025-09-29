import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({})) as any
    const lookback = String(body?.lookback || '30')
    if ((process.env.GATES_ENABLED || '1') === '1' && (process.env.AELP2_ALLOW_BANDIT_MUTATIONS || '0') !== '1') {
      return NextResponse.json({ ok: false, error: 'flag_denied: AELP2_ALLOW_BANDIT_MUTATIONS=0' }, { status: 200 })
    }
    const env = { ...process.env }
    try {
      const cwd = process.cwd()
      const idx = cwd.indexOf('/AELP2/')
      const projectRoot = idx >= 0 ? cwd.slice(0, idx) : cwd
      env.PYTHONPATH = env.PYTHONPATH ? `${env.PYTHONPATH}:${projectRoot}` : projectRoot
    } catch {}
    const py = process.env.PYTHON_BIN || 'python3'
    // Use the orchestrator to log proposals (shadow)
    const cwd = process.cwd()
    const idx = cwd.indexOf('/AELP2/')
    const projectRoot = idx >= 0 ? cwd.slice(0, idx) : cwd
    const script = `${projectRoot}/AELP2/core/optimization/bandit_orchestrator.py`
    const child = spawn(py, [script, '--lookback', lookback], { env, cwd: projectRoot })
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
