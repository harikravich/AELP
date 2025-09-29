import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

export async function POST(req: Request) {
  const body = await req.json().catch(()=>({})) as any
  const reason = String(body?.reason || 'manual_stop')
  const note = String(body?.note || 'dashboard')
  const py = process.env.PYTHON_BIN || 'python3'
  const args = ['AELP2/scripts/emergency_stop.py', '--reason', reason, '--note', note]
  const child = spawn(py, args, { env: { ...process.env }})
  let out = ''; let err = ''
  child.stdout.on('data', (d)=> out += d.toString())
  child.stderr.on('data', (d)=> err += d.toString())
  const code: number = await new Promise((resolve)=> child.on('close', resolve)) as number
  return NextResponse.json({ ok: code === 0, code, out, err })
}

export const dynamic = 'force-dynamic'

