import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

export async function POST() {
  try {
    const env = { ...process.env }
    const py = process.env.PYTHON_BIN || 'python3'
    const child = spawn(py, ['-m', 'AELP2.pipelines.youtube_reach_planner'], { env })
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

