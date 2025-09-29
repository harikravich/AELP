import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

const MAP: Record<string, string> = {
  meta: 'AELP2.pipelines.meta_to_bq',
  linkedin: 'AELP2.pipelines.linkedin_to_bq',
  tiktok: 'AELP2.pipelines.tiktok_to_bq',
}

export async function POST(req: Request) {
  const { searchParams } = new URL(req.url)
  const platform = String(searchParams.get('platform') || '')
  const mod = MAP[platform]
  if (!mod) return NextResponse.json({ ok: false, error: 'unknown_platform' }, { status: 200 })
  try {
    const env = { ...process.env, DRY_RUN: '1' }
    const py = process.env.PYTHON_BIN || 'python3'
    const child = spawn(py, ['-m', mod], { env })
    let out = ''; let err = ''
    child.stdout.on('data', (d) => out += d.toString())
    child.stderr.on('data', (d) => err += d.toString())
    const code: number = await new Promise((resolve)=> child.on('close', resolve)) as number
    return NextResponse.json({ ok: code === 0, out, err })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

export const dynamic = 'force-dynamic'

