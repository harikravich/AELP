import { NextResponse } from 'next/server'
import { spawn } from 'child_process'

export async function POST() {
  const sh = 'bash'
  const args = ['AELP2/scripts/run_ads_ingestion.sh']
  const env = { ...process.env, DRY_RUN: process.env.DRY_RUN || '1' }
  const child = spawn(sh, args, { env })
  let out = ''; let err = ''
  child.stdout.on('data', (d)=> out += d.toString())
  child.stderr.on('data', (d)=> err += d.toString())
  const code: number = await new Promise((resolve)=> child.on('close', resolve)) as number
  return NextResponse.json({ ok: code === 0, code, out, err })
}

export const dynamic = 'force-dynamic'

