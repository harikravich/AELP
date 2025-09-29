import { NextResponse } from 'next/server'
export const dynamic = 'force-dynamic'
import { promises as fs } from 'fs'
import path from 'path'

export async function GET() {
  const reports = process.env.AELP_REPORTS_DIR || path.resolve(process.cwd(), 'AELP2', 'reports')
  const read = async (name: string) => {
    try { return JSON.parse(await fs.readFile(path.join(reports, name), 'utf-8')) } catch { return null }
  }
  const test = await read('rl_test_pack.json')
  const balance = await read('rl_balance_pack.json')
  const offline = await read('rl_offline_simulation.json')
  return NextResponse.json({ test, balance, offline })
}
