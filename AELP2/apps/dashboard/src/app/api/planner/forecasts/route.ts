import { NextResponse } from 'next/server'
export const dynamic = 'force-dynamic'
import { promises as fs } from 'fs'
import path from 'path'

export async function GET() {
  const reports = process.env.AELP_REPORTS_DIR || path.resolve(process.cwd(), 'AELP2', 'reports')
  async function readJSON(p: string) {
    try { return JSON.parse(await fs.readFile(p, 'utf-8')) } catch { return null }
  }
  const security = await readJSON(path.join(reports, 'us_cac_volume_forecasts.json'))
  const balance = await readJSON(path.join(reports, 'us_balance_forecasts.json'))
  return NextResponse.json({ security, balance })
}
