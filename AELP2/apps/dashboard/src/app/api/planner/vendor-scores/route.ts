import { NextResponse } from 'next/server'
export const dynamic = 'force-dynamic'
import { promises as fs } from 'fs'
import path from 'path'

export async function GET() {
  const base = process.env.AELP_REPORTS_DIR || path.resolve(process.cwd(), 'AELP2', 'reports')
  const p = path.join(base, 'vendor_scores.json')
  try {
    const j = JSON.parse(await fs.readFile(p, 'utf-8'))
    return NextResponse.json(j)
  } catch (e) {
    return NextResponse.json({ items: [], error: 'vendor_scores.json not found' })
  }
}
