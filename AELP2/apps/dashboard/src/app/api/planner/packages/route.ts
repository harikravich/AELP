import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

export const dynamic = 'force-dynamic'

type Forecasts = {
  budgets: number[]
  items: Array<{
    creative_id: string
    p_win: number
    lcb?: number
    budgets: Record<string, {
      impressions: { p10:number,p50:number,p90:number }
      clicks: { p10:number,p50:number,p90:number }
      signups: { p10:number,p50:number,p90:number }
      cac: { p10:number,p50:number,p90:number }
      p_cac_le_240?: number
    }>
  }>
}

const loadJSON = async (p: string) => JSON.parse(await fs.readFile(p, 'utf-8'))

export async function GET(req: Request) {
  try {
    const base = process.env.AELP_REPORTS_DIR || path.resolve(process.cwd(), 'AELP2', 'reports')
    const sec: Forecasts = await loadJSON(path.join(base, 'us_cac_volume_forecasts.json'))
    const bal: Forecasts = await loadJSON(path.join(base, 'us_balance_forecasts.json'))

    function buildPackage(f: Forecasts, B: 30000|50000, k=8) {
      const key = String(B)
      const sorted = [...f.items].sort((a,b)=> (b.p_win - a.p_win))
      const pick = sorted.slice(0, k)
      const per = B / k
      let totals = { budget: B, signups_p50: 0, cac_p50_weighted: 0 }
      const rows = pick.map(r => {
        const fx = r.budgets?.[key]
        const su = fx?.signups?.p50 ?? 0
        const cac = fx?.cac?.p50 ?? 0
        totals.signups_p50 += su
        totals.cac_p50_weighted += (per / B) * cac
        return { creative_id: r.creative_id, p_win: r.p_win, budget: per, su_p50: su, cac_p50: cac }
      })
      return { budget: B, k, items: rows, totals }
    }

    const out = {
      security: { b30: buildPackage(sec, 30000), b50: buildPackage(sec, 50000) },
      balance:  { b30: buildPackage(bal, 30000), b50: buildPackage(bal, 50000) },
    }
    return NextResponse.json(out)
  } catch (e:any) {
    return NextResponse.json({ error: e?.message||String(e) }, { status: 500 })
  }
}

