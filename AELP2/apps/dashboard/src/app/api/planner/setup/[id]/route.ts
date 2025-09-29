import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

export const dynamic = 'force-dynamic'

export async function GET(_: Request, { params }: { params: { id: string } }) {
  const id = params.id
  const base = process.env.AELP_REPORTS_DIR || path.resolve(process.cwd(), 'AELP2', 'reports')
  // Load placement hints if available from briefs/rl pack
  let placements: any = null
  try {
    const rl = JSON.parse(await fs.readFile(path.join(base, 'rl_test_pack.json'), 'utf-8'))
    placements = rl.items?.find((x:any)=> x.creative_id === id)?.placements || null
  } catch {}
  const steps = [
    { screen: 'Campaign', items: [
      'Objective: Sales (optimize for purchases)',
      'Buying type: Auction',
      'CBO: OFF (use ad set budgets) for precision; ON if you prefer auto',
      'A/B test: Off (use our offline RL plan instead)'
    ]},
    { screen: 'Ad Set', items: [
      'Conversion location: Website',
      'Pixel/Event: Aura pixel • Purchase (paid)',
      'Geography: United States',
      'Age: 25+ (adjust per plan)',
      'Placements: Manual — match package placements' + (placements?` (${JSON.stringify(placements)})`:' (Feed, Reels, Stories)') ,
      'Budget: per package (see Creative Planner totals)',
      'Schedule: Start today, run continuously',
    ]},
    { screen: 'Ad', items: [
      `Creative: ${id} (asset kit)`,
      'Primary text: from copy bank (kit)',
      'Headline: from brief',
      'Call to action: Learn More / Sign Up',
      'URL: production LP (security or Balance offer)',
      'UTM: utm_source=meta&utm_medium=paid&utm_campaign={campaign_id}&utm_content={ad_id}',
    ]},
    { screen: 'Checks', items: [
      'Brand safety toggles OK',
      'Advantage+ creative: OFF (first run)',
      'Tracking: Pixel firing on thank-you page',
    ]}
  ]
  return NextResponse.json({ creative_id: id, steps })
}

