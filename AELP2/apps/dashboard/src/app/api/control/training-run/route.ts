import { NextResponse } from 'next/server'
import { resolveDatasetForAction } from '../../../../lib/dataset'
import { spawn } from 'node:child_process'

export const dynamic = 'force-dynamic'

export async function POST(req: Request) {
  const { dataset, mode, allowed, reason } = resolveDatasetForAction('write')
  if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })

  const url = new URL(req.url)
  const episodes = parseInt(url.searchParams.get('episodes') || '1', 10)
  const steps = parseInt(url.searchParams.get('steps') || '300', 10)
  const budget = parseFloat(url.searchParams.get('budget') || '3000')
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string

  const py = process.env.PYTHON_BIN || 'python3'
  // Prefer real orchestrator if available; fallback to stub
  const script = 'AELP2/train_aura_agent.py'
  const args = [py, script, '--episodes', String(episodes), '--steps', String(steps), '--budget', String(budget)]
  try {
    const child = spawn(args[0], args.slice(1), {
      env: { ...process.env, GOOGLE_CLOUD_PROJECT: projectId, BIGQUERY_TRAINING_DATASET: dataset },
      cwd: process.cwd(),
      stdio: 'ignore',
      detached: true,
    })
    child.unref()
    return NextResponse.json({ ok: true, queued: true, dataset, episodes, steps, budget })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e), hint: 'This requires Python + dependencies in the image.' }, { status: 500 })
  }
}
