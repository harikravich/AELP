import { NextResponse } from 'next/server'
import { resolveDatasetForAction } from '../../../../lib/dataset'
import { spawn } from 'node:child_process'

export const dynamic = 'force-dynamic'

export async function POST() {
  const { dataset, mode, allowed, reason } = resolveDatasetForAction('write')
  if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })

  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const script = '/workspace/AELP2/scripts/run_ga4_attribution.sh'
  const args: string[] = [script]
  try {
    const child = spawn('bash', args, {
      env: { ...process.env, GOOGLE_CLOUD_PROJECT: projectId, BIGQUERY_TRAINING_DATASET: dataset, PYTHONPATH: '/workspace' },
      cwd: '/workspace',
      stdio: 'ignore',
      detached: true,
    })
    child.unref()
    return NextResponse.json({ ok: true, queued: true, dataset })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 500 })
  }
}
