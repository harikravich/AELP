import { NextResponse } from 'next/server'
import { resolveDatasetForAction } from '../../../../lib/dataset'
import { spawn } from 'node:child_process'
import path from 'node:path'

export const dynamic = 'force-dynamic'

export async function POST() {
  const { dataset, mode, allowed, reason } = resolveDatasetForAction('write')
  if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })

  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const ga4Prop = process.env.GA4_PROPERTY_ID || ''
  if (!ga4Prop.startsWith('properties/')) {
    return NextResponse.json({ ok: false, error: 'Set GA4_PROPERTY_ID=properties/<id> on the service' }, { status: 400 })
  }
  const root = process.env.AELP2_ROOT || process.cwd()
  const script = path.join(root, 'AELP2/scripts/run_ga4_ingestion.sh')
  const args = ['bash', script, '--last28']
  try {
    const child = spawn(args[0], args.slice(1), {
      env: { ...process.env, GOOGLE_CLOUD_PROJECT: projectId, BIGQUERY_TRAINING_DATASET: dataset, GA4_PROPERTY_ID: ga4Prop, PYTHONPATH: '/workspace' },
      cwd: root,
      stdio: 'ignore',
      detached: true,
    })
    child.unref()
    return NextResponse.json({ ok: true, queued: true, dataset })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 500 })
  }
}
