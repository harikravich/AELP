import { NextResponse } from 'next/server'
import { resolveDatasetForAction } from '../../../../lib/dataset'
import { spawn } from 'node:child_process'
import path from 'node:path'

export const dynamic = 'force-dynamic'

export async function POST(req: Request) {
  // Safety: block writes on prod
  const { dataset, mode, allowed, reason } = resolveDatasetForAction('write')
  if (!allowed) {
    return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })
  }

  const url = new URL(req.url)
  const only = (url.searchParams.get('only') || '').trim()
  if (!/^\d{10}$/.test(only)) {
    return NextResponse.json({ ok: false, error: 'Provide ?only=<10_digit_personal_customer_id>' }, { status: 400 })
  }
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mcc = process.env.GOOGLE_ADS_LOGIN_CUSTOMER_ID || ''
  if (!/^\d{6,}$/.test(mcc)) {
    return NextResponse.json({ ok: false, error: 'Set env GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC) on the service' }, { status: 400 })
  }

  // Attempt to run the bundled script if present. If not, return 202 with instructions.
  const root = process.env.AELP2_ROOT || process.cwd()
  const script = path.join(root, 'AELP2/scripts/run_ads_ingestion.sh')
  const args = ['bash', script, '--mcc', mcc, '--last14', '--tasks', 'campaigns,ad_performance,conversion_actions,conversion_action_stats', '--only', only]

  try {
    const child = spawn(args[0], args.slice(1), {
      env: {
        ...process.env,
        GOOGLE_CLOUD_PROJECT: projectId,
        BIGQUERY_TRAINING_DATASET: dataset,
        AELP2_BQ_USE_GCE: '1',
        PYTHONPATH: root,
      },
      cwd: root,
      stdio: 'ignore',
      detached: true,
    })
    child.unref()
    return NextResponse.json({ ok: true, queued: true, dataset, only })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: `Spawn failed: ${e?.message || String(e)}`, hint: 'Ensure /workspace/AELP2 exists in the image and Ads env vars are set.' }, { status: 500 })
  }
}
