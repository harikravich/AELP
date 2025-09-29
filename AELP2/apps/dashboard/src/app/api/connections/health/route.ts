import { NextResponse } from 'next/server'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export async function GET() {
  const envDataset = process.env.BIGQUERY_TRAINING_DATASET || null
  const { dataset, mode } = getDatasetFromCookie()
  const checks = {
    project: process.env.GOOGLE_CLOUD_PROJECT || null,
    dataset_selected: dataset,
    dataset_mode: mode,
    dataset_env_default: envDataset,
    gatesEnabled: (process.env.GATES_ENABLED || '1') === '1',
  }
  return NextResponse.json({ ok: true, checks })
}

export const dynamic = 'force-dynamic'
