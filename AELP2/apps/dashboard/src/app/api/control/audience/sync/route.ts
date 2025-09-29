import { NextResponse } from 'next/server'
import { resolveDatasetForAction } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function POST() {
  const { allowed, reason, mode, dataset } = resolveDatasetForAction('write')
  if (!allowed) return NextResponse.json({ ok: false, error: reason, mode, dataset }, { status: 403 })
  // Stub: enqueue an audience sync job (no-op here)
  return NextResponse.json({ ok: true, queued: true, dataset })
}

