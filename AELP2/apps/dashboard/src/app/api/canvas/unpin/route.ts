import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export const dynamic = 'force-dynamic'

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const id = String(form.get('id') || '')
    if (!id) return NextResponse.json({ ok: false, error: 'id required' }, { status: 400 })
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const usersDs = process.env.BIGQUERY_USERS_DATASET as string
    const bq = new BigQuery({ projectId })
    await bq.query({
      query: `DELETE FROM \`${projectId}.${usersDs}.canvas_pins\` WHERE id=@id`,
      params: { id },
    })
    return NextResponse.json({ ok: true, id })
  } catch (e:any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}

