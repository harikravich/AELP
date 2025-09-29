import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export const dynamic = 'force-dynamic'

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const title = String(form.get('title') || 'Chart')
    const vizStr = String(form.get('viz') || '')
    if (!vizStr) return NextResponse.json({ error: 'Missing viz' }, { status: 400 })
    const viz = JSON.parse(vizStr)
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const usersDs = process.env.BIGQUERY_USERS_DATASET as string
    const bq = new BigQuery({ projectId })
    // Ensure table
    await bq.query({ query: `
      CREATE TABLE IF NOT EXISTS \`${projectId}.${usersDs}.canvas_pins\` (
        id STRING,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        title STRING,
        viz JSON
      )` })
    const id = crypto.randomUUID()
    const vizStr2 = JSON.stringify(viz)
    await bq.query({
      query: `INSERT \`${projectId}.${usersDs}.canvas_pins\` (id, title, viz) VALUES (@id, @title, TO_JSON(@viz))`,
      params: { id, title, viz: vizStr2 },
    })
    return NextResponse.redirect(new URL('/canvas', req.url))
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
