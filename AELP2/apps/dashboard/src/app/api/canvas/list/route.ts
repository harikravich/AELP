import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const usersDs = process.env.BIGQUERY_USERS_DATASET as string
    const bq = new BigQuery({ projectId })
    await bq.query({ query: `
      CREATE TABLE IF NOT EXISTS \`${projectId}.${usersDs}.canvas_pins\` (
        id STRING,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        title STRING,
        viz JSON
      )` })
    const [rows] = await bq.query({ query: `
      SELECT id, created_at, title, viz FROM \`${projectId}.${usersDs}.canvas_pins\`
      ORDER BY created_at DESC LIMIT 50` })
    const items = (rows || []).map((r:any)=>{
      let v:any = r.viz
      try {
        if (typeof v === 'string') {
          // Some inserts may have a JSON string or a JSON-string-wrapped string
          v = JSON.parse(v)
          if (typeof v === 'string') v = JSON.parse(v)
        }
      } catch {}
      return { ...r, viz: v }
    })
    return NextResponse.json({ items })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
