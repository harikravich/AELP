import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const dataset = process.env.BIGQUERY_TRAINING_DATASET as string
    if (!projectId || !dataset) throw new Error('Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET')
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({
      query: `SELECT lag, corr FROM \`${projectId}.${dataset}.impact_seed_harvest_stats\` ORDER BY lag`
    })
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 })
  }
}

