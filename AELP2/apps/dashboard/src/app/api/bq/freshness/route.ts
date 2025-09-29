import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../../../lib/dataset'
import { createBigQueryClient } from '../../../../lib/bigquery-client'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)

    // Optionally include GA4 export freshness from a US dataset if present.
    // We use gaelp_users (US) to avoid cross-region issues with gaelp_training (us-central1).
    const targets: Array<{ table: string, field: string, datasetOverride?: string, location?: string }> = [
      { table: 'ads_campaign_performance', field: 'date' },
      { table: 'training_episodes', field: 'timestamp' },
      { table: 'ga4_daily', field: 'date' },
      // Best-effort: show native GA4 export freshness if view exists in US region
      { table: 'ga4_export_daily', field: 'date', datasetOverride: 'gaelp_users', location: 'US' },
    ]

    const rows: any[] = []
    for (const t of targets) {
      try {
        // Cast MAX() to STRING so client code always gets a plain string
        const ds = t.datasetOverride || dataset
        const [r] = await bq.query({
          query: `SELECT CAST(MAX(${t.field}) AS STRING) AS max_date FROM \`${projectId}.${ds}.${t.table}\``,
          // Provide location for US-scoped GA4 export queries
          ...(t.location ? { location: t.location } : {}),
        })
        const val = r?.[0]?.max_date ?? null
        const name = t.datasetOverride ? `${t.table} (${t.datasetOverride})` : t.table
        rows.push({ table_name: name, max_date: typeof val === 'string' ? val : (val ? String(val) : null) })
      } catch {
        const name = t.datasetOverride ? `${t.table} (${t.datasetOverride})` : t.table
        rows.push({ table_name: name, max_date: null })
      }
    }
    return NextResponse.json({ rows })
  } catch (e: any) {
    return NextResponse.json({ rows: [], error: e?.message || String(e) }, { status: 200 })
  }
}
