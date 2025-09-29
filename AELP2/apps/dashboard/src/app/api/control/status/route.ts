import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const ds = `${projectId}.${dataset}`
  const sql = `
    SELECT 'ads_ingest_last_date' AS k, CAST(MAX(DATE(date)) AS STRING) AS v FROM \`${ds}.ads_campaign_performance\`
    UNION ALL
    SELECT 'ga4_ingest_last_date', CAST(MAX(DATE(date)) AS STRING) FROM \`${ds}.ga4_aggregates\`
    UNION ALL
    SELECT 'ga4_attribution_last_date', CAST(MAX(DATE(date)) AS STRING) FROM \`${ds}.ga4_lagged_attribution\`
    UNION ALL
    SELECT 'training_last_timestamp', CAST(MAX(TIMESTAMP(timestamp)) AS STRING) FROM \`${ds}.training_episodes\`
    UNION ALL
    SELECT 'fidelity_last_timestamp', CAST(MAX(TIMESTAMP(timestamp)) AS STRING) FROM \`${ds}.fidelity_evaluations\`
     UNION ALL
    SELECT 'impact_invoices_last_date', CAST(MAX(DATE(CreatedDate)) AS STRING) FROM \`${ds}.impact_invoices\`
     UNION ALL
    SELECT 'impact_partner_daily_last_date', CAST(MAX(DATE(date)) AS STRING) FROM \`${ds}.impact_partner_daily\`
    `
  try {
    const [rows] = await bq.query({ query: sql })
    const map: Record<string, string|null> = {}
    for (const r of rows as any[]) map[r.k] = r.v ?? null
    // Best-effort extra: impact_performance_last_date (if table exists)
    try {
      const [r2] = await bq.query({
        query: `SELECT CAST(MAX(DATE(date)) AS STRING) AS v FROM \`${ds}.impact_partner_performance\``
      })
      if (r2 && (r2 as any[])[0]) map['impact_performance_last_date'] = (r2 as any[])[0].v
    } catch (_ignored) {}
    return NextResponse.json({ ok: true, status: map })
  } catch (e: any) {
    return NextResponse.json({ ok: false, error: e?.message || String(e) }, { status: 200 })
  }
}
