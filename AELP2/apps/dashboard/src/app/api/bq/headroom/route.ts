import { NextResponse } from 'next/server'
import { createBigQueryClient } from '../../../../lib/bigquery-client'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const { dataset } = getDatasetFromCookie()
    const bq = createBigQueryClient(projectId)
    // Use latest mmm_allocations; optionally include impression share cap if available
    try {
      const [rows] = await bq.query({ query: `
        WITH latest AS (
          SELECT channel, expected_cac, proposed_daily_budget, timestamp
          FROM \`${projectId}.${dataset}.mmm_allocations\`
          QUALIFY ROW_NUMBER() OVER (PARTITION BY channel ORDER BY timestamp DESC) = 1
        )
        SELECT channel, expected_cac AS cac, proposed_daily_budget AS proposed
        FROM latest` })
      const enriched = [] as any[]
      for (const r of rows as any[]) {
        let is_cap: number | null = null
        try {
          const [isRows] = await bq.query({ query: `
            SELECT AVG(impression_share) AS is_avg
            FROM \`${projectId}.${dataset}.ads_campaign_performance\`
            WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
              AND LOWER(channel) = LOWER(@ch)`, params: { ch: r.channel } })
          is_cap = Number(isRows?.[0]?.is_avg ?? null)
          if (!Number.isFinite(is_cap)) is_cap = null
        } catch {}
        const proposed = Math.max(0, Number(r.proposed||0))
        const cac = Number(r.cac||0)
        const extra = cac>0 ? Math.round(proposed/cac) : 0
        enriched.push({ channel: r.channel, cac, room: proposed, extra_per_day: extra, impression_share: is_cap })
      }
      return NextResponse.json({ rows: enriched })
    } catch {}
    // No allocations → return empty (no placeholders)
    return NextResponse.json({ rows: [] })
  } catch (e:any) { return NextResponse.json({ rows: [], error: e?.message||String(e) }, { status: 200 }) }
}

export const dynamic = 'force-dynamic'
