import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `
      SELECT date, AVG(ltv_90) AS ltv_90
      FROM \`${projectId}.${dataset}.ltv_priors_daily\`
      GROUP BY date ORDER BY date DESC LIMIT 1
    `})
    const r:any = rows?.[0] || null
    return NextResponse.json({ date: r?.date || null, ltv_90: Number(r?.ltv_90||0), mode, dataset })
  } catch (e:any) { return NextResponse.json({ error: e?.message||String(e) }, { status: 200 }) }
}

