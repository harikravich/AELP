import React from 'react'
import { BigQuery } from '@google-cloud/bigquery'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../lib/dataset'
import { fmtWhen } from '../../lib/utils'
import Card from '../../components/Card'
import JourneysSankey from '../../components/JourneysSankey'

async function fetchGa4() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  let byDevice: any[] = []
  let byChannel: any[] = []
  let lagged: any[] = []
  try {
    const [rows] = await bq.query({ query: `
      SELECT date, device_category, SUM(conversions) AS conv
      FROM \`${projectId}.${dataset}.ga4_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
      GROUP BY date, device_category ORDER BY date DESC, conv DESC LIMIT 28
    `})
    byDevice = rows as any[]
  } catch {}
  try {
    const [rows] = await bq.query({ query: `
      SELECT default_channel_group, SUM(conversions) AS conv
      FROM \`${projectId}.${dataset}.ga4_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      GROUP BY default_channel_group ORDER BY conv DESC LIMIT 12
    `})
    byChannel = rows as any[]
  } catch {}
  try {
    const [rows] = await bq.query({ query: `
      SELECT date, ga4_conversions_lagged FROM \`${projectId}.${dataset}.ga4_lagged_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      ORDER BY date DESC LIMIT 28
    `})
    lagged = rows as any[]
  } catch {}
  let paths: any[] = []
  try {
    const [rows] = await bq.query({ query: `
      SELECT date, path, count FROM \`${projectId}.${dataset}.journey_paths_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
      ORDER BY date DESC, count DESC LIMIT 20
    `})
    paths = rows as any[]
  } catch {}
  return { byDevice, byChannel, lagged, paths }
}

export default async function JourneysPage() {
  const { byDevice, byChannel, lagged, paths } = await fetchGa4()
  return (
    <div className="space-y-6">
      <Card title="GA4 Conversions by Device (14d)">
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            {byDevice.slice(0,12).map((r:any,i:number)=> (
              <div key={i} className="flex justify-between border-b py-1 last:border-0">
                <span>{fmtWhen(r.date)} · {r.device_category}</span>
                <span>{Number(r.conv||0)}</span>
              </div>
            ))}
          </div>
          <div>
            <h3 className="text-sm font-semibold">Top Channels (28d)</h3>
            {byChannel.map((r:any,i:number)=> (
              <div key={i} className="flex justify-between border-b py-1 last:border-0">
                <span>{r.default_channel_group}</span>
                <span>{Number(r.conv||0)}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card title="Lagged Attribution (GA4, 28d)">
        <div className="mt-3 text-sm">
          {lagged.map((r:any,i:number)=> (
            <div key={i} className="flex justify-between border-b py-1 last:border-0">
              <span>{fmtWhen(r.date)}</span>
              <span>{Number(r.ga4_conversions_lagged||0)}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Journey Paths (Sankey)" subtitle="Enable GA4 events export for richer path depth">
        <JourneysSankey />
      </Card>

      <Card title="Top Paths (14d)">
        <div className="mt-3 text-sm">
          {paths.length === 0 ? (
            <div className="text-gray-500">No journey path rows yet.</div>
          ) : (
            paths.map((r:any,i:number)=> (
              <div key={i} className="flex justify-between border-b py-1 last:border-0">
                <span>{fmtWhen(r.date)} · {r.path}</span>
                <span>{Number(r.count||0)}</span>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  )
}

export const dynamic = 'force-dynamic'
