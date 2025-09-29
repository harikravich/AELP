import React from 'react'
import { BigQuery } from '@google-cloud/bigquery'
import { TimeSeriesChart } from '../../components/TimeSeriesChart'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../lib/dataset'
import { Input } from '../../components/ui/input'
import { Button } from '../../components/ui/button'

type MinuteRow = { minute: string, auctions: number, wins: number, win_rate: number, avg_bid: number, avg_price_paid: number }

async function fetchPerMinute(): Promise<MinuteRow[]> {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  try {
    const sql = `
      SELECT * FROM \`${projectId}.${dataset}.bidding_events_per_minute\`
      WHERE minute >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
      ORDER BY minute ASC`
    const [rows] = await bq.query({ query: sql })
    return rows as MinuteRow[]
  } catch {
    // If table is missing or query fails, treat as empty to enable empty state
    return []
  }
}

async function fetchRecent(limit = 2000): Promise<any[]> {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })
  try {
    const sql = `
      SELECT timestamp, bid_amount, price_paid, won, episode_id, step
      FROM \`${projectId}.${dataset}.bidding_events\`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
      ORDER BY timestamp DESC
      LIMIT ${Math.min(Math.max(limit, 100), 5000)}`
    const [rows] = await bq.query({ query: sql })
    return rows as any[]
  } catch {
    // Gracefully fall back to no data
    return []
  }
}

function Gauge({ value, minBand, maxBand }: { value: number, minBand: number, maxBand: number }) {
  const pct = Math.max(0, Math.min(1, value))
  const bandStart = Math.max(0, Math.min(1, minBand))
  const bandEnd = Math.max(0, Math.min(1, maxBand))
  const color = (pct >= bandStart && pct <= bandEnd) ? '#16a34a' : '#ef4444'
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>0%</span><span>100%</span>
      </div>
      <div className="relative h-4 bg-gray-200 rounded">
        <div className="absolute top-0 left-[calc(100%*var(--start))] h-4" style={{ width: `calc(${(bandEnd - bandStart) * 100}% )`, ['--start' as any]: bandStart }}>
          <div className="h-4 bg-amber-200 opacity-70 rounded" />
        </div>
        <div className="h-4 rounded" style={{ width: `${pct * 100}%`, backgroundColor: color }} />
      </div>
      <div className="mt-1 text-sm">
        <span className="font-semibold" style={{ color }}>{(pct * 100).toFixed(1)}%</span>
        <span className="ml-2 text-xs text-gray-500">target {Math.round(bandStart*100)}%-{Math.round(bandEnd*100)}%</span>
      </div>
    </div>
  )
}

export default async function AuctionsMonitor() {
  const perMin = await fetchPerMinute().catch(() => [])
  const recent = await fetchRecent().catch(() => [])
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const latest = perMin.length > 0 ? perMin[perMin.length - 1] : null
  const minBand = Number(process.env.AELP2_TARGET_WIN_RATE_MIN || '0.30')
  const maxBand = Number(process.env.AELP2_TARGET_WIN_RATE_MAX || '0.60')
  const priceSeries = perMin.map(r => ({ time: String((r as any).minute), price: Number((r as any).avg_price_paid || 0), bid: Number((r as any).avg_bid || 0), wr: Number((r as any).win_rate || 0) }))
  const priceSeriesPlain = JSON.parse(JSON.stringify(priceSeries))
  let totalAuctions = 0
  let totalWins = 0
  let winRateSum = 0
  for (const r of perMin) {
    totalAuctions += Number(r.auctions || 0)
    totalWins += Number(r.wins || 0)
    winRateSum += Number(r.win_rate || 0)
  }
  const avgWinRateStr: string = (winRateSum / Math.max(perMin.length, 1)).toFixed(3)

  // Build histogram of recent bid amounts
  const buckets: { [k: string]: number } = {}
  // Simpler: fixed-width $0.5 buckets between $0 and $10
  for (const r of recent) {
    const b = Math.max(0, Math.min(10, Number(r.bid_amount || 0)))
    const idx = Math.floor(b / 0.5)
    const label = `$${(idx*0.5).toFixed(1)}-$${((idx+1)*0.5).toFixed(1)}`
    buckets[label] = (buckets[label] || 0) + 1
  }
  const hist = Object.entries(buckets).map(([label, count]) => ({ label, count }))
  hist.sort((a, b) => parseFloat(a.label.split('-')[0].substring(1)) - parseFloat(b.label.split('-')[0].substring(1)))

  if (perMin.length === 0 && recent.length === 0) {
    return (
      <div className="space-y-4">
        <div className="bg-white rounded shadow-sm p-4">
          <h2 className="text-lg font-medium">Auctions Monitor</h2>
          <p className="text-sm text-slate-600 mt-2">No auction data found in {dataset}. Ensure tables <code>bidding_events_per_minute</code> and <code>bidding_events</code> exist and are populated. You can trigger pipelines from Control or via /api/control endpoints.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded shadow-sm p-4">
          <div className="text-sm font-medium">Win Rate vs Target</div>
          <div className="text-xs text-gray-500 mb-2">Latest per-minute</div>
          <Gauge value={Number(latest?.win_rate || 0)} minBand={minBand} maxBand={maxBand} />
        </div>
        <div className="bg-white rounded shadow-sm p-4">
          <div className="text-sm font-medium">Auctions (24h)</div>
          <div className="text-2xl font-semibold">{totalAuctions}</div>
          <div className="text-xs text-gray-500">Wins: {totalWins}</div>
        </div>
        <div className="bg-white rounded shadow-sm p-4">
          <div className="text-sm font-medium">Avg Win Rate (24h)</div>
          <div className="text-2xl font-semibold">{avgWinRateStr}</div>
        </div>
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Price Paid Trend (and Avg Bid)</h2>
        <div className="mt-4">
          <TimeSeriesChart
            data={priceSeriesPlain}
            series={[
              { name: 'Avg Price Paid', dataKey: 'price', color: '#ef4444', yAxisId: 'left' },
              { name: 'Avg Bid', dataKey: 'bid', color: '#6366f1', yAxisId: 'left' },
              { name: 'Win Rate', dataKey: 'wr', color: '#14b8a6', yAxisId: 'right' },
            ]}
          />
        </div>
      </div>

      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Bid Distribution (last 7 days)</h2>
        <div className="mt-4 overflow-x-auto">
          <div className="flex items-end gap-2 h-40">
            {hist.map((h) => (
              <div key={h.label} className="flex flex-col items-center" style={{ minWidth: 28 }}>
                <div className="bg-indigo-500 w-6 rounded-t" style={{ height: `${(h.count / Math.max(...hist.map(x=>x.count),1)) * 100}%` }} />
                <div className="text-[10px] text-gray-600 rotate-45 origin-top-left mt-2">{h.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <BidReplayInspector />
    </div>
  )
}

function BidReplayInspector() {
  // Client component wrapper via RSC boundary
  return (
    <div className="bg-white shadow-sm rounded p-4">
      <h2 className="text-lg font-medium">Bid Replay Inspector</h2>
      <p className="text-sm text-gray-600">Lookup explanation by episode_id and step.</p>
      {/* Simple form that posts to the API and renders result on client */}
      <form className="mt-3 flex items-center gap-2" action="/api/bq/bidding_replay" method="GET" target="_blank">
        <Input name="episode_id" placeholder="episode_id" className="w-64" />
        <Input name="step" placeholder="step" type="number" className="w-24" />
        <Button type="submit" size="sm">Open</Button>
      </form>
      <div className="text-xs text-gray-500 mt-2">Tip: Use Training Center to find latest episode IDs.</div>
    </div>
  )
}

export const dynamic = 'force-dynamic'
