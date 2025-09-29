import type {
  DatasetInfo,
  KpiSummary,
  HeadroomResult,
  TableResult,
  CreativePerfRow,
  ChannelAttributionRow,
  Ga4ChannelRow,
  MmmAllocationRow,
  AbExperimentRow,
  AbExposureRow,
  AuctionsMinutelyRow,
} from './types'

const BASE = (import.meta.env.VITE_API_BASE_URL as string) || ''

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

const f = (path: string, init?: RequestInit) => (
  fetch(`${BASE}${path}`, { credentials: 'include', cache: 'no-store', ...init })
)

export const api = {
  dataset: {
    get: () => f('/api/dataset').then(j<DatasetInfo>),
    set: (mode: 'sandbox' | 'prod') => f(`/api/dataset?mode=${mode}`, { method: 'POST' }).then(j<DatasetInfo>),
  },
  kpi: {
    summary: () => f('/api/bq/kpi/summary').then(j<any>),
    daily: (days = 28) => f(`/api/bq/kpi/daily?days=${days}`).then(j<KpiSummary>),
    yesterday: () => f('/api/bq/kpi/yesterday').then(j<any>),
  },
  freshness: () => f('/api/bq/freshness').then(j<any>),
  headroom: () => f('/api/bq/headroom').then(j<HeadroomResult>),
  creatives: () => f('/api/bq/creatives').then(j<TableResult<CreativePerfRow>>),
  copySuggestions: () => f('/api/bq/copy-suggestions').then(j<TableResult<any>>),
  creativeVariants: () => f('/api/bq/creative-variants').then(j<TableResult<any>>),
  channelAttribution: () => f('/api/bq/channel-attribution').then(j<TableResult<ChannelAttributionRow>>),
  ga4Channels: () => f('/api/bq/ga4/channels').then(j<TableResult<Ga4ChannelRow>>),
  mmm: {
    allocations: () => f('/api/bq/mmm/allocations').then(j<TableResult<MmmAllocationRow>>),
    curves: (channel: string) => f(`/api/bq/mmm/curves?channel=${encodeURIComponent(channel)}`).then(j<any>),
  },
  approvals: () => f('/api/bq/opportunities').then(j<TableResult<any>>),
  approvalsQueue: () => f('/api/bq/approvals/queue').then(j<TableResult<any>>),
  ads: {
    creative: (ad_id: string, customer_id: string, campaign_id?: string) => {
      const params = new URLSearchParams({ ad_id, customer_id })
      if (campaign_id) params.set('campaign_id', campaign_id)
      return f(`/api/ads/creative?${params.toString()}`).then(j<any>)
    },
  },
  ab: {
    experiments: () => f('/api/bq/ab-experiments').then(j<TableResult<AbExperimentRow>>),
    exposures: () => f('/api/bq/ab-exposures').then(j<TableResult<AbExposureRow>>),
  },
  auctions: {
    minutely: () => f('/api/bq/auctions/minutely').then(j<TableResult<AuctionsMinutelyRow>>),
    policy: () => f('/api/bq/policy-enforcement').then(j<TableResult<any>>),
    opsAlerts: () => f('/api/bq/ops-alerts').then(j<TableResult<any>>),
  },
  offpolicy: () => f('/api/bq/offpolicy').then(j<TableResult<any>>),
  interference: () => f('/api/bq/interference').then(j<TableResult<any>>),
  ops: {
    flows: () => f('/api/ops/flows').then(j<TableResult<any>>),
    status: () => f('/api/control/status').then(j<any>),
  },
  chat: (prompt: string) => f('/api/chat', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt }) }).then(j<any>),
}

export type Api = typeof api
