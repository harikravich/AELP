export type DatasetMode = 'sandbox' | 'prod'

export interface DatasetInfo { mode: DatasetMode; dataset: string }

export interface KpiRow {
  date: string | Date
  conversions: number
  revenue: number
  cost: number
  cac?: number
  roas?: number
}

export interface KpiSummary {
  rows: KpiRow[]
  error?: string
  dataset?: string
}

export interface HeadroomRow {
  channel: string
  cac: number
  room: number
  extra_per_day: number
}

export interface HeadroomResult { rows: HeadroomRow[]; error?: string }

export interface CreativePerfRow {
  date: string
  ad_group_id: string
  ad_id: string
  campaign_id?: string
  customer_id?: string
  impressions: number
  clicks: number
  cost: number
  conversions: number
  revenue: number
  ctr: number
  cvr: number
  cac: number
  roas: number
}

export interface TableResult<T> { rows: T[]; error?: string }

export interface ApprovalsRow {
  run_id: string
  platform: string
  type: string
  enqueued_at: string
}

export interface ChannelAttributionRow {
  channel: string
  conversions: number
  revenue: number
  cost: number
  cac: number
  roas: number
}

export interface Ga4ChannelRow {
  channel: string
  sessions: number
  purchases: number
  revenue: number
}

export interface MmmAllocationRow {
  channel: string
  expected_cac?: number
  proposed_daily_budget?: number
}

export interface AbExperimentRow {
  experiment_id: string
  name: string
  status: string
  start_date?: string
  end_date?: string
  uplift?: number
}

export interface AbExposureRow {
  date: string
  experiment_id: string
  variant: string
  impressions: number
  clicks: number
  conversions: number
}

export interface AuctionsMinutelyRow {
  ts: string
  impressions: number
  clicks: number
  cpc: number
  cpm: number
}
