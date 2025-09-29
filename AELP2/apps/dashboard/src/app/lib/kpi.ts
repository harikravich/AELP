import { cookies } from 'next/headers'

export type KpiSource = 'ads' | 'ga4_all' | 'ga4_google' | 'pacer'
export const KPI_COOKIE = 'aelp-kpi-source'

export function getKpiSourceFromCookie(): KpiSource {
  const c = cookies().get(KPI_COOKIE)?.value as KpiSource | undefined
  if (c === 'ga4_all' || c === 'ga4_google' || c === 'ads' || c === 'pacer') return c
  return 'ga4_google'
}
