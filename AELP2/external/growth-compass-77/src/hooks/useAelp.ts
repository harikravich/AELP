import { useQuery } from '@tanstack/react-query'
import { api } from '../integrations/aelp-api/client'

export const useDataset = () => useQuery({ queryKey: ['dataset'], queryFn: api.dataset.get })
export const useKpiSummary = () => useQuery({ queryKey: ['kpi','summary'], queryFn: api.kpi.summary, staleTime: 10_000 })
export const useKpiDaily = (days = 28) => useQuery({ queryKey: ['kpi','daily',days], queryFn: () => api.kpi.daily(days), staleTime: 10_000 })
export const useHeadroom = () => useQuery({ queryKey: ['headroom'], queryFn: api.headroom, staleTime: 10_000 })
export const useCreatives = () => useQuery({ queryKey: ['creatives'], queryFn: api.creatives, staleTime: 15_000 })
export const useChannelAttribution = () => useQuery({ queryKey: ['channel-attribution'], queryFn: api.channelAttribution })
export const useGa4Channels = () => useQuery({ queryKey: ['ga4-channels'], queryFn: api.ga4Channels })
export const useMmmAllocations = () => useQuery({ queryKey: ['mmm','allocations'], queryFn: api.mmm.allocations })
export const useAuctionsMinutely = () => useQuery({ queryKey: ['auctions','minutely'], queryFn: api.auctions.minutely })
export const useOpsStatus = () => useQuery({ queryKey: ['ops','status'], queryFn: api.ops.status })
export const useApprovalsQueue = () => useQuery({ queryKey: ['approvals','queue'], queryFn: api.approvalsQueue, refetchInterval: 10000 })
export const useOffpolicy = () => useQuery({ queryKey: ['offpolicy'], queryFn: api.offpolicy })
export const useInterference = () => useQuery({ queryKey: ['interference'], queryFn: api.interference })
