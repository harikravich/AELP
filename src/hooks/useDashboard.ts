import { create } from 'zustand';
import { DashboardState, TimeRange, DashboardFilters } from '@/types';
import { getTimeRangePreset } from '@/lib/utils';

interface DashboardStore extends DashboardState {
  // Actions
  setTimeRange: (range: TimeRange) => void;
  setTimeRangePreset: (preset: string) => void;
  setSelectedAgents: (agentIds: string[]) => void;
  setSelectedCampaigns: (campaignIds: string[]) => void;
  setActiveView: (view: DashboardState['activeView']) => void;
  setFilters: (filters: Partial<DashboardFilters>) => void;
  resetFilters: () => void;
  addSelectedAgent: (agentId: string) => void;
  removeSelectedAgent: (agentId: string) => void;
  addSelectedCampaign: (campaignId: string) => void;
  removeSelectedCampaign: (campaignId: string) => void;
}

const defaultTimeRange: TimeRange = {
  ...getTimeRangePreset('24h'),
  preset: '24h',
};

const defaultFilters: DashboardFilters = {
  platforms: [],
  agentStatus: [],
  campaignStatus: [],
};

export const useDashboard = create<DashboardStore>((set, get) => ({
  // State
  selectedTimeRange: defaultTimeRange,
  selectedAgents: [],
  selectedCampaigns: [],
  activeView: 'overview',
  filters: defaultFilters,

  // Actions
  setTimeRange: (range) => set({ selectedTimeRange: range }),
  
  setTimeRangePreset: (preset) => {
    const range = getTimeRangePreset(preset);
    set({
      selectedTimeRange: {
        ...range,
        preset: preset as TimeRange['preset'],
      },
    });
  },

  setSelectedAgents: (agentIds) => set({ selectedAgents: agentIds }),
  
  setSelectedCampaigns: (campaignIds) => set({ selectedCampaigns: campaignIds }),
  
  setActiveView: (view) => set({ activeView: view }),
  
  setFilters: (newFilters) => set((state) => ({
    filters: { ...state.filters, ...newFilters },
  })),
  
  resetFilters: () => set({ filters: defaultFilters }),
  
  addSelectedAgent: (agentId) => set((state) => ({
    selectedAgents: [...new Set([...state.selectedAgents, agentId])],
  })),
  
  removeSelectedAgent: (agentId) => set((state) => ({
    selectedAgents: state.selectedAgents.filter(id => id !== agentId),
  })),
  
  addSelectedCampaign: (campaignId) => set((state) => ({
    selectedCampaigns: [...new Set([...state.selectedCampaigns, campaignId])],
  })),
  
  removeSelectedCampaign: (campaignId) => set((state) => ({
    selectedCampaigns: state.selectedCampaigns.filter(id => id !== campaignId),
  })),
}));