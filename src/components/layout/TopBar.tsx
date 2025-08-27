'use client';

import { useState } from 'react';
import { Calendar, Download, Filter, RefreshCw, Search } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useDashboard } from '@/hooks/useDashboard';
import { formatDateTime } from '@/lib/utils';

interface TimeRangeOption {
  label: string;
  value: string;
}

const timeRangeOptions: TimeRangeOption[] = [
  { label: 'Last Hour', value: '1h' },
  { label: 'Last 24 Hours', value: '24h' },
  { label: 'Last 7 Days', value: '7d' },
  { label: 'Last 30 Days', value: '30d' },
  { label: 'Custom', value: 'custom' },
];

interface TopBarProps {
  title: string;
  showTimeRange?: boolean;
  showFilters?: boolean;
  showSearch?: boolean;
  showExport?: boolean;
  onRefresh?: () => void;
  children?: React.ReactNode;
}

export function TopBar({
  title,
  showTimeRange = true,
  showFilters = true,
  showSearch = false,
  showExport = false,
  onRefresh,
  children
}: TopBarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [showCustomDatePicker, setShowCustomDatePicker] = useState(false);
  const { 
    selectedTimeRange, 
    setTimeRangePreset, 
    setTimeRange,
    selectedAgents,
    selectedCampaigns,
    filters 
  } = useDashboard();

  const handleTimeRangeChange = (value: string) => {
    if (value === 'custom') {
      setShowCustomDatePicker(true);
    } else {
      setTimeRangePreset(value);
      setShowCustomDatePicker(false);
    }
  };

  const handleCustomDateChange = (start: string, end: string) => {
    setTimeRange({
      start: new Date(start),
      end: new Date(end),
      preset: 'custom',
    });
    setShowCustomDatePicker(false);
  };

  const getActiveFiltersCount = () => {
    let count = 0;
    if (selectedAgents.length > 0) count++;
    if (selectedCampaigns.length > 0) count++;
    if (filters.platforms.length > 0) count++;
    if (filters.agentStatus.length > 0) count++;
    if (filters.campaignStatus.length > 0) count++;
    return count;
  };

  return (
    <div className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
          <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
            {showTimeRange && (
              <div className="flex items-center gap-2">
                <Calendar size={16} />
                <span>
                  {selectedTimeRange.preset !== 'custom' 
                    ? timeRangeOptions.find(opt => opt.value === selectedTimeRange.preset)?.label
                    : `${formatDateTime(selectedTimeRange.start)} - ${formatDateTime(selectedTimeRange.end)}`
                  }
                </span>
              </div>
            )}
            {selectedAgents.length > 0 && (
              <Badge variant="info" size="sm">
                {selectedAgents.length} agent{selectedAgents.length !== 1 ? 's' : ''} selected
              </Badge>
            )}
            {selectedCampaigns.length > 0 && (
              <Badge variant="info" size="sm">
                {selectedCampaigns.length} campaign{selectedCampaigns.length !== 1 ? 's' : ''} selected
              </Badge>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {showSearch && (
            <div className="relative">
              <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          )}

          {showTimeRange && (
            <div className="relative">
              <select
                value={selectedTimeRange.preset || 'custom'}
                onChange={(e) => handleTimeRangeChange(e.target.value)}
                className="appearance-none bg-white border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                {timeRangeOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          {showCustomDatePicker && (
            <div className="absolute top-full mt-2 right-0 bg-white border border-gray-300 rounded-md shadow-lg p-4 z-10">
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Start Date
                  </label>
                  <input
                    type="datetime-local"
                    defaultValue={selectedTimeRange.start.toISOString().slice(0, 16)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                    id="start-date"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    End Date
                  </label>
                  <input
                    type="datetime-local"
                    defaultValue={selectedTimeRange.end.toISOString().slice(0, 16)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                    id="end-date"
                  />
                </div>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    onClick={() => {
                      const startInput = document.getElementById('start-date') as HTMLInputElement;
                      const endInput = document.getElementById('end-date') as HTMLInputElement;
                      handleCustomDateChange(startInput.value, endInput.value);
                    }}
                  >
                    Apply
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowCustomDatePicker(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            </div>
          )}

          {showFilters && (
            <Button variant="outline" size="sm">
              <Filter size={16} className="mr-2" />
              Filters
              {getActiveFiltersCount() > 0 && (
                <Badge variant="info" size="sm" className="ml-2">
                  {getActiveFiltersCount()}
                </Badge>
              )}
            </Button>
          )}

          {onRefresh && (
            <Button variant="outline" size="sm" onClick={onRefresh}>
              <RefreshCw size={16} className="mr-2" />
              Refresh
            </Button>
          )}

          {showExport && (
            <Button variant="outline" size="sm">
              <Download size={16} className="mr-2" />
              Export
            </Button>
          )}

          {children}
        </div>
      </div>
    </div>
  );
}