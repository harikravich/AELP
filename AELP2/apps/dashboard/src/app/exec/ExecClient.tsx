"use client"
import React, { useState } from 'react'
import { TimeSeriesChart } from '../../components/TimeSeriesChart'
import { BanditProposalsTable } from '../../components/BanditProposalsTable'
import { Calendar, Download, RefreshCw } from 'lucide-react'
import dayjs from 'dayjs'

interface ExecClientProps {
  initialData: any
  kpiSeries: any[]
  isSeries: any[]
  banditProps: any[]
}

export default function ExecClient({ initialData, kpiSeries, isSeries, banditProps }: ExecClientProps) {
  const [dateRange, setDateRange] = useState({
    startDate: dayjs().subtract(28, 'day').format('YYYY-MM-DD'),
    endDate: dayjs().format('YYYY-MM-DD')
  })
  const [downloading, setDownloading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [filteredData, setFilteredData] = useState({
    kpi: kpiSeries,
    impression: isSeries,
    bandit: banditProps
  })

  // Filter data based on date range
  const filterDataByDateRange = () => {
    const start = dayjs(dateRange.startDate)
    const end = dayjs(dateRange.endDate)

    const filteredKpi = kpiSeries.filter(item => {
      const date = dayjs(item.date, 'MM-DD')
      return date.isAfter(start) && date.isBefore(end.add(1, 'day'))
    })

    const filteredImpression = isSeries.filter(item => {
      const date = dayjs(item.date, 'MM-DD')
      return date.isAfter(start) && date.isBefore(end.add(1, 'day'))
    })

    const filteredBandit = banditProps.filter(item => {
      // Parse the timestamp which is already formatted
      const timestamp = dayjs(item.timestamp)
      return timestamp.isAfter(start) && timestamp.isBefore(end.add(1, 'day'))
    })

    setFilteredData({
      kpi: filteredKpi,
      impression: filteredImpression,
      bandit: filteredBandit
    })
  }

  // Handle date change
  const handleDateChange = (field: 'startDate' | 'endDate', value: string) => {
    setDateRange(prev => ({
      ...prev,
      [field]: value
    }))
  }

  // Apply date filter
  const applyDateFilter = () => {
    setRefreshing(true)
    filterDataByDateRange()
    setTimeout(() => setRefreshing(false), 500)
  }

  // Download executive report
  const downloadReport = async () => {
    setDownloading(true)
    try {
      const response = await fetch(
        `/api/reports/executive?startDate=${dateRange.startDate}&endDate=${dateRange.endDate}`
      )

      if (!response.ok) {
        throw new Error('Failed to generate report')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `executive-report-${dateRange.startDate}-to-${dateRange.endDate}.html`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Download failed:', error)
      alert('Failed to download report. Please try again.')
    } finally {
      setDownloading(false)
    }
  }

  // Quick date range presets
  const setQuickRange = (days: number) => {
    setDateRange({
      startDate: dayjs().subtract(days, 'day').format('YYYY-MM-DD'),
      endDate: dayjs().format('YYYY-MM-DD')
    })
    setTimeout(applyDateFilter, 100)
  }

  return (
    <div className="space-y-6">
      {/* Date Controls */}
      <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-gray-300" />
            <span className="text-sm text-gray-300">Date Range:</span>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="date"
              value={dateRange.startDate}
              onChange={(e) => handleDateChange('startDate', e.target.value)}
              className="px-3 py-1.5 bg-white/10 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:border-blue-400"
            />
            <span className="text-gray-400">to</span>
            <input
              type="date"
              value={dateRange.endDate}
              onChange={(e) => handleDateChange('endDate', e.target.value)}
              className="px-3 py-1.5 bg-white/10 border border-white/20 rounded-lg text-white text-sm focus:outline-none focus:border-blue-400"
            />
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setQuickRange(7)}
              className="px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-lg text-sm transition-colors"
            >
              Last 7 Days
            </button>
            <button
              onClick={() => setQuickRange(14)}
              className="px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-lg text-sm transition-colors"
            >
              Last 14 Days
            </button>
            <button
              onClick={() => setQuickRange(28)}
              className="px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-lg text-sm transition-colors"
            >
              Last 28 Days
            </button>
            <button
              onClick={() => setQuickRange(90)}
              className="px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-lg text-sm transition-colors"
            >
              Last 90 Days
            </button>
          </div>

          <div className="flex items-center gap-2 ml-auto">
            <button
              onClick={applyDateFilter}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg text-sm transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              Apply Filter
            </button>

            <button
              onClick={downloadReport}
              disabled={downloading}
              className="flex items-center gap-2 px-4 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg text-sm transition-colors"
            >
              <Download className={`w-4 h-4 ${downloading ? 'animate-pulse' : ''}`} />
              {downloading ? 'Generating...' : 'Executive Report'}
            </button>
          </div>
        </div>
      </div>

      {/* Charts with filtered data */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass-card p-6 rounded-xl hover-lift">
          <h3 className="text-xl font-semibold mb-4 text-gray-100">
            KPI Trends
            {refreshing && <span className="ml-2 text-sm text-blue-400">(Updating...)</span>}
          </h3>
          <TimeSeriesChart
            data={filteredData.kpi}
            height={320}
            series={[
              { name: 'Conversions', dataKey: 'conversions', color: '#8b5cf6', yAxisId: 'left' },
              { name: 'Cost ($)', dataKey: 'cost', color: '#ef4444', yAxisId: 'left' },
              { name: 'Revenue ($)', dataKey: 'revenue', color: '#10b981', yAxisId: 'left' },
            ]}
          />
        </div>

        <div className="glass-card p-6 rounded-xl hover-lift">
          <h3 className="text-xl font-semibold mb-4 text-gray-100">
            Efficiency Metrics
            {refreshing && <span className="ml-2 text-sm text-blue-400">(Updating...)</span>}
          </h3>
          <TimeSeriesChart
            data={filteredData.kpi}
            height={320}
            series={[
              { name: 'CAC ($)', dataKey: 'cac', color: '#f97316', yAxisId: 'left' },
              { name: 'ROAS (x)', dataKey: 'roas', color: '#06b6d4', yAxisId: 'right' },
            ]}
          />
        </div>
      </div>

      {/* Bandit Proposals with filtered data */}
      <div className="glass-card p-6 rounded-xl hover-lift">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-gray-100">
            Optimization Proposals ({filteredData.bandit.length})
          </h3>
        </div>
        <BanditProposalsTable proposals={filteredData.bandit} />
      </div>
    </div>
  )
}