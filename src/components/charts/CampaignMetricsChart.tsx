'use client';

import { useMemo } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { CampaignMetrics } from '@/types';
import { formatDateTime, formatCurrency, formatNumber, formatPercentage } from '@/lib/utils';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

interface CampaignMetricsChartProps {
  data: CampaignMetrics[];
  selectedMetrics?: string[];
  chartType?: 'line' | 'area';
  height?: number;
  title?: string;
  groupBy?: 'hour' | 'day' | 'platform';
}

export function CampaignMetricsChart({ 
  data, 
  selectedMetrics = ['spend', 'revenue', 'roas'], 
  chartType = 'line',
  height = 400,
  title = 'Campaign Performance',
  groupBy = 'hour'
}: CampaignMetricsChartProps) {
  const chartData = useMemo(() => {
    if (groupBy === 'platform') {
      // Group by platform
      const platformData = data.reduce((acc, metric) => {
        const platform = metric.platform;
        if (!acc[platform]) {
          acc[platform] = {
            platform,
            spend: 0,
            revenue: 0,
            impressions: 0,
            clicks: 0,
            conversions: 0,
            count: 0,
          };
        }
        
        acc[platform].spend += metric.spend;
        acc[platform].revenue += metric.revenue;
        acc[platform].impressions += metric.impressions;
        acc[platform].clicks += metric.clicks;
        acc[platform].conversions += metric.conversions;
        acc[platform].count += 1;
        
        return acc;
      }, {} as Record<string, any>);

      return Object.values(platformData).map(platform => ({
        ...platform,
        ctr: platform.clicks / platform.impressions,
        conversionRate: platform.conversions / platform.clicks,
        roas: platform.revenue / platform.spend,
        cpc: platform.spend / platform.clicks,
        cpa: platform.spend / platform.conversions,
      }));
    } else {
      // Group by time
      const timeGrouping = groupBy === 'hour' ? 1000 * 60 * 60 : 1000 * 60 * 60 * 24;
      const groupedData = data.reduce((acc, metric) => {
        const timeKey = Math.floor(metric.timestamp.getTime() / timeGrouping) * timeGrouping;
        const key = new Date(timeKey).toISOString();
        
        if (!acc[key]) {
          acc[key] = {
            timestamp: new Date(timeKey),
            spend: 0,
            revenue: 0,
            impressions: 0,
            clicks: 0,
            conversions: 0,
            count: 0,
          };
        }
        
        acc[key].spend += metric.spend;
        acc[key].revenue += metric.revenue;
        acc[key].impressions += metric.impressions;
        acc[key].clicks += metric.clicks;
        acc[key].conversions += metric.conversions;
        acc[key].count += 1;
        
        return acc;
      }, {} as Record<string, any>);

      return Object.values(groupedData)
        .map(group => ({
          ...group,
          ctr: group.clicks / group.impressions,
          conversionRate: group.conversions / group.clicks,
          roas: group.revenue / group.spend,
          cpc: group.spend / group.clicks,
          cpa: group.spend / group.conversions,
          timeKey: formatDateTime(group.timestamp),
        }))
        .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
    }
  }, [data, groupBy]);

  const metricColors = {
    spend: '#ef4444',
    revenue: '#10b981',
    roas: '#3b82f6',
    impressions: '#8b5cf6',
    clicks: '#f59e0b',
    conversions: '#06b6d4',
    ctr: '#84cc16',
    conversionRate: '#f97316',
    cpc: '#ec4899',
    cpa: '#6366f1',
  };

  const formatTooltipValue = (value: number, name: string) => {
    switch (name) {
      case 'spend':
      case 'revenue':
      case 'cpc':
      case 'cpa':
        return [formatCurrency(value), name.toUpperCase()];
      case 'ctr':
      case 'conversionRate':
        return [formatPercentage(value), name === 'ctr' ? 'CTR' : 'Conv. Rate'];
      case 'roas':
        return [`${formatNumber(value, 2)}x`, 'ROAS'];
      case 'impressions':
      case 'clicks':
      case 'conversions':
        return [formatNumber(value), name.charAt(0).toUpperCase() + name.slice(1)];
      default:
        return [formatNumber(value, 2), name];
    }
  };

  const xAxisKey = groupBy === 'platform' ? 'platform' : 'timeKey';

  const ChartComponent = chartType === 'area' ? AreaChart : LineChart;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ChartComponent data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis 
              dataKey={xAxisKey}
              tick={{ fontSize: 12 }}
              tickLine={{ stroke: '#d1d5db' }}
              angle={groupBy === 'platform' ? 0 : -45}
              textAnchor={groupBy === 'platform' ? 'middle' : 'end'}
              height={groupBy === 'platform' ? 50 : 80}
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              tickLine={{ stroke: '#d1d5db' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              }}
              formatter={formatTooltipValue}
            />
            <Legend />
            {selectedMetrics.map(metric => (
              chartType === 'area' ? (
                <Area
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={metricColors[metric as keyof typeof metricColors] || '#6b7280'}
                  fill={metricColors[metric as keyof typeof metricColors] || '#6b7280'}
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
              ) : (
                <Line
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={metricColors[metric as keyof typeof metricColors] || '#6b7280'}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )
            ))}
          </ChartComponent>
        </ResponsiveContainer>
        
        {data.length > 0 && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Total Spend:</span>
              <div className="font-semibold text-red-600">
                {formatCurrency(data.reduce((sum, d) => sum + d.spend, 0))}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Total Revenue:</span>
              <div className="font-semibold text-green-600">
                {formatCurrency(data.reduce((sum, d) => sum + d.revenue, 0))}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Avg ROAS:</span>
              <div className="font-semibold text-blue-600">
                {formatNumber(data.reduce((sum, d) => sum + d.roas, 0) / data.length, 2)}x
              </div>
            </div>
            <div>
              <span className="text-gray-600">Total Conversions:</span>
              <div className="font-semibold">
                {formatNumber(data.reduce((sum, d) => sum + d.conversions, 0))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}