'use client';

import { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ResourceUtilization } from '@/types';
import { formatDateTime, formatCurrency, formatPercentage } from '@/lib/utils';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

interface ResourceUtilizationChartProps {
  data: ResourceUtilization[];
  selectedMetrics?: string[];
  height?: number;
  title?: string;
}

export function ResourceUtilizationChart({ 
  data, 
  selectedMetrics = ['cpuUsage', 'memoryUsage', 'budgetSpent'], 
  height = 400,
  title = 'Resource Utilization'
}: ResourceUtilizationChartProps) {
  const chartData = useMemo(() => {
    return data.map(point => ({
      timestamp: formatDateTime(point.timestamp),
      cpuUsage: point.cpuUsage * 100,
      memoryUsage: point.memoryUsage * 100,
      gpuUsage: point.gpuUsage ? point.gpuUsage * 100 : 0,
      budgetSpent: point.budgetSpent,
      computeCost: point.computeCost,
      rawTimestamp: point.timestamp,
    })).sort((a, b) => a.rawTimestamp.getTime() - b.rawTimestamp.getTime());
  }, [data]);

  const metricColors = {
    cpuUsage: '#3b82f6',
    memoryUsage: '#10b981',
    gpuUsage: '#f59e0b',
    budgetSpent: '#ef4444',
    computeCost: '#8b5cf6',
  };

  const formatTooltipValue = (value: number, name: string) => {
    switch (name) {
      case 'cpuUsage':
        return [formatPercentage(value / 100), 'CPU Usage'];
      case 'memoryUsage':
        return [formatPercentage(value / 100), 'Memory Usage'];
      case 'gpuUsage':
        return [formatPercentage(value / 100), 'GPU Usage'];
      case 'budgetSpent':
        return [formatCurrency(value), 'Budget Spent'];
      case 'computeCost':
        return [formatCurrency(value), 'Compute Cost'];
      default:
        return [value.toFixed(2), name];
    }
  };

  // Split metrics by scale (percentage vs currency)
  const percentageMetrics = selectedMetrics.filter(m => 
    ['cpuUsage', 'memoryUsage', 'gpuUsage'].includes(m)
  );
  const currencyMetrics = selectedMetrics.filter(m => 
    ['budgetSpent', 'computeCost'].includes(m)
  );

  return (
    <div className="space-y-6">
      {percentageMetrics.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>{title} - System Resources</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={height}>
              <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis 
                  dataKey="timestamp"
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#d1d5db' }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis 
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#d1d5db' }}
                  tickFormatter={(value) => `${value}%`}
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
                {percentageMetrics.map(metric => (
                  <Area
                    key={metric}
                    type="monotone"
                    dataKey={metric}
                    stroke={metricColors[metric as keyof typeof metricColors]}
                    fill={metricColors[metric as keyof typeof metricColors]}
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {currencyMetrics.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>{title} - Cost Tracking</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={height}>
              <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis 
                  dataKey="timestamp"
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#d1d5db' }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#d1d5db' }}
                  tickFormatter={(value) => formatCurrency(value)}
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
                {currencyMetrics.map(metric => (
                  <Area
                    key={metric}
                    type="monotone"
                    dataKey={metric}
                    stroke={metricColors[metric as keyof typeof metricColors]}
                    fill={metricColors[metric as keyof typeof metricColors]}
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {data.length > 0 && (
        <Card>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Avg CPU:</span>
                <div className="font-semibold text-blue-600">
                  {formatPercentage(data.reduce((sum, d) => sum + d.cpuUsage, 0) / data.length)}
                </div>
              </div>
              <div>
                <span className="text-gray-600">Avg Memory:</span>
                <div className="font-semibold text-green-600">
                  {formatPercentage(data.reduce((sum, d) => sum + d.memoryUsage, 0) / data.length)}
                </div>
              </div>
              {data.some(d => d.gpuUsage) && (
                <div>
                  <span className="text-gray-600">Avg GPU:</span>
                  <div className="font-semibold text-yellow-600">
                    {formatPercentage(data.reduce((sum, d) => sum + (d.gpuUsage || 0), 0) / data.length)}
                  </div>
                </div>
              )}
              <div>
                <span className="text-gray-600">Total Budget:</span>
                <div className="font-semibold text-red-600">
                  {formatCurrency(Math.max(...data.map(d => d.budgetSpent)))}
                </div>
              </div>
              <div>
                <span className="text-gray-600">Total Compute:</span>
                <div className="font-semibold text-purple-600">
                  {formatCurrency(data.reduce((sum, d) => sum + d.computeCost, 0))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}