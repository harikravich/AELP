'use client';

import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrainingMetrics } from '@/types';
import { formatDateTime, formatNumber } from '@/lib/utils';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

interface TrainingChartProps {
  data: TrainingMetrics[];
  selectedMetrics?: string[];
  height?: number;
  title?: string;
}

export function TrainingChart({ 
  data, 
  selectedMetrics = ['reward', 'loss'], 
  height = 400,
  title = 'Training Progress'
}: TrainingChartProps) {
  const chartData = useMemo(() => {
    return data.map(point => ({
      episode: point.episode,
      reward: point.reward,
      loss: point.loss,
      policyEntropy: point.policyEntropy,
      valueFunction: point.valueFunction,
      timestamp: point.timestamp.toISOString(),
      phase: point.phase,
    }));
  }, [data]);

  const metricColors = {
    reward: '#10b981',
    loss: '#ef4444',
    policyEntropy: '#f59e0b',
    valueFunction: '#3b82f6',
  };

  const formatTooltipValue = (value: number, name: string) => {
    switch (name) {
      case 'reward':
        return [formatNumber(value, 2), 'Reward'];
      case 'loss':
        return [formatNumber(value, 4), 'Loss'];
      case 'policyEntropy':
        return [formatNumber(value, 3), 'Policy Entropy'];
      case 'valueFunction':
        return [formatNumber(value, 2), 'Value Function'];
      default:
        return [formatNumber(value, 2), name];
    }
  };

  const formatTooltipLabel = (label: string) => {
    const point = data.find(d => d.episode.toString() === label);
    if (point) {
      return `Episode ${label} • ${formatDateTime(point.timestamp)}`;
    }
    return `Episode ${label}`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis 
              dataKey="episode" 
              tick={{ fontSize: 12 }}
              tickLine={{ stroke: '#d1d5db' }}
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
              labelFormatter={formatTooltipLabel}
            />
            <Legend />
            {selectedMetrics.map(metric => (
              <Line
                key={metric}
                type="monotone"
                dataKey={metric}
                stroke={metricColors[metric as keyof typeof metricColors] || '#6b7280'}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
        
        {data.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-4 text-sm text-gray-600">
            <div>
              <span className="font-medium">Latest Episode:</span> {data[data.length - 1]?.episode}
            </div>
            <div>
              <span className="font-medium">Phase:</span> {data[data.length - 1]?.phase}
            </div>
            <div>
              <span className="font-medium">Completion:</span> {formatNumber(data[data.length - 1]?.episodeCompletion * 100, 1)}%
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}