'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Activity, 
  TrendingUp, 
  DollarSign, 
  Users, 
  AlertTriangle,
  Zap,
  Target,
  BarChart3
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TrainingChart } from '@/components/charts/TrainingChart';
import { CampaignMetricsChart } from '@/components/charts/CampaignMetricsChart';
import { ResourceUtilizationChart } from '@/components/charts/ResourceUtilizationChart';
import { useDashboard } from '@/hooks/useDashboard';
import { useWebSocket } from '@/hooks/useWebSocket';
import { formatCurrency, formatNumber, formatPercentage, formatTimeAgo } from '@/lib/utils';
import { Agent, TrainingMetrics, CampaignMetrics, SafetyEvent } from '@/types';

interface OverviewStats {
  totalAgents: number;
  activeAgents: number;
  totalCampaigns: number;
  activeCampaigns: number;
  totalSpend: number;
  totalRevenue: number;
  avgROAS: number;
  safetyEvents: number;
}

interface MetricCardProps {
  title: string;
  value: string;
  change?: number;
  icon: React.ReactNode;
  color?: string;
}

function MetricCard({ title, value, change, icon, color = 'blue' }: MetricCardProps) {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-50',
    green: 'text-green-600 bg-green-50',
    red: 'text-red-600 bg-red-50',
    yellow: 'text-yellow-600 bg-yellow-50',
    purple: 'text-purple-600 bg-purple-50',
  };

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            {change !== undefined && (
              <p className={`text-sm ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {change >= 0 ? '+' : ''}{change.toFixed(1)}% from last period
              </p>
            )}
          </div>
          <div className={`p-3 rounded-full ${colorClasses[color as keyof typeof colorClasses]}`}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function OverviewDashboard() {
  const { selectedTimeRange, selectedAgents } = useDashboard();
  const [realtimeUpdates, setRealtimeUpdates] = useState<any[]>([]);

  // WebSocket connection for real-time updates
  useWebSocket({
    url: `${process.env.NODE_ENV === 'production' ? 'wss' : 'ws'}://${typeof window !== 'undefined' ? window.location.host : 'localhost:3000'}/api/ws`,
    onMessage: (message) => {
      setRealtimeUpdates(prev => [message, ...prev.slice(0, 9)]);
    },
  });

  // Fetch overview statistics
  const { data: stats } = useQuery<OverviewStats>({
    queryKey: ['overview-stats', selectedTimeRange],
    queryFn: async () => {
      const response = await fetch('/api/overview/stats?' + new URLSearchParams({
        startTime: selectedTimeRange.start.toISOString(),
        endTime: selectedTimeRange.end.toISOString(),
      }));
      return response.json();
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Fetch training metrics for selected agents
  const { data: trainingData } = useQuery<TrainingMetrics[]>({
    queryKey: ['training-metrics', selectedAgents, selectedTimeRange],
    queryFn: async () => {
      if (selectedAgents.length === 0) return [];
      
      const response = await fetch('/api/training/metrics?' + new URLSearchParams({
        agentIds: selectedAgents.join(','),
        startTime: selectedTimeRange.start.toISOString(),
        endTime: selectedTimeRange.end.toISOString(),
      }));
      return response.json();
    },
    enabled: selectedAgents.length > 0,
  });

  // Fetch campaign metrics
  const { data: campaignData } = useQuery<CampaignMetrics[]>({
    queryKey: ['campaign-metrics', selectedAgents, selectedTimeRange],
    queryFn: async () => {
      const response = await fetch('/api/campaigns/metrics?' + new URLSearchParams({
        ...(selectedAgents.length > 0 && { agentIds: selectedAgents.join(',') }),
        startTime: selectedTimeRange.start.toISOString(),
        endTime: selectedTimeRange.end.toISOString(),
      }));
      return response.json();
    },
  });

  // Fetch recent safety events
  const { data: safetyEvents } = useQuery<SafetyEvent[]>({
    queryKey: ['safety-events', selectedAgents],
    queryFn: async () => {
      const response = await fetch('/api/safety/events?' + new URLSearchParams({
        ...(selectedAgents.length > 0 && { agentIds: selectedAgents.join(',') }),
        limit: '10',
        resolved: 'false',
      }));
      return response.json();
    },
  });

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Active Agents"
          value={formatNumber(stats?.activeAgents || 0)}
          change={12.5}
          icon={<Activity size={24} />}
          color="blue"
        />
        <MetricCard
          title="Total Revenue"
          value={formatCurrency(stats?.totalRevenue || 0)}
          change={8.3}
          icon={<DollarSign size={24} />}
          color="green"
        />
        <MetricCard
          title="Average ROAS"
          value={`${formatNumber(stats?.avgROAS || 0, 2)}x`}
          change={-2.1}
          icon={<TrendingUp size={24} />}
          color="purple"
        />
        <MetricCard
          title="Safety Events"
          value={formatNumber(stats?.safetyEvents || 0)}
          icon={<AlertTriangle size={24} />}
          color="red"
        />
      </div>

      {/* Real-time Updates */}
      {realtimeUpdates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap size={20} />
              Real-time Updates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {realtimeUpdates.map((update, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="text-sm">
                    <span className="font-medium">{update.type}:</span> {update.data.message}
                  </div>
                  <span className="text-xs text-gray-500">
                    {formatTimeAgo(new Date(update.timestamp))}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Progress */}
        {trainingData && trainingData.length > 0 && (
          <TrainingChart
            data={trainingData}
            selectedMetrics={['reward', 'loss']}
            title="Training Progress"
          />
        )}

        {/* Campaign Performance */}
        {campaignData && campaignData.length > 0 && (
          <CampaignMetricsChart
            data={campaignData}
            selectedMetrics={['spend', 'revenue']}
            title="Campaign Performance"
            chartType="area"
          />
        )}
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Top Performing Agents */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target size={20} />
              Top Performing Agents
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map((rank) => (
                <div key={rank} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Badge variant="info" size="sm">#{rank}</Badge>
                    <div>
                      <div className="font-medium">Agent-{rank * 123}</div>
                      <div className="text-sm text-gray-500">v2.{rank}.0</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">{(5.2 - rank * 0.3).toFixed(1)}x</div>
                    <div className="text-sm text-gray-500">ROAS</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Safety Events */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle size={20} />
              Safety Events
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {safetyEvents?.slice(0, 5).map((event) => (
                <div key={event.id} className="flex items-start gap-3">
                  <Badge 
                    variant={
                      event.severity === 'critical' ? 'error' : 
                      event.severity === 'high' ? 'warning' : 'info'
                    }
                    size="sm"
                  >
                    {event.severity}
                  </Badge>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{event.eventType}</div>
                    <div className="text-xs text-gray-500">{event.description}</div>
                    <div className="text-xs text-gray-400">{formatTimeAgo(event.timestamp)}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 size={20} />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Training Pipeline</span>
                <Badge variant="success">Healthy</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Campaign Manager</span>
                <Badge variant="success">Healthy</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Safety Monitor</span>
                <Badge variant="warning">Degraded</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Data Pipeline</span>
                <Badge variant="success">Healthy</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">API Gateway</span>
                <Badge variant="success">Healthy</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}