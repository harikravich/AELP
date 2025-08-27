'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Trophy, TrendingUp, Crown, Medal, Award } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useDashboard } from '@/hooks/useDashboard';
import { formatNumber, formatCurrency, formatPercentage, formatTimeAgo } from '@/lib/utils';
import { LeaderboardEntry, CreativePerformance } from '@/types';

const METRICS = [
  { key: 'roas', label: 'ROAS', icon: <TrendingUp size={16} /> },
  { key: 'revenue', label: 'Revenue', icon: <TrendingUp size={16} /> },
  { key: 'ctr', label: 'CTR', icon: <TrendingUp size={16} /> },
  { key: 'conversion_rate', label: 'Conversion Rate', icon: <TrendingUp size={16} /> },
  { key: 'efficiency', label: 'Cost Efficiency', icon: <TrendingUp size={16} /> },
];

export function LeaderboardDashboard() {
  const { selectedTimeRange } = useDashboard();
  const [selectedMetric, setSelectedMetric] = useState('roas');

  // Fetch leaderboard data
  const { data: leaderboardData = [] } = useQuery<LeaderboardEntry[]>({
    queryKey: ['leaderboard', selectedMetric, selectedTimeRange],
    queryFn: async () => {
      const response = await fetch(`/api/leaderboard?${new URLSearchParams({
        metric: selectedMetric,
        startTime: selectedTimeRange.start.toISOString(),
        endTime: selectedTimeRange.end.toISOString(),
      })}`);
      return response.json();
    },
  });

  // Fetch top creatives
  const { data: topCreatives = [] } = useQuery<CreativePerformance[]>({
    queryKey: ['top-creatives', selectedTimeRange],
    queryFn: async () => {
      const response = await fetch(`/api/creatives/top?${new URLSearchParams({
        startTime: selectedTimeRange.start.toISOString(),
        endTime: selectedTimeRange.end.toISOString(),
      })}`);
      return response.json();
    },
  });

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Crown size={20} className="text-yellow-500" />;
      case 2:
        return <Medal size={20} className="text-gray-400" />;
      case 3:
        return <Award size={20} className="text-amber-600" />;
      default:
        return <span className="text-gray-600 font-bold">#{rank}</span>;
    }
  };

  const formatMetricValue = (value: number, metric: string) => {
    switch (metric) {
      case 'roas':
        return `${formatNumber(value, 2)}x`;
      case 'revenue':
        return formatCurrency(value);
      case 'ctr':
      case 'conversion_rate':
        return formatPercentage(value);
      case 'efficiency':
        return formatNumber(value, 2);
      default:
        return formatNumber(value, 2);
    }
  };

  return (
    <div className="space-y-6">
      {/* Metric Selector */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Trophy size={20} />
            Select Ranking Metric
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {METRICS.map(metric => (
              <Button
                key={metric.key}
                variant={selectedMetric === metric.key ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setSelectedMetric(metric.key)}
                className="flex items-center gap-2"
              >
                {metric.icon}
                {metric.label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Main Leaderboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Top 3 Podium */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Trophy size={20} />
              Top Performers - {METRICS.find(m => m.key === selectedMetric)?.label}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {leaderboardData.length >= 3 && (
              <div className="flex items-end justify-center gap-4 mb-8">
                {/* Second Place */}
                <div className="text-center">
                  <div className="w-20 h-16 bg-gray-200 rounded-t-lg flex items-end justify-center pb-2">
                    <Medal size={24} className="text-gray-500" />
                  </div>
                  <div className="bg-gray-100 p-3 rounded-b-lg">
                    <div className="font-semibold text-sm">{leaderboardData[1]?.agentName}</div>
                    <div className="text-lg font-bold text-gray-600">
                      {formatMetricValue(leaderboardData[1]?.score, selectedMetric)}
                    </div>
                    <Badge variant="default" size="sm">#2</Badge>
                  </div>
                </div>

                {/* First Place */}
                <div className="text-center">
                  <div className="w-24 h-20 bg-yellow-200 rounded-t-lg flex items-end justify-center pb-2">
                    <Crown size={28} className="text-yellow-600" />
                  </div>
                  <div className="bg-yellow-100 p-4 rounded-b-lg">
                    <div className="font-semibold">{leaderboardData[0]?.agentName}</div>
                    <div className="text-xl font-bold text-yellow-700">
                      {formatMetricValue(leaderboardData[0]?.score, selectedMetric)}
                    </div>
                    <Badge variant="warning" size="sm">#1</Badge>
                  </div>
                </div>

                {/* Third Place */}
                <div className="text-center">
                  <div className="w-20 h-12 bg-amber-200 rounded-t-lg flex items-end justify-center pb-2">
                    <Award size={20} className="text-amber-700" />
                  </div>
                  <div className="bg-amber-100 p-3 rounded-b-lg">
                    <div className="font-semibold text-sm">{leaderboardData[2]?.agentName}</div>
                    <div className="text-lg font-bold text-amber-700">
                      {formatMetricValue(leaderboardData[2]?.score, selectedMetric)}
                    </div>
                    <Badge variant="warning" size="sm">#3</Badge>
                  </div>
                </div>
              </div>
            )}

            {/* Full Rankings */}
            <div className="space-y-2">
              {leaderboardData.map((entry) => (
                <div
                  key={entry.agentId}
                  className={`
                    flex items-center justify-between p-4 rounded-lg border
                    ${entry.rank <= 3 ? 'bg-gradient-to-r from-yellow-50 to-transparent border-yellow-200' : 'bg-white border-gray-200'}
                  `}
                >
                  <div className="flex items-center gap-4">
                    <div className="flex items-center justify-center w-8 h-8">
                      {getRankIcon(entry.rank)}
                    </div>
                    <div>
                      <div className="font-semibold text-gray-900">{entry.agentName}</div>
                      <div className="text-sm text-gray-500">
                        {entry.campaigns} campaigns • Updated {formatTimeAgo(entry.lastUpdated)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-gray-900">
                      {formatMetricValue(entry.score, selectedMetric)}
                    </div>
                    <div className="text-sm text-gray-500">
                      {METRICS.find(m => m.key === selectedMetric)?.label}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Top Creatives */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award size={20} />
              Top Performing Creatives
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {topCreatives.slice(0, 5).map((creative, index) => (
                <div key={creative.creativeId} className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="info" size="sm">#{index + 1}</Badge>
                      <span className="font-medium text-sm">{creative.name}</span>
                    </div>
                    <Badge variant={
                      creative.type === 'video' ? 'info' :
                      creative.type === 'image' ? 'success' : 'default'
                    } size="sm">
                      {creative.type}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <div className="text-gray-500">CTR</div>
                      <div className="font-semibold">{formatPercentage(creative.avgCTR)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Conv. Rate</div>
                      <div className="font-semibold">{formatPercentage(creative.avgConversionRate)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">ROAS</div>
                      <div className="font-semibold">{formatNumber(creative.avgROAS, 2)}x</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Impressions</div>
                      <div className="font-semibold">{formatNumber(creative.impressions)}</div>
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    Platforms: {creative.platforms.join(', ')}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Statistics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6 text-center">
            <Trophy size={32} className="mx-auto mb-2 text-yellow-500" />
            <div className="text-2xl font-bold text-gray-900">{leaderboardData.length}</div>
            <div className="text-sm text-gray-600">Total Competing Agents</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <TrendingUp size={32} className="mx-auto mb-2 text-green-500" />
            <div className="text-2xl font-bold text-gray-900">
              {leaderboardData.length > 0 ? 
                formatMetricValue(leaderboardData[0]?.score || 0, selectedMetric) : 
                '0'
              }
            </div>
            <div className="text-sm text-gray-600">Best Performance</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <Award size={32} className="mx-auto mb-2 text-blue-500" />
            <div className="text-2xl font-bold text-gray-900">
              {leaderboardData.reduce((sum, entry) => sum + entry.campaigns, 0)}
            </div>
            <div className="text-sm text-gray-600">Total Campaigns</div>
          </CardContent>
        </Card>
      </div>

      {leaderboardData.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Trophy size={64} className="mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No Leaderboard Data Available
            </h3>
            <p className="text-gray-600">
              Agent rankings will appear here once campaigns start generating performance data.
              Try adjusting the time range to see historical rankings.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}