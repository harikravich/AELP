'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Target, 
  DollarSign, 
  TrendingUp, 
  Eye, 
  MousePointer, 
  ShoppingCart,
  BarChart3,
  Filter
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { CampaignMetricsChart } from '@/components/charts/CampaignMetricsChart';
import { useDashboard } from '@/hooks/useDashboard';
import { 
  formatCurrency, 
  formatNumber, 
  formatPercentage, 
  formatTimeAgo,
  calculateGrowth 
} from '@/lib/utils';
import { CampaignMetrics, ABTestResult } from '@/types';

interface CampaignSummary {
  totalSpend: number;
  totalRevenue: number;
  totalImpressions: number;
  totalClicks: number;
  totalConversions: number;
  avgCTR: number;
  avgConversionRate: number;
  avgROAS: number;
  avgCPC: number;
  avgCPA: number;
}

export function CampaignsDashboard() {
  const { selectedTimeRange, selectedAgents, filters } = useDashboard();
  const [selectedPlatforms, setSelectedPlatforms] = useState<string[]>([]);
  const [chartView, setChartView] = useState<'spend' | 'performance' | 'platform'>('performance');

  // Fetch campaign metrics
  const { data: campaignData = [] } = useQuery<CampaignMetrics[]>({
    queryKey: ['campaign-metrics', selectedAgents, selectedTimeRange, selectedPlatforms],
    queryFn: async () => {
      const params = new URLSearchParams({
        startTime: selectedTimeRange.start.toISOString(),
        endTime: selectedTimeRange.end.toISOString(),
      });
      
      if (selectedAgents.length > 0) {
        params.set('agentIds', selectedAgents.join(','));
      }
      
      if (selectedPlatforms.length > 0) {
        params.set('platforms', selectedPlatforms.join(','));
      }

      const response = await fetch(`/api/campaigns/metrics?${params}`);
      return response.json();
    },
  });

  // Fetch A/B test results
  const { data: abTestData = [] } = useQuery<ABTestResult[]>({
    queryKey: ['ab-test-results'],
    queryFn: async () => {
      const response = await fetch('/api/campaigns/ab-tests');
      return response.json();
    },
  });

  // Calculate summary metrics
  const summary: CampaignSummary = {
    totalSpend: campaignData.reduce((sum, d) => sum + d.spend, 0),
    totalRevenue: campaignData.reduce((sum, d) => sum + d.revenue, 0),
    totalImpressions: campaignData.reduce((sum, d) => sum + d.impressions, 0),
    totalClicks: campaignData.reduce((sum, d) => sum + d.clicks, 0),
    totalConversions: campaignData.reduce((sum, d) => sum + d.conversions, 0),
    avgCTR: campaignData.length > 0 ? campaignData.reduce((sum, d) => sum + d.ctr, 0) / campaignData.length : 0,
    avgConversionRate: campaignData.length > 0 ? campaignData.reduce((sum, d) => sum + d.conversionRate, 0) / campaignData.length : 0,
    avgROAS: campaignData.length > 0 ? campaignData.reduce((sum, d) => sum + d.roas, 0) / campaignData.length : 0,
    avgCPC: campaignData.length > 0 ? campaignData.reduce((sum, d) => sum + d.cpc, 0) / campaignData.length : 0,
    avgCPA: campaignData.length > 0 ? campaignData.reduce((sum, d) => sum + d.cpa, 0) / campaignData.length : 0,
  };

  // Get unique platforms
  const availablePlatforms = [...new Set(campaignData.map(d => d.platform))];

  const togglePlatform = (platform: string) => {
    setSelectedPlatforms(prev => 
      prev.includes(platform) 
        ? prev.filter(p => p !== platform)
        : [...prev, platform]
    );
  };

  const getChartData = () => {
    switch (chartView) {
      case 'spend':
        return { data: campaignData, metrics: ['spend', 'revenue'] };
      case 'performance':
        return { data: campaignData, metrics: ['ctr', 'conversionRate', 'roas'] };
      case 'platform':
        return { data: campaignData, metrics: ['spend', 'revenue'], groupBy: 'platform' as const };
      default:
        return { data: campaignData, metrics: ['spend', 'revenue'] };
    }
  };

  return (
    <div className="space-y-6">
      {/* Platform Filter */}
      {availablePlatforms.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Filter size={20} />
              Platform Filter
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {availablePlatforms.map(platform => (
                <Button
                  key={platform}
                  variant={selectedPlatforms.includes(platform) ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => togglePlatform(platform)}
                >
                  {platform}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Spend</p>
                <p className="text-2xl font-bold text-red-600">
                  {formatCurrency(summary.totalSpend)}
                </p>
                <p className="text-sm text-gray-500">
                  Avg CPC: {formatCurrency(summary.avgCPC)}
                </p>
              </div>
              <div className="p-3 rounded-full bg-red-50 text-red-600">
                <DollarSign size={24} />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Revenue</p>
                <p className="text-2xl font-bold text-green-600">
                  {formatCurrency(summary.totalRevenue)}
                </p>
                <p className="text-sm text-gray-500">
                  ROAS: {formatNumber(summary.avgROAS, 2)}x
                </p>
              </div>
              <div className="p-3 rounded-full bg-green-50 text-green-600">
                <TrendingUp size={24} />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Impressions</p>
                <p className="text-2xl font-bold text-blue-600">
                  {formatNumber(summary.totalImpressions)}
                </p>
                <p className="text-sm text-gray-500">
                  CTR: {formatPercentage(summary.avgCTR)}
                </p>
              </div>
              <div className="p-3 rounded-full bg-blue-50 text-blue-600">
                <Eye size={24} />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Conversions</p>
                <p className="text-2xl font-bold text-purple-600">
                  {formatNumber(summary.totalConversions)}
                </p>
                <p className="text-sm text-gray-500">
                  Rate: {formatPercentage(summary.avgConversionRate)}
                </p>
              </div>
              <div className="p-3 rounded-full bg-purple-50 text-purple-600">
                <ShoppingCart size={24} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chart View Selector */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 size={20} />
              Campaign Performance Charts
            </CardTitle>
            <div className="flex gap-2">
              <Button
                variant={chartView === 'spend' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setChartView('spend')}
              >
                Spend & Revenue
              </Button>
              <Button
                variant={chartView === 'performance' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setChartView('performance')}
              >
                Performance
              </Button>
              <Button
                variant={chartView === 'platform' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setChartView('platform')}
              >
                By Platform
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Charts */}
      {campaignData.length > 0 && (
        <CampaignMetricsChart
          {...getChartData()}
          height={500}
          title={`Campaign ${chartView === 'spend' ? 'Spend & Revenue' : 
                           chartView === 'performance' ? 'Performance Metrics' : 
                           'Performance by Platform'}`}
          chartType={chartView === 'platform' ? 'line' : 'area'}
        />
      )}

      {/* Campaign Performance Table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Campaigns */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target size={20} />
              Top Performing Campaigns
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {campaignData
                .slice()
                .sort((a, b) => b.roas - a.roas)
                .slice(0, 5)
                .map((campaign, index) => (
                  <div key={campaign.campaignId} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <Badge variant="info" size="sm">#{index + 1}</Badge>
                      <div>
                        <div className="font-medium text-sm">{campaign.campaignId}</div>
                        <div className="text-xs text-gray-500">{campaign.platform}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-green-600">
                        {formatNumber(campaign.roas, 2)}x
                      </div>
                      <div className="text-xs text-gray-500">ROAS</div>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>

        {/* A/B Test Results */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 size={20} />
              Recent A/B Test Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {abTestData.slice(0, 5).map((test) => (
                <div key={test.testId} className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium text-sm">{test.testId}</div>
                    <Badge 
                      variant={test.significance ? 'success' : 'warning'} 
                      size="sm"
                    >
                      {test.significance ? 'Significant' : 'Not Significant'}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div>
                      <div className="text-gray-500">Variant A</div>
                      <div className="font-medium">{formatNumber(test.variantAValue, 2)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Variant B</div>
                      <div className="font-medium">{formatNumber(test.variantBValue, 2)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">P-Value</div>
                      <div className="font-medium">{test.pValue.toFixed(4)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Sample Size</div>
                      <div className="font-medium">{formatNumber(test.sampleSize)}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {campaignData.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Target size={64} className="mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No Campaign Data Available
            </h3>
            <p className="text-gray-600">
              Campaign metrics will appear here once your agents start running campaigns.
              Adjust the time range or filters to see historical data.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}