'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Brain, Activity, Zap, Clock, TrendingUp, AlertTriangle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { TrainingChart } from '@/components/charts/TrainingChart';
import { ResourceUtilizationChart } from '@/components/charts/ResourceUtilizationChart';
import { useDashboard } from '@/hooks/useDashboard';
import { formatNumber, formatTimeAgo, formatCurrency } from '@/lib/utils';
import { Agent, TrainingMetrics, ResourceUtilization } from '@/types';

export function TrainingDashboard() {
  const { selectedTimeRange, selectedAgents, addSelectedAgent, removeSelectedAgent } = useDashboard();
  const [selectedMetrics, setSelectedMetrics] = useState(['reward', 'loss']);

  // Fetch all agents
  const { data: agents = [] } = useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: async () => {
      const response = await fetch('/api/agents');
      return response.json();
    },
  });

  // Fetch training metrics for selected agents
  const { data: trainingData = [] } = useQuery<TrainingMetrics[]>({
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

  // Fetch resource utilization
  const { data: resourceData = [] } = useQuery<ResourceUtilization[]>({
    queryKey: ['resource-utilization', selectedAgents, selectedTimeRange],
    queryFn: async () => {
      if (selectedAgents.length === 0) return [];
      
      const response = await fetch('/api/training/resources?' + new URLSearchParams({
        agentIds: selectedAgents.join(','),
        startTime: selectedTimeRange.start.toISOString(),
        endTime: selectedTimeRange.end.toISOString(),
      }));
      return response.json();
    },
    enabled: selectedAgents.length > 0,
  });

  const trainingAgents = agents.filter(agent => 
    agent.status === 'training' || agent.status === 'deployed'
  );

  const toggleAgent = (agentId: string) => {
    if (selectedAgents.includes(agentId)) {
      removeSelectedAgent(agentId);
    } else {
      addSelectedAgent(agentId);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training': return 'text-blue-600 bg-blue-50';
      case 'deployed': return 'text-green-600 bg-green-50';
      case 'paused': return 'text-yellow-600 bg-yellow-50';
      case 'failed': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="space-y-6">
      {/* Agent Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain size={20} />
            Active Training Agents ({trainingAgents.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {trainingAgents.map((agent) => (
              <div
                key={agent.id}
                className={`
                  border rounded-lg p-4 cursor-pointer transition-all
                  ${selectedAgents.includes(agent.id) 
                    ? 'border-primary-500 bg-primary-50' 
                    : 'border-gray-200 hover:border-gray-300'
                  }
                `}
                onClick={() => toggleAgent(agent.id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-900">{agent.name}</h3>
                  <Badge className={getStatusColor(agent.status)} size="sm">
                    {agent.status}
                  </Badge>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Version: {agent.version}</div>
                  <div>Episodes: {agent.metrics.training?.totalEpisodes || 0}</div>
                  <div>Reward: {formatNumber(agent.metrics.training?.currentReward || 0, 2)}</div>
                  <div className="text-xs text-gray-500">
                    Updated: {formatTimeAgo(agent.updatedAt)}
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {trainingAgents.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <Brain size={48} className="mx-auto mb-4 text-gray-300" />
              <p>No active training agents found</p>
            </div>
          )}
        </CardContent>
      </Card>

      {selectedAgents.length > 0 && (
        <>
          {/* Training Progress Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Metric Selection */}
            <Card>
              <CardHeader>
                <CardTitle>Training Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Select Metrics to Display
                    </label>
                    <div className="space-y-2">
                      {[
                        { key: 'reward', label: 'Reward' },
                        { key: 'loss', label: 'Loss' },
                        { key: 'policyEntropy', label: 'Policy Entropy' },
                        { key: 'valueFunction', label: 'Value Function' },
                      ].map(metric => (
                        <label key={metric.key} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedMetrics.includes(metric.key)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedMetrics([...selectedMetrics, metric.key]);
                              } else {
                                setSelectedMetrics(selectedMetrics.filter(m => m !== metric.key));
                              }
                            }}
                            className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                          />
                          <span className="ml-2 text-sm text-gray-700">{metric.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Training Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity size={20} />
                  Training Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-2xl font-bold text-blue-600">
                      {formatNumber(trainingData.length)}
                    </div>
                    <div className="text-sm text-gray-600">Total Episodes</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-600">
                      {trainingData.length > 0 
                        ? formatNumber(trainingData[trainingData.length - 1]?.reward || 0, 2)
                        : '0'
                      }
                    </div>
                    <div className="text-sm text-gray-600">Latest Reward</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-purple-600">
                      {trainingData.filter(d => d.phase === 'simulation').length}
                    </div>
                    <div className="text-sm text-gray-600">Simulation Phase</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-orange-600">
                      {trainingData.filter(d => d.phase === 'real_deployment').length}
                    </div>
                    <div className="text-sm text-gray-600">Real Deployment</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Training Progress Chart */}
          {trainingData.length > 0 && (
            <TrainingChart
              data={trainingData}
              selectedMetrics={selectedMetrics}
              height={500}
              title="Training Progress Over Time"
            />
          )}

          {/* Resource Utilization */}
          {resourceData.length > 0 && (
            <ResourceUtilizationChart
              data={resourceData}
              selectedMetrics={['cpuUsage', 'memoryUsage', 'budgetSpent']}
              title="Resource Utilization"
            />
          )}

          {/* Episode Progress Table */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock size={20} />
                Recent Episodes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Episode
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Reward
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Loss
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Phase
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Completion
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Time
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {trainingData.slice(-10).reverse().map((episode, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {episode.episode}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatNumber(episode.reward, 3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatNumber(episode.loss, 6)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <Badge 
                            variant={episode.phase === 'simulation' ? 'info' : 'success'}
                            size="sm"
                          >
                            {episode.phase}
                          </Badge>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatNumber(episode.episodeCompletion * 100, 1)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatTimeAgo(episode.timestamp)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {selectedAgents.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Brain size={64} className="mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Select Agents to View Training Data
            </h3>
            <p className="text-gray-600">
              Choose one or more agents from the list above to see their training progress, 
              resource utilization, and detailed metrics.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}