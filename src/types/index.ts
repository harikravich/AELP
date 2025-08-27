// Core Agent Types
export interface Agent {
  id: string;
  name: string;
  version: string;
  status: 'training' | 'deployed' | 'paused' | 'failed';
  createdAt: Date;
  updatedAt: Date;
  config: AgentConfig;
  metrics: AgentMetrics;
}

export interface AgentConfig {
  learningRate: number;
  batchSize: number;
  architecture: string;
  hyperparameters: Record<string, any>;
  targetPlatforms: string[];
  budgetConstraints: BudgetConstraints;
}

export interface BudgetConstraints {
  maxDailySpend: number;
  maxCPC: number;
  maxCPM: number;
  totalBudget: number;
}

// Training Progress Types
export interface TrainingMetrics {
  episode: number;
  reward: number;
  loss: number;
  policyEntropy: number;
  valueFunction: number;
  timestamp: Date;
  phase: 'simulation' | 'real_deployment';
  episodeCompletion: number;
}

export interface ResourceUtilization {
  cpuUsage: number;
  memoryUsage: number;
  gpuUsage?: number;
  budgetSpent: number;
  computeCost: number;
  timestamp: Date;
}

// Campaign Performance Types
export interface CampaignMetrics {
  campaignId: string;
  agentId: string;
  impressions: number;
  clicks: number;
  conversions: number;
  spend: number;
  revenue: number;
  cpc: number;
  cpm: number;
  cpa: number;
  ctr: number;
  conversionRate: number;
  roas: number;
  timestamp: Date;
  platform: string;
}

export interface ABTestResult {
  testId: string;
  variantA: string;
  variantB: string;
  metricName: string;
  variantAValue: number;
  variantBValue: number;
  pValue: number;
  significance: boolean;
  confidenceInterval: [number, number];
  sampleSize: number;
  startDate: Date;
  endDate: Date;
}

// Agent Analytics Types
export interface AgentBehavior {
  agentId: string;
  actionDistribution: Record<string, number>;
  explorationRate: number;
  policyStability: number;
  decisionLatency: number;
  timestamp: Date;
}

export interface PolicyEvolution {
  checkpoint: string;
  policyParams: Record<string, number>;
  performance: number;
  timestamp: Date;
  modelSize: number;
}

export interface SafetyEvent {
  id: string;
  agentId: string;
  eventType: 'budget_exceeded' | 'performance_degradation' | 'policy_violation' | 'anomaly_detected';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: Date;
  resolved: boolean;
  resolvedAt?: Date;
}

// Simulation Types
export interface LLMPersona {
  id: string;
  name: string;
  demographics: Record<string, any>;
  interests: string[];
  behaviorPatterns: Record<string, any>;
  responseHistory: PersonaResponse[];
}

export interface PersonaResponse {
  adId: string;
  response: 'click' | 'ignore' | 'negative';
  confidence: number;
  reasoning: string;
  timestamp: Date;
}

export interface SimulationGap {
  metric: string;
  simulationValue: number;
  realWorldValue: number;
  gap: number;
  timestamp: Date;
}

// Leaderboard Types
export interface LeaderboardEntry {
  rank: number;
  agentId: string;
  agentName: string;
  score: number;
  metric: string;
  campaigns: number;
  lastUpdated: Date;
}

export interface CreativePerformance {
  creativeId: string;
  name: string;
  type: 'image' | 'video' | 'text';
  avgCTR: number;
  avgConversionRate: number;
  avgROAS: number;
  impressions: number;
  platforms: string[];
  lastUsed: Date;
}

// UI State Types
export interface DashboardState {
  selectedTimeRange: TimeRange;
  selectedAgents: string[];
  selectedCampaigns: string[];
  activeView: 'overview' | 'training' | 'campaigns' | 'analytics' | 'leaderboard';
  filters: DashboardFilters;
}

export interface TimeRange {
  start: Date;
  end: Date;
  preset?: '1h' | '24h' | '7d' | '30d' | 'custom';
}

export interface DashboardFilters {
  platforms: string[];
  agentStatus: string[];
  campaignStatus: string[];
  minBudget?: number;
  maxBudget?: number;
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  meta?: {
    total: number;
    page: number;
    limit: number;
  };
  error?: string;
}

export interface WebSocketMessage {
  type: 'training_update' | 'campaign_update' | 'safety_alert' | 'system_status';
  data: any;
  timestamp: Date;
}

// Agent Metrics Aggregated
export interface AgentMetrics {
  training: {
    totalEpisodes: number;
    currentReward: number;
    avgReward: number;
    convergenceProgress: number;
    lastCheckpoint: Date;
  };
  campaigns: {
    totalCampaigns: number;
    activeCampaigns: number;
    avgROAS: number;
    totalSpend: number;
    totalRevenue: number;
  };
  performance: {
    rank: number;
    scoreChange: number;
    efficiency: number;
    reliability: number;
  };
}

// User and Auth Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'researcher' | 'operator' | 'admin' | 'viewer';
  organization: string;
  permissions: string[];
  lastLogin: Date;
}

export interface Organization {
  id: string;
  name: string;
  subscription: 'free' | 'pro' | 'enterprise';
  quotas: {
    maxAgents: number;
    maxCampaigns: number;
    maxBudget: number;
  };
  usage: {
    agents: number;
    campaigns: number;
    budget: number;
  };
}