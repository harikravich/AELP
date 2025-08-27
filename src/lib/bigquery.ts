import { BigQuery } from '@google-cloud/bigquery';
import { 
  TrainingMetrics, 
  CampaignMetrics, 
  AgentBehavior, 
  SafetyEvent,
  Agent,
  ABTestResult,
  SimulationGap,
  ResourceUtilization
} from '@/types';

class BigQueryClient {
  private client: BigQuery;
  private dataset: string;

  constructor() {
    this.client = new BigQuery({
      projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
    });
    this.dataset = process.env.BIGQUERY_DATASET || 'gaelp_data';
  }

  async getAgents(filters?: {
    status?: string[];
    limit?: number;
    offset?: number;
  }): Promise<Agent[]> {
    let query = `
      SELECT *
      FROM \`${this.dataset}.agents\`
    `;
    
    const conditions = [];
    if (filters?.status?.length) {
      const statusList = filters.status.map(s => `'${s}'`).join(',');
      conditions.push(`status IN (${statusList})`);
    }
    
    if (conditions.length) {
      query += ` WHERE ${conditions.join(' AND ')}`;
    }
    
    query += ` ORDER BY updatedAt DESC`;
    
    if (filters?.limit) {
      query += ` LIMIT ${filters.limit}`;
    }
    
    if (filters?.offset) {
      query += ` OFFSET ${filters.offset}`;
    }

    const [rows] = await this.client.query(query);
    return rows.map(this.transformAgent);
  }

  async getTrainingMetrics(
    agentId: string,
    startTime: Date,
    endTime: Date
  ): Promise<TrainingMetrics[]> {
    const query = `
      SELECT *
      FROM \`${this.dataset}.training_metrics\`
      WHERE agent_id = @agentId
        AND timestamp BETWEEN @startTime AND @endTime
      ORDER BY timestamp ASC
    `;

    const [rows] = await this.client.query({
      query,
      params: {
        agentId,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
      },
    });

    return rows.map(this.transformTrainingMetrics);
  }

  async getCampaignMetrics(
    filters: {
      agentIds?: string[];
      campaignIds?: string[];
      platforms?: string[];
      startTime: Date;
      endTime: Date;
    }
  ): Promise<CampaignMetrics[]> {
    let query = `
      SELECT *
      FROM \`${this.dataset}.campaign_metrics\`
      WHERE timestamp BETWEEN @startTime AND @endTime
    `;
    
    const params: any = {
      startTime: filters.startTime.toISOString(),
      endTime: filters.endTime.toISOString(),
    };

    if (filters.agentIds?.length) {
      query += ` AND agent_id IN UNNEST(@agentIds)`;
      params.agentIds = filters.agentIds;
    }

    if (filters.campaignIds?.length) {
      query += ` AND campaign_id IN UNNEST(@campaignIds)`;
      params.campaignIds = filters.campaignIds;
    }

    if (filters.platforms?.length) {
      query += ` AND platform IN UNNEST(@platforms)`;
      params.platforms = filters.platforms;
    }

    query += ` ORDER BY timestamp DESC`;

    const [rows] = await this.client.query({ query, params });
    return rows.map(this.transformCampaignMetrics);
  }

  async getAgentBehavior(
    agentId: string,
    startTime: Date,
    endTime: Date
  ): Promise<AgentBehavior[]> {
    const query = `
      SELECT *
      FROM \`${this.dataset}.agent_behavior\`
      WHERE agent_id = @agentId
        AND timestamp BETWEEN @startTime AND @endTime
      ORDER BY timestamp DESC
    `;

    const [rows] = await this.client.query({
      query,
      params: {
        agentId,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
      },
    });

    return rows.map(this.transformAgentBehavior);
  }

  async getSafetyEvents(
    filters: {
      agentIds?: string[];
      severity?: string[];
      resolved?: boolean;
      limit?: number;
    }
  ): Promise<SafetyEvent[]> {
    let query = `
      SELECT *
      FROM \`${this.dataset}.safety_events\`
      WHERE 1=1
    `;
    
    const params: any = {};

    if (filters.agentIds?.length) {
      query += ` AND agent_id IN UNNEST(@agentIds)`;
      params.agentIds = filters.agentIds;
    }

    if (filters.severity?.length) {
      query += ` AND severity IN UNNEST(@severity)`;
      params.severity = filters.severity;
    }

    if (filters.resolved !== undefined) {
      query += ` AND resolved = @resolved`;
      params.resolved = filters.resolved;
    }

    query += ` ORDER BY timestamp DESC`;

    if (filters.limit) {
      query += ` LIMIT ${filters.limit}`;
    }

    const [rows] = await this.client.query({ query, params });
    return rows.map(this.transformSafetyEvent);
  }

  async getABTestResults(
    testIds?: string[]
  ): Promise<ABTestResult[]> {
    let query = `
      SELECT *
      FROM \`${this.dataset}.ab_test_results\`
    `;
    
    const params: any = {};

    if (testIds?.length) {
      query += ` WHERE test_id IN UNNEST(@testIds)`;
      params.testIds = testIds;
    }

    query += ` ORDER BY end_date DESC`;

    const [rows] = await this.client.query({ query, params });
    return rows.map(this.transformABTestResult);
  }

  async getResourceUtilization(
    agentId: string,
    startTime: Date,
    endTime: Date
  ): Promise<ResourceUtilization[]> {
    const query = `
      SELECT *
      FROM \`${this.dataset}.resource_utilization\`
      WHERE agent_id = @agentId
        AND timestamp BETWEEN @startTime AND @endTime
      ORDER BY timestamp ASC
    `;

    const [rows] = await this.client.query({
      query,
      params: {
        agentId,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
      },
    });

    return rows.map(this.transformResourceUtilization);
  }

  async getSimulationGaps(
    agentId: string,
    startTime: Date,
    endTime: Date
  ): Promise<SimulationGap[]> {
    const query = `
      SELECT *
      FROM \`${this.dataset}.simulation_gaps\`
      WHERE agent_id = @agentId
        AND timestamp BETWEEN @startTime AND @endTime
      ORDER BY timestamp DESC
    `;

    const [rows] = await this.client.query({
      query,
      params: {
        agentId,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
      },
    });

    return rows.map(this.transformSimulationGap);
  }

  async getLeaderboardData(
    metric: string,
    timeRange: { start: Date; end: Date },
    limit = 50
  ): Promise<any[]> {
    const query = `
      WITH agent_performance AS (
        SELECT 
          c.agent_id,
          a.name as agent_name,
          AVG(c.${metric}) as avg_metric,
          COUNT(DISTINCT c.campaign_id) as campaign_count,
          MAX(c.timestamp) as last_updated
        FROM \`${this.dataset}.campaign_metrics\` c
        JOIN \`${this.dataset}.agents\` a ON c.agent_id = a.id
        WHERE c.timestamp BETWEEN @startTime AND @endTime
        GROUP BY c.agent_id, a.name
      )
      SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY avg_metric DESC) as rank
      FROM agent_performance
      ORDER BY avg_metric DESC
      LIMIT ${limit}
    `;

    const [rows] = await this.client.query({
      query,
      params: {
        startTime: timeRange.start.toISOString(),
        endTime: timeRange.end.toISOString(),
      },
    });

    return rows;
  }

  // Transform functions
  private transformAgent(row: any): Agent {
    return {
      id: row.id,
      name: row.name,
      version: row.version,
      status: row.status,
      createdAt: new Date(row.created_at),
      updatedAt: new Date(row.updated_at),
      config: JSON.parse(row.config || '{}'),
      metrics: JSON.parse(row.metrics || '{}'),
    };
  }

  private transformTrainingMetrics(row: any): TrainingMetrics {
    return {
      episode: row.episode,
      reward: row.reward,
      loss: row.loss,
      policyEntropy: row.policy_entropy,
      valueFunction: row.value_function,
      timestamp: new Date(row.timestamp),
      phase: row.phase,
      episodeCompletion: row.episode_completion,
    };
  }

  private transformCampaignMetrics(row: any): CampaignMetrics {
    return {
      campaignId: row.campaign_id,
      agentId: row.agent_id,
      impressions: row.impressions,
      clicks: row.clicks,
      conversions: row.conversions,
      spend: row.spend,
      revenue: row.revenue,
      cpc: row.cpc,
      cpm: row.cpm,
      cpa: row.cpa,
      ctr: row.ctr,
      conversionRate: row.conversion_rate,
      roas: row.roas,
      timestamp: new Date(row.timestamp),
      platform: row.platform,
    };
  }

  private transformAgentBehavior(row: any): AgentBehavior {
    return {
      agentId: row.agent_id,
      actionDistribution: JSON.parse(row.action_distribution),
      explorationRate: row.exploration_rate,
      policyStability: row.policy_stability,
      decisionLatency: row.decision_latency,
      timestamp: new Date(row.timestamp),
    };
  }

  private transformSafetyEvent(row: any): SafetyEvent {
    return {
      id: row.id,
      agentId: row.agent_id,
      eventType: row.event_type,
      severity: row.severity,
      description: row.description,
      timestamp: new Date(row.timestamp),
      resolved: row.resolved,
      resolvedAt: row.resolved_at ? new Date(row.resolved_at) : undefined,
    };
  }

  private transformABTestResult(row: any): ABTestResult {
    return {
      testId: row.test_id,
      variantA: row.variant_a,
      variantB: row.variant_b,
      metricName: row.metric_name,
      variantAValue: row.variant_a_value,
      variantBValue: row.variant_b_value,
      pValue: row.p_value,
      significance: row.significance,
      confidenceInterval: [row.confidence_interval_lower, row.confidence_interval_upper],
      sampleSize: row.sample_size,
      startDate: new Date(row.start_date),
      endDate: new Date(row.end_date),
    };
  }

  private transformResourceUtilization(row: any): ResourceUtilization {
    return {
      cpuUsage: row.cpu_usage,
      memoryUsage: row.memory_usage,
      gpuUsage: row.gpu_usage,
      budgetSpent: row.budget_spent,
      computeCost: row.compute_cost,
      timestamp: new Date(row.timestamp),
    };
  }

  private transformSimulationGap(row: any): SimulationGap {
    return {
      metric: row.metric,
      simulationValue: row.simulation_value,
      realWorldValue: row.real_world_value,
      gap: row.gap,
      timestamp: new Date(row.timestamp),
    };
  }
}

export const bigQueryClient = new BigQueryClient();