import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { createBigQueryClient } from '../../../../lib/bigquery-client'
import { cookies } from 'next/headers'
import { DATASET_COOKIE, SANDBOX_DATASET, PROD_DATASET } from '../../../../lib/dataset'

interface ScenarioParams {
  type: 'budget' | 'bidding' | 'audience' | 'creative' | 'channel'
  baselineMetrics?: any
  changes: {
    budgetChange?: number // percentage change
    biddingStrategy?: string
    audienceSegments?: string[]
    creativeVariants?: number
    channelMix?: Record<string, number>
  }
  timeHorizon: number // days
}

export async function POST(req: NextRequest) {
  try {
    const body: ScenarioParams = await req.json()
    const { type, changes, timeHorizon = 30 } = body

    const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
    const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
    const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
    const bq = createBigQueryClient(projectId)

    // Fetch baseline metrics
    const [baseline] = await bq.query({
      query: `
        SELECT
          AVG(conversions) as avg_conversions,
          AVG(revenue) as avg_revenue,
          AVG(cost) as avg_cost,
          AVG(SAFE_DIVIDE(revenue, NULLIF(cost, 0))) as avg_roas,
          AVG(SAFE_DIVIDE(cost, NULLIF(conversions, 0))) as avg_cac,
          STDDEV(conversions) as std_conversions,
          STDDEV(revenue) as std_revenue
        FROM \`${projectId}.${dataset}.ads_kpi_daily\`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
      `
    })

    const baselineData = baseline[0] || {}

    // Model the scenario based on type
    let projection: any = {}

    switch (type) {
      case 'budget':
        projection = await modelBudgetScenario(baselineData, changes, timeHorizon)
        break

      case 'bidding':
        projection = await modelBiddingScenario(baselineData, changes, timeHorizon, bq, projectId, dataset)
        break

      case 'audience':
        projection = await modelAudienceScenario(baselineData, changes, timeHorizon, bq, projectId, dataset)
        break

      case 'creative':
        projection = await modelCreativeScenario(baselineData, changes, timeHorizon)
        break

      case 'channel':
        projection = await modelChannelMixScenario(baselineData, changes, timeHorizon, bq, projectId, dataset)
        break

      default:
        throw new Error(`Unknown scenario type: ${type}`)
    }

    // Calculate confidence intervals
    const confidence = calculateConfidenceIntervals(projection, baselineData)

    // Store scenario result for tracking
    await storeScenarioResult(bq, projectId, dataset, {
      type,
      changes,
      timeHorizon,
      baseline: baselineData,
      projection,
      confidence
    })

    return NextResponse.json({
      success: true,
      scenario: {
        type,
        changes,
        timeHorizon,
        baseline: baselineData,
        projection,
        confidence,
        recommendations: generateRecommendations(projection, baselineData)
      }
    })
  } catch (error: any) {
    console.error('Scenario modeling error:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to model scenario' },
      { status: 500 }
    )
  }
}

async function modelBudgetScenario(baseline: any, changes: any, horizon: number) {
  const budgetMultiplier = 1 + (changes.budgetChange || 0) / 100

  // Elasticity modeling - diminishing returns at scale
  const elasticity = 0.7 // Less than 1 indicates diminishing returns
  const effectiveMultiplier = Math.pow(budgetMultiplier, elasticity)

  return {
    projected_cost: baseline.avg_cost * budgetMultiplier * horizon,
    projected_conversions: baseline.avg_conversions * effectiveMultiplier * horizon,
    projected_revenue: baseline.avg_revenue * effectiveMultiplier * horizon,
    projected_roas: baseline.avg_roas * Math.pow(effectiveMultiplier / budgetMultiplier, 0.5),
    projected_cac: baseline.avg_cac * (budgetMultiplier / effectiveMultiplier),
    efficiency_score: effectiveMultiplier / budgetMultiplier,
    breakeven_point: baseline.avg_cost * (1 / baseline.avg_roas) * budgetMultiplier
  }
}

async function modelBiddingScenario(baseline: any, changes: any, horizon: number, bq: any, projectId: string, dataset: string) {
  // Fetch historical performance by bidding strategy
  const [strategyData] = await bq.query({
    query: `
      SELECT
        bidding_strategy,
        AVG(SAFE_DIVIDE(conversions, clicks)) as avg_cvr,
        AVG(SAFE_DIVIDE(clicks, impressions)) as avg_ctr,
        AVG(cost_per_conversion) as avg_cpc
      FROM \`${projectId}.${dataset}.ads_campaign_performance\`
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
        AND bidding_strategy IS NOT NULL
      GROUP BY bidding_strategy
    `
  })

  const newStrategy = changes.biddingStrategy || 'MAXIMIZE_CONVERSIONS'
  const strategyPerf = strategyData.find((s: any) => s.bidding_strategy === newStrategy) || {}

  // Calculate expected impact
  const cvrLift = (strategyPerf.avg_cvr || baseline.avg_conversions / 1000) / (baseline.avg_conversions / 1000) - 1
  const costLift = (strategyPerf.avg_cpc || baseline.avg_cac) / baseline.avg_cac - 1

  return {
    projected_conversions: baseline.avg_conversions * (1 + cvrLift) * horizon,
    projected_cost: baseline.avg_cost * (1 + costLift * 0.8) * horizon, // Dampened cost increase
    projected_revenue: baseline.avg_revenue * (1 + cvrLift) * horizon,
    projected_roas: baseline.avg_roas * (1 + cvrLift) / (1 + costLift * 0.8),
    strategy_impact: {
      conversion_lift: `${(cvrLift * 100).toFixed(1)}%`,
      cost_change: `${(costLift * 100).toFixed(1)}%`,
      efficiency_change: `${((1 + cvrLift) / (1 + costLift) - 1) * 100}%`
    }
  }
}

async function modelAudienceScenario(baseline: any, changes: any, horizon: number, bq: any, projectId: string, dataset: string) {
  const segments = changes.audienceSegments || []

  if (segments.length === 0) {
    return {
      projected_conversions: baseline.avg_conversions * horizon,
      projected_revenue: baseline.avg_revenue * horizon,
      message: 'No audience changes specified'
    }
  }

  // Fetch segment performance
  const [segmentData] = await bq.query({
    query: `
      SELECT
        segment,
        AVG(score) as avg_score,
        AVG(conversion_rate) as avg_cvr,
        AVG(revenue_per_user) as avg_rpu
      FROM \`${projectId}.${dataset}.segment_scores_daily\`
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
        AND segment IN UNNEST(@segments)
      GROUP BY segment
    `,
    params: { segments }
  })

  // Calculate weighted performance
  const segmentLift = segmentData.reduce((acc: number, seg: any) => {
    return acc + (seg.avg_score || 0) * (seg.avg_cvr || 0)
  }, 0) / Math.max(1, segmentData.length)

  const audienceMultiplier = 1 + segmentLift * 0.5 // Conservative estimate

  return {
    projected_conversions: baseline.avg_conversions * audienceMultiplier * horizon,
    projected_revenue: baseline.avg_revenue * audienceMultiplier * 1.2 * horizon, // Higher value users
    projected_cac: baseline.avg_cac * 0.9, // Better targeting reduces CAC
    audience_quality_score: segmentLift,
    targeted_segments: segments,
    segment_performance: segmentData
  }
}

async function modelCreativeScenario(baseline: any, changes: any, horizon: number) {
  const variantCount = changes.creativeVariants || 1

  // Model creative fatigue and testing impact
  const testingLift = Math.log(variantCount + 1) / Math.log(2) * 0.15 // Logarithmic improvement
  const fatigueReduction = 1 - Math.exp(-variantCount / 10) // Reduced fatigue with more variants

  const performanceMultiplier = 1 + testingLift + fatigueReduction * 0.1

  return {
    projected_conversions: baseline.avg_conversions * performanceMultiplier * horizon,
    projected_revenue: baseline.avg_revenue * performanceMultiplier * horizon,
    projected_ctr_improvement: `${(testingLift * 100).toFixed(1)}%`,
    fatigue_mitigation: `${(fatigueReduction * 100).toFixed(1)}%`,
    optimal_variants: Math.min(variantCount, 5), // Diminishing returns after 5
    testing_velocity: variantCount * 7, // Days to statistical significance
    expected_winner_lift: testingLift * 2 // Best performer expectation
  }
}

async function modelChannelMixScenario(baseline: any, changes: any, horizon: number, bq: any, projectId: string, dataset: string) {
  const channelMix = changes.channelMix || {}

  // Fetch channel performance
  const [channelData] = await bq.query({
    query: `
      SELECT
        traffic_source.source as channel,
        COUNT(DISTINCT user_pseudo_id) as users,
        SUM(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) as conversions,
        AVG(CASE WHEN event_name = 'purchase' THEN event_value_in_usd ELSE 0 END) as avg_order_value
      FROM \`${projectId}.${dataset}.events_*\`
      WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY))
      GROUP BY channel
    `
  })

  // Calculate weighted performance based on new mix
  let totalWeight = 0
  let weightedConversions = 0
  let weightedRevenue = 0

  for (const [channel, weight] of Object.entries(channelMix)) {
    const channelPerf = channelData.find((c: any) => c.channel === channel) || {
      conversions: baseline.avg_conversions * 0.8,
      avg_order_value: baseline.avg_revenue / baseline.avg_conversions
    }

    totalWeight += weight as number
    weightedConversions += (channelPerf.conversions || 0) * (weight as number)
    weightedRevenue += (channelPerf.avg_order_value || 0) * (channelPerf.conversions || 0) * (weight as number)
  }

  const mixEfficiency = totalWeight > 0 ? weightedConversions / totalWeight : 1

  return {
    projected_conversions: mixEfficiency * horizon,
    projected_revenue: (weightedRevenue / Math.max(1, totalWeight)) * horizon,
    channel_allocation: channelMix,
    channel_performance: channelData,
    diversification_score: calculateDiversificationScore(channelMix),
    risk_adjusted_return: mixEfficiency * (1 - calculateConcentrationRisk(channelMix))
  }
}

function calculateConfidenceIntervals(projection: any, baseline: any) {
  const confidence95 = 1.96 // Z-score for 95% confidence

  const intervals: any = {}

  for (const [key, value] of Object.entries(projection)) {
    if (typeof value === 'number' && key.startsWith('projected_')) {
      const baseKey = key.replace('projected_', 'avg_')
      const stdKey = key.replace('projected_', 'std_')

      const baseValue = baseline[baseKey] || value
      const stdValue = baseline[stdKey] || baseValue * 0.1 // 10% std if not available

      intervals[key] = {
        value,
        lower: Math.max(0, value - confidence95 * stdValue),
        upper: value + confidence95 * stdValue,
        confidence: 0.95
      }
    }
  }

  return intervals
}

function generateRecommendations(projection: any, baseline: any) {
  const recommendations = []

  // Check ROAS improvement
  if (projection.projected_roas && baseline.avg_roas) {
    const roasLift = (projection.projected_roas - baseline.avg_roas) / baseline.avg_roas
    if (roasLift > 0.1) {
      recommendations.push({
        type: 'positive',
        message: `Expected ROAS improvement of ${(roasLift * 100).toFixed(1)}%`,
        priority: 'high'
      })
    } else if (roasLift < -0.1) {
      recommendations.push({
        type: 'warning',
        message: `Potential ROAS decline of ${Math.abs(roasLift * 100).toFixed(1)}%`,
        priority: 'high'
      })
    }
  }

  // Check efficiency
  if (projection.efficiency_score && projection.efficiency_score < 0.7) {
    recommendations.push({
      type: 'warning',
      message: 'Diminishing returns expected at this scale',
      priority: 'medium'
    })
  }

  // Check diversification
  if (projection.diversification_score && projection.diversification_score < 0.5) {
    recommendations.push({
      type: 'warning',
      message: 'Consider diversifying channel mix to reduce risk',
      priority: 'medium'
    })
  }

  return recommendations
}

function calculateDiversificationScore(channelMix: Record<string, number>): number {
  const weights = Object.values(channelMix)
  const total = weights.reduce((sum, w) => sum + w, 0)

  if (total === 0) return 0

  // Calculate Herfindahl-Hirschman Index (HHI)
  const hhi = weights.reduce((sum, w) => sum + Math.pow(w / total, 2), 0)

  // Convert to diversification score (1 - HHI)
  return 1 - hhi
}

function calculateConcentrationRisk(channelMix: Record<string, number>): number {
  const weights = Object.values(channelMix)
  const total = weights.reduce((sum, w) => sum + w, 0)

  if (total === 0) return 1

  // Find max concentration
  const maxWeight = Math.max(...weights) / total

  // Risk increases exponentially with concentration
  return Math.pow(maxWeight, 2)
}

async function storeScenarioResult(bq: any, projectId: string, dataset: string, result: any) {
  try {
    const table = bq.dataset(dataset).table('scenario_modeling_results')

    await table.insert({
      scenario_id: `scenario_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      scenario_type: result.type,
      parameters: JSON.stringify(result.changes),
      time_horizon: result.timeHorizon,
      baseline_metrics: JSON.stringify(result.baseline),
      projections: JSON.stringify(result.projection),
      confidence_intervals: JSON.stringify(result.confidence),
      created_at: new Date().toISOString(),
      created_by: 'dashboard_api'
    })
  } catch (error) {
    console.error('Failed to store scenario result:', error)
  }
}