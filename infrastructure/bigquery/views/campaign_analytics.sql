-- =============================================================================
-- GAELP Campaign Analytics Views
-- =============================================================================

-- =============================================================================
-- CAMPAIGN PERFORMANCE SUMMARY VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `gaelp_campaigns.campaign_performance_summary` AS
WITH daily_metrics AS (
  SELECT 
    campaign_id,
    DATE(timestamp) as date,
    SUM(impressions) as daily_impressions,
    SUM(clicks) as daily_clicks,
    SUM(conversions) as daily_conversions,
    SUM(spend) as daily_spend,
    AVG(ctr) as avg_ctr,
    AVG(cpc) as avg_cpc,
    AVG(roas) as avg_roas
  FROM `gaelp_campaigns.performance_metrics`
  WHERE granularity = 'daily' 
    AND metric_type = 'actual'
  GROUP BY campaign_id, DATE(timestamp)
),
campaign_totals AS (
  SELECT 
    c.campaign_id,
    c.campaign_name,
    c.agent_id,
    c.campaign_type,
    c.status,
    c.platform,
    c.created_at,
    c.started_at,
    c.ended_at,
    
    -- Total metrics
    COALESCE(SUM(dm.daily_impressions), 0) as total_impressions,
    COALESCE(SUM(dm.daily_clicks), 0) as total_clicks,
    COALESCE(SUM(dm.daily_conversions), 0) as total_conversions,
    COALESCE(SUM(dm.daily_spend), 0) as total_spend,
    
    -- Average metrics
    AVG(dm.avg_ctr) as avg_ctr,
    AVG(dm.avg_cpc) as avg_cpc,
    AVG(dm.avg_roas) as avg_roas,
    
    -- Calculated metrics
    SAFE_DIVIDE(SUM(dm.daily_clicks), SUM(dm.daily_impressions)) as overall_ctr,
    SAFE_DIVIDE(SUM(dm.daily_spend), SUM(dm.daily_clicks)) as overall_cpc,
    SAFE_DIVIDE(SUM(dm.daily_conversions) * 100, SUM(dm.daily_spend)) as overall_roas,
    
    -- Time metrics
    DATE_DIFF(COALESCE(c.ended_at, CURRENT_TIMESTAMP()), c.started_at, DAY) as duration_days,
    COUNT(dm.date) as active_days
    
  FROM `gaelp_campaigns.campaigns` c
  LEFT JOIN daily_metrics dm ON c.campaign_id = dm.campaign_id
  GROUP BY 
    c.campaign_id, c.campaign_name, c.agent_id, c.campaign_type, 
    c.status, c.platform, c.created_at, c.started_at, c.ended_at
)
SELECT 
  *,
  CASE 
    WHEN total_spend > 0 THEN SAFE_DIVIDE(total_spend, active_days)
    ELSE 0 
  END as avg_daily_spend,
  CASE 
    WHEN total_impressions > 0 THEN SAFE_DIVIDE(total_impressions, active_days)
    ELSE 0 
  END as avg_daily_impressions,
  
  -- Performance grades
  CASE 
    WHEN overall_ctr >= 0.02 THEN 'A'
    WHEN overall_ctr >= 0.015 THEN 'B'
    WHEN overall_ctr >= 0.01 THEN 'C'
    WHEN overall_ctr >= 0.005 THEN 'D'
    ELSE 'F'
  END as ctr_grade,
  
  CASE 
    WHEN overall_roas >= 300 THEN 'A'
    WHEN overall_roas >= 200 THEN 'B'
    WHEN overall_roas >= 150 THEN 'C'
    WHEN overall_roas >= 100 THEN 'D'
    ELSE 'F'
  END as roas_grade

FROM campaign_totals;

-- =============================================================================
-- AGENT PERFORMANCE COMPARISON VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `gaelp_campaigns.agent_performance_comparison` AS
WITH agent_metrics AS (
  SELECT 
    c.agent_id,
    c.campaign_type,
    COUNT(DISTINCT c.campaign_id) as total_campaigns,
    COUNT(DISTINCT CASE WHEN c.status = 'completed' THEN c.campaign_id END) as completed_campaigns,
    
    -- Performance aggregations
    AVG(cps.total_impressions) as avg_impressions,
    AVG(cps.total_clicks) as avg_clicks,
    AVG(cps.total_conversions) as avg_conversions,
    AVG(cps.total_spend) as avg_spend,
    AVG(cps.overall_ctr) as avg_ctr,
    AVG(cps.overall_cpc) as avg_cpc,
    AVG(cps.overall_roas) as avg_roas,
    
    -- Best and worst performance
    MAX(cps.overall_roas) as best_roas,
    MIN(cps.overall_roas) as worst_roas,
    STDDEV(cps.overall_roas) as roas_std,
    
    -- Success metrics
    SAFE_DIVIDE(
      COUNT(DISTINCT CASE WHEN cps.overall_roas > 150 THEN c.campaign_id END),
      COUNT(DISTINCT c.campaign_id)
    ) as success_rate,
    
    -- Recent performance (last 30 days)
    AVG(CASE 
      WHEN c.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) 
      THEN cps.overall_roas 
    END) as recent_avg_roas
    
  FROM `gaelp_campaigns.campaigns` c
  JOIN `gaelp_campaigns.campaign_performance_summary` cps 
    ON c.campaign_id = cps.campaign_id
  GROUP BY c.agent_id, c.campaign_type
),
agent_rankings AS (
  SELECT 
    *,
    ROW_NUMBER() OVER (PARTITION BY campaign_type ORDER BY avg_roas DESC) as roas_rank,
    ROW_NUMBER() OVER (PARTITION BY campaign_type ORDER BY success_rate DESC) as success_rank,
    ROW_NUMBER() OVER (PARTITION BY campaign_type ORDER BY recent_avg_roas DESC) as recent_rank
  FROM agent_metrics
)
SELECT 
  *,
  CASE 
    WHEN roas_rank <= 3 THEN 'Top Performer'
    WHEN roas_rank <= 10 THEN 'Good Performer'
    WHEN roas_rank <= 20 THEN 'Average Performer'
    ELSE 'Needs Improvement'
  END as performance_tier
FROM agent_rankings;

-- =============================================================================
-- SIMULATION VS REAL PERFORMANCE VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `gaelp_campaigns.simulation_vs_real_performance` AS
WITH sim_performance AS (
  SELECT 
    c.agent_id,
    c.parent_campaign_id,
    cps.overall_ctr as sim_ctr,
    cps.overall_cpc as sim_cpc,
    cps.overall_roas as sim_roas,
    cps.total_impressions as sim_impressions,
    cps.total_spend as sim_spend
  FROM `gaelp_campaigns.campaigns` c
  JOIN `gaelp_campaigns.campaign_performance_summary` cps 
    ON c.campaign_id = cps.campaign_id
  WHERE c.campaign_type = 'simulation'
    AND c.parent_campaign_id IS NOT NULL
),
real_performance AS (
  SELECT 
    c.campaign_id,
    c.agent_id,
    cps.overall_ctr as real_ctr,
    cps.overall_cpc as real_cpc,
    cps.overall_roas as real_roas,
    cps.total_impressions as real_impressions,
    cps.total_spend as real_spend
  FROM `gaelp_campaigns.campaigns` c
  JOIN `gaelp_campaigns.campaign_performance_summary` cps 
    ON c.campaign_id = cps.campaign_id
  WHERE c.campaign_type = 'real'
)
SELECT 
  rp.campaign_id as real_campaign_id,
  rp.agent_id,
  
  -- Simulation metrics
  sp.sim_ctr,
  sp.sim_cpc,
  sp.sim_roas,
  sp.sim_impressions,
  sp.sim_spend,
  
  -- Real metrics
  rp.real_ctr,
  rp.real_cpc,
  rp.real_roas,
  rp.real_impressions,
  rp.real_spend,
  
  -- Prediction accuracy
  ABS(rp.real_ctr - sp.sim_ctr) / NULLIF(rp.real_ctr, 0) as ctr_prediction_error,
  ABS(rp.real_cpc - sp.sim_cpc) / NULLIF(rp.real_cpc, 0) as cpc_prediction_error,
  ABS(rp.real_roas - sp.sim_roas) / NULLIF(rp.real_roas, 0) as roas_prediction_error,
  
  -- Performance comparison
  (rp.real_roas - sp.sim_roas) as roas_difference,
  CASE 
    WHEN rp.real_roas > sp.sim_roas THEN 'Outperformed'
    WHEN rp.real_roas < sp.sim_roas THEN 'Underperformed'
    ELSE 'Met Expectations'
  END as performance_vs_simulation

FROM real_performance rp
JOIN sim_performance sp ON rp.campaign_id = sp.parent_campaign_id;

-- =============================================================================
-- SAFETY MONITORING DASHBOARD VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `gaelp_campaigns.safety_monitoring_dashboard` AS
WITH recent_events AS (
  SELECT 
    event_type,
    severity,
    category,
    COUNT(*) as event_count,
    COUNT(DISTINCT campaign_id) as affected_campaigns,
    AVG(financial_impact) as avg_financial_impact,
    SUM(financial_impact) as total_financial_impact
  FROM `gaelp_campaigns.safety_events`
  WHERE detected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  GROUP BY event_type, severity, category
),
trend_analysis AS (
  SELECT 
    DATE(detected_at) as event_date,
    event_type,
    severity,
    COUNT(*) as daily_events
  FROM `gaelp_campaigns.safety_events`
  WHERE detected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY DATE(detected_at), event_type, severity
),
unresolved_events AS (
  SELECT 
    event_type,
    severity,
    COUNT(*) as unresolved_count,
    MIN(detected_at) as oldest_unresolved,
    MAX(detected_at) as newest_unresolved
  FROM `gaelp_campaigns.safety_events`
  WHERE resolved_at IS NULL
  GROUP BY event_type, severity
)
SELECT 
  re.event_type,
  re.severity,
  re.category,
  re.event_count,
  re.affected_campaigns,
  re.avg_financial_impact,
  re.total_financial_impact,
  
  -- Unresolved events
  COALESCE(ue.unresolved_count, 0) as unresolved_count,
  ue.oldest_unresolved,
  
  -- Trend indicators
  CASE 
    WHEN re.event_count > 
      (SELECT AVG(daily_events) * 7 
       FROM trend_analysis ta 
       WHERE ta.event_type = re.event_type 
         AND ta.severity = re.severity)
    THEN 'Increasing'
    ELSE 'Normal'
  END as trend_indicator

FROM recent_events re
LEFT JOIN unresolved_events ue 
  ON re.event_type = ue.event_type 
  AND re.severity = ue.severity;

-- =============================================================================
-- LEARNING PROGRESS VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `gaelp_campaigns.learning_progress` AS
WITH episode_stats AS (
  SELECT 
    agent_id,
    campaign_id,
    episode_number,
    COUNT(*) as steps_per_episode,
    AVG(reward) as avg_reward,
    SUM(reward) as total_reward,
    MIN(timestamp) as episode_start,
    MAX(timestamp) as episode_end
  FROM `gaelp_campaigns.agent_episodes`
  GROUP BY agent_id, campaign_id, episode_number
),
learning_trends AS (
  SELECT 
    agent_id,
    episode_number,
    AVG(avg_reward) as avg_episode_reward,
    AVG(total_reward) as avg_total_reward,
    COUNT(DISTINCT campaign_id) as concurrent_campaigns,
    
    -- Moving averages for trend analysis
    AVG(avg_reward) OVER (
      PARTITION BY agent_id 
      ORDER BY episode_number 
      ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as reward_moving_avg_10,
    
    AVG(avg_reward) OVER (
      PARTITION BY agent_id 
      ORDER BY episode_number 
      ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
    ) as reward_moving_avg_50
    
  FROM episode_stats
  GROUP BY agent_id, episode_number
)
SELECT 
  agent_id,
  episode_number,
  avg_episode_reward,
  avg_total_reward,
  concurrent_campaigns,
  reward_moving_avg_10,
  reward_moving_avg_50,
  
  -- Learning progress indicators
  reward_moving_avg_10 - LAG(reward_moving_avg_10, 10) OVER (
    PARTITION BY agent_id ORDER BY episode_number
  ) as reward_improvement_10ep,
  
  CASE 
    WHEN reward_moving_avg_10 > reward_moving_avg_50 THEN 'Improving'
    WHEN reward_moving_avg_10 < reward_moving_avg_50 THEN 'Declining'
    ELSE 'Stable'
  END as learning_trend,
  
  -- Convergence indicators
  STDDEV(avg_episode_reward) OVER (
    PARTITION BY agent_id 
    ORDER BY episode_number 
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
  ) as reward_volatility_20ep

FROM learning_trends
WHERE episode_number >= 50; -- Only show after sufficient learning