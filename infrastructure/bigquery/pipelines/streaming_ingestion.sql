-- =============================================================================
-- GAELP Streaming Data Ingestion Configuration
-- =============================================================================

-- =============================================================================
-- STREAMING TABLES FOR REAL-TIME INGESTION
-- =============================================================================

-- Real-time performance metrics streaming table
CREATE OR REPLACE TABLE `gaelp_campaigns.performance_metrics_stream` (
  -- Primary identifiers
  campaign_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  
  -- Streaming metadata
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  source_system STRING NOT NULL, -- 'meta_api', 'google_api', 'simulation_engine'
  batch_id STRING,
  sequence_number INT64,
  
  -- Core metrics (raw from source)
  raw_data JSON NOT NULL, -- Full API response
  
  -- Parsed metrics
  impressions INT64,
  clicks INT64,
  conversions FLOAT64,
  spend FLOAT64,
  
  -- Data quality flags
  is_backfill BOOL DEFAULT FALSE,
  is_estimated BOOL DEFAULT FALSE,
  confidence_score FLOAT64,
  quality_checks JSON
)
PARTITION BY DATE(timestamp)
CLUSTER BY campaign_id, source_system
OPTIONS (
  description = "Streaming table for real-time performance metrics ingestion",
  partition_expiration_days = 7 -- Keep streaming data for 7 days before moving to main table
);

-- Real-time agent actions streaming table
CREATE OR REPLACE TABLE `gaelp_campaigns.agent_actions_stream` (
  -- Primary identifiers
  action_id STRING NOT NULL,
  campaign_id STRING NOT NULL,
  agent_id STRING NOT NULL,
  
  -- Timing
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  
  -- Action details
  action_type STRING NOT NULL,
  action_payload JSON NOT NULL,
  
  -- Context
  state_before JSON,
  expected_outcome JSON,
  
  -- Execution tracking
  execution_status STRING DEFAULT 'pending', -- 'pending', 'executing', 'completed', 'failed'
  execution_details JSON,
  
  -- Metadata
  source_system STRING NOT NULL,
  trace_id STRING -- For distributed tracing
)
PARTITION BY DATE(timestamp)
CLUSTER BY campaign_id, agent_id, execution_status
OPTIONS (
  description = "Streaming table for real-time agent actions",
  partition_expiration_days = 7
);

-- =============================================================================
-- DATAFLOW ETL JOBS CONFIGURATION
-- =============================================================================

-- Create external table for Dataflow job metadata
CREATE OR REPLACE EXTERNAL TABLE `gaelp_campaigns.dataflow_job_configs`
OPTIONS (
  format = 'JSON',
  uris = ['gs://gaelp-config-bucket/dataflow/jobs/*.json']
);

-- =============================================================================
-- DATA VALIDATION FUNCTIONS
-- =============================================================================

-- Function to validate performance metrics
CREATE OR REPLACE FUNCTION `gaelp_campaigns.validate_performance_metrics`(
  impressions INT64,
  clicks INT64,
  conversions FLOAT64,
  spend FLOAT64
) RETURNS STRUCT<is_valid BOOL, issues ARRAY<STRING>>
LANGUAGE js AS """
  const issues = [];
  
  // Basic range validations
  if (impressions < 0) issues.push('Negative impressions');
  if (clicks < 0) issues.push('Negative clicks');
  if (conversions < 0) issues.push('Negative conversions');
  if (spend < 0) issues.push('Negative spend');
  
  // Logical validations
  if (clicks > impressions) issues.push('Clicks exceed impressions');
  if (conversions > clicks) issues.push('Conversions exceed clicks');
  
  // CTR validation
  const ctr = impressions > 0 ? clicks / impressions : 0;
  if (ctr > 0.5) issues.push('Unusually high CTR');
  
  // CPC validation
  const cpc = clicks > 0 ? spend / clicks : 0;
  if (cpc > 100) issues.push('Unusually high CPC');
  
  return {
    is_valid: issues.length === 0,
    issues: issues
  };
""";

-- Function to calculate reward components
CREATE OR REPLACE FUNCTION `gaelp_campaigns.calculate_reward_components`(
  current_roas FLOAT64,
  previous_roas FLOAT64,
  spend_efficiency FLOAT64,
  safety_score FLOAT64
) RETURNS STRUCT<
  total_reward FLOAT64,
  roas_component FLOAT64,
  efficiency_component FLOAT64,
  safety_component FLOAT64,
  improvement_component FLOAT64
>
LANGUAGE js AS """
  // Normalize ROAS improvement (-1 to 1)
  const roas_improvement = previous_roas > 0 ? 
    Math.max(-1, Math.min(1, (current_roas - previous_roas) / previous_roas)) : 0;
  
  // Calculate components
  const roas_component = Math.max(-1, Math.min(1, (current_roas - 150) / 150)); // 150% ROAS as baseline
  const efficiency_component = Math.max(-1, Math.min(1, (spend_efficiency - 0.5) / 0.5));
  const safety_component = Math.max(-1, Math.min(1, (safety_score - 0.5) / 0.5));
  const improvement_component = roas_improvement;
  
  // Weighted total reward
  const total_reward = (
    roas_component * 0.4 +
    efficiency_component * 0.2 +
    safety_component * 0.2 +
    improvement_component * 0.2
  );
  
  return {
    total_reward: total_reward,
    roas_component: roas_component,
    efficiency_component: efficiency_component,
    safety_component: safety_component,
    improvement_component: improvement_component
  };
""";

-- =============================================================================
-- STREAMING ETL PROCEDURES
-- =============================================================================

-- Procedure to process streaming performance metrics
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.process_streaming_metrics`()
BEGIN
  -- Validate and move data from streaming table to main table
  INSERT INTO `gaelp_campaigns.performance_metrics` (
    campaign_id,
    timestamp,
    granularity,
    metric_type,
    impressions,
    clicks,
    conversions,
    spend,
    ctr,
    cpc,
    cpm,
    cpa,
    roas,
    platform_metrics,
    data_source,
    is_estimated,
    quality_flags,
    created_at
  )
  SELECT 
    stream.campaign_id,
    stream.timestamp,
    'hourly' as granularity,
    CASE 
      WHEN stream.source_system LIKE '%simulation%' THEN 'simulated'
      ELSE 'actual'
    END as metric_type,
    stream.impressions,
    stream.clicks,
    stream.conversions,
    stream.spend,
    
    -- Calculate derived metrics
    SAFE_DIVIDE(stream.clicks, stream.impressions) as ctr,
    SAFE_DIVIDE(stream.spend, stream.clicks) as cpc,
    SAFE_DIVIDE(stream.spend * 1000, stream.impressions) as cpm,
    SAFE_DIVIDE(stream.spend, stream.conversions) as cpa,
    SAFE_DIVIDE(stream.conversions * 100, stream.spend) as roas,
    
    stream.raw_data as platform_metrics,
    stream.source_system as data_source,
    stream.is_estimated,
    
    -- Add validation results as quality flags
    CASE 
      WHEN validation.is_valid THEN []
      ELSE validation.issues
    END as quality_flags,
    
    CURRENT_TIMESTAMP()
    
  FROM `gaelp_campaigns.performance_metrics_stream` stream
  CROSS JOIN UNNEST([`gaelp_campaigns.validate_performance_metrics`(
    stream.impressions, 
    stream.clicks, 
    stream.conversions, 
    stream.spend
  )]) as validation
  WHERE stream.ingestion_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
    AND validation.is_valid = TRUE; -- Only insert valid data
  
  -- Log validation failures
  INSERT INTO `gaelp_campaigns.data_quality_log` (
    table_name,
    record_id,
    validation_issues,
    raw_data,
    created_at
  )
  SELECT 
    'performance_metrics_stream' as table_name,
    CONCAT(campaign_id, '_', CAST(timestamp AS STRING)) as record_id,
    validation.issues,
    TO_JSON_STRING(stream),
    CURRENT_TIMESTAMP()
  FROM `gaelp_campaigns.performance_metrics_stream` stream
  CROSS JOIN UNNEST([`gaelp_campaigns.validate_performance_metrics`(
    stream.impressions, 
    stream.clicks, 
    stream.conversions, 
    stream.spend
  )]) as validation
  WHERE stream.ingestion_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
    AND validation.is_valid = FALSE;
  
  -- Clean up processed streaming data
  DELETE FROM `gaelp_campaigns.performance_metrics_stream`
  WHERE ingestion_time < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR);
  
END;

-- Procedure to process agent episodes and calculate rewards
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.process_agent_episodes`()
BEGIN
  -- Process agent actions from stream and create episodes
  INSERT INTO `gaelp_campaigns.agent_episodes` (
    episode_id,
    campaign_id,
    agent_id,
    episode_number,
    step_number,
    state,
    action,
    reward,
    next_state,
    done,
    action_type,
    action_value,
    reward_components,
    immediate_reward,
    future_reward_estimate,
    policy_version,
    exploration_factor,
    confidence_score,
    environment_response,
    execution_status,
    timestamp,
    action_executed_at,
    reward_observed_at
  )
  WITH action_outcomes AS (
    SELECT 
      actions.action_id,
      actions.campaign_id,
      actions.agent_id,
      actions.timestamp,
      actions.action_type,
      actions.action_payload,
      actions.state_before,
      
      -- Get performance metrics after action
      metrics.impressions,
      metrics.clicks,
      metrics.conversions,
      metrics.spend,
      SAFE_DIVIDE(metrics.conversions * 100, metrics.spend) as current_roas,
      
      -- Get previous performance for comparison
      LAG(SAFE_DIVIDE(metrics.conversions * 100, metrics.spend)) OVER (
        PARTITION BY actions.campaign_id 
        ORDER BY actions.timestamp
      ) as previous_roas,
      
      -- Calculate efficiency metrics
      SAFE_DIVIDE(metrics.clicks, metrics.impressions) as spend_efficiency,
      
      -- Safety score (placeholder - would be calculated by safety system)
      0.8 as safety_score
      
    FROM `gaelp_campaigns.agent_actions_stream` actions
    LEFT JOIN `gaelp_campaigns.performance_metrics` metrics
      ON actions.campaign_id = metrics.campaign_id
      AND metrics.timestamp BETWEEN actions.timestamp 
        AND TIMESTAMP_ADD(actions.timestamp, INTERVAL 1 HOUR)
    WHERE actions.execution_status = 'completed'
      AND actions.ingestion_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
  ),
  episode_data AS (
    SELECT 
      GENERATE_UUID() as episode_id,
      ao.campaign_id,
      ao.agent_id,
      ROW_NUMBER() OVER (PARTITION BY ao.agent_id ORDER BY ao.timestamp) as episode_number,
      ROW_NUMBER() OVER (PARTITION BY ao.campaign_id ORDER BY ao.timestamp) as step_number,
      ao.state_before as state,
      ao.action_payload as action,
      ao.action_type,
      ao.action_payload as action_value,
      ao.timestamp,
      ao.timestamp as action_executed_at,
      TIMESTAMP_ADD(ao.timestamp, INTERVAL 1 HOUR) as reward_observed_at,
      
      -- Calculate reward using the function
      `gaelp_campaigns.calculate_reward_components`(
        ao.current_roas,
        COALESCE(ao.previous_roas, 150.0),
        ao.spend_efficiency,
        ao.safety_score
      ) as reward_calc,
      
      -- Determine if episode is done (simplified logic)
      CASE 
        WHEN ao.current_roas < 50 THEN TRUE  -- Very poor performance
        WHEN ao.spend > 1000 THEN TRUE       -- Budget exhausted
        ELSE FALSE
      END as done
      
    FROM action_outcomes ao
  )
  SELECT 
    episode_id,
    campaign_id,
    agent_id,
    episode_number,
    step_number,
    state,
    action,
    reward_calc.total_reward as reward,
    NULL as next_state, -- Will be updated in next iteration
    done,
    action_type,
    action_value,
    TO_JSON_STRING(reward_calc) as reward_components,
    reward_calc.total_reward as immediate_reward,
    reward_calc.total_reward * 0.9 as future_reward_estimate, -- Simple discount
    'v1.0' as policy_version,
    0.1 as exploration_factor,
    0.8 as confidence_score,
    NULL as environment_response,
    'success' as execution_status,
    timestamp,
    action_executed_at,
    reward_observed_at
  FROM episode_data;
  
  -- Clean up processed actions
  DELETE FROM `gaelp_campaigns.agent_actions_stream`
  WHERE execution_status = 'completed'
    AND ingestion_time < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR);
    
END;

-- =============================================================================
-- SCHEDULED JOBS CONFIGURATION
-- =============================================================================

-- Create data quality log table for tracking validation issues
CREATE OR REPLACE TABLE `gaelp_campaigns.data_quality_log` (
  log_id STRING DEFAULT GENERATE_UUID(),
  table_name STRING NOT NULL,
  record_id STRING,
  validation_issues ARRAY<STRING>,
  raw_data JSON,
  severity STRING DEFAULT 'warning',
  resolved BOOL DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  resolved_at TIMESTAMP,
  resolution_notes STRING
)
PARTITION BY DATE(created_at)
CLUSTER BY table_name, severity
OPTIONS (
  description = "Data quality monitoring and validation issues log",
  partition_expiration_days = 90
);