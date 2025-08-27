-- =============================================================================
-- GAELP Data Export and Integration Functions
-- =============================================================================

-- =============================================================================
-- ML TRAINING DATA EXPORT FUNCTIONS
-- =============================================================================

-- Function to prepare training data for RL models
CREATE OR REPLACE TABLE FUNCTION `gaelp_campaigns.prepare_training_data`(
  agent_id STRING,
  start_date DATE,
  end_date DATE
)
RETURNS TABLE<
  state_vector ARRAY<FLOAT64>,
  action_vector ARRAY<FLOAT64>,
  reward FLOAT64,
  next_state_vector ARRAY<FLOAT64>,
  done BOOL,
  episode_id STRING,
  step_number INT64,
  metadata JSON
>
AS (
  WITH episode_sequences AS (
    SELECT 
      e.episode_id,
      e.campaign_id,
      e.agent_id,
      e.step_number,
      e.reward,
      e.done,
      e.timestamp,
      
      -- Parse state and action from JSON
      e.state,
      e.action,
      e.next_state,
      
      -- Get campaign context
      c.campaign_type,
      c.platform,
      JSON_EXTRACT_SCALAR(c.configuration, '$.budget.daily_limit') as daily_budget,
      JSON_EXTRACT_SCALAR(c.configuration, '$.targeting.age_range') as age_range,
      
      -- Get performance context from metrics
      pm.ctr,
      pm.cpc,
      pm.roas,
      pm.spend,
      pm.impressions,
      pm.clicks,
      pm.conversions
      
    FROM `gaelp_campaigns.agent_episodes` e
    JOIN `gaelp_campaigns.campaigns` c ON e.campaign_id = c.campaign_id
    LEFT JOIN `gaelp_campaigns.performance_metrics` pm 
      ON e.campaign_id = pm.campaign_id
      AND pm.timestamp BETWEEN e.timestamp AND TIMESTAMP_ADD(e.timestamp, INTERVAL 1 HOUR)
      AND pm.granularity = 'hourly'
    WHERE e.agent_id = agent_id
      AND DATE(e.timestamp) BETWEEN start_date AND end_date
  ),
  feature_vectors AS (
    SELECT 
      episode_id,
      step_number,
      reward,
      done,
      
      -- Create state vector (normalized features)
      [
        -- Performance metrics (normalized 0-1)
        COALESCE(ctr / 0.1, 0.0), -- Assume max CTR of 10%
        COALESCE(cpc / 10.0, 0.0), -- Assume max CPC of $10
        COALESCE(roas / 500.0, 0.0), -- Assume max ROAS of 500%
        COALESCE(spend / CAST(daily_budget AS FLOAT64), 0.0), -- Spend as % of budget
        
        -- Volume metrics (log-normalized)
        COALESCE(LOG(impressions + 1) / 20.0, 0.0), -- Log-scale impressions
        COALESCE(LOG(clicks + 1) / 15.0, 0.0), -- Log-scale clicks
        COALESCE(LOG(conversions + 1) / 10.0, 0.0), -- Log-scale conversions
        
        -- Campaign metadata (one-hot encoded)
        CASE WHEN campaign_type = 'simulation' THEN 1.0 ELSE 0.0 END,
        CASE WHEN campaign_type = 'real' THEN 1.0 ELSE 0.0 END,
        CASE WHEN platform = 'meta' THEN 1.0 ELSE 0.0 END,
        CASE WHEN platform = 'google' THEN 1.0 ELSE 0.0 END,
        
        -- Time features
        EXTRACT(HOUR FROM timestamp) / 24.0, -- Hour of day (0-1)
        EXTRACT(DAYOFWEEK FROM timestamp) / 7.0, -- Day of week (0-1)
        
        -- Targeting features (simplified)
        CASE WHEN age_range LIKE '%18-24%' THEN 1.0 ELSE 0.0 END,
        CASE WHEN age_range LIKE '%25-34%' THEN 1.0 ELSE 0.0 END,
        CASE WHEN age_range LIKE '%35-44%' THEN 1.0 ELSE 0.0 END
        
      ] as state_vector,
      
      -- Create action vector (one-hot + continuous)
      [
        -- Action type (one-hot)
        CASE WHEN JSON_EXTRACT_SCALAR(action, '$.type') = 'budget_adjust' THEN 1.0 ELSE 0.0 END,
        CASE WHEN JSON_EXTRACT_SCALAR(action, '$.type') = 'bid_adjust' THEN 1.0 ELSE 0.0 END,
        CASE WHEN JSON_EXTRACT_SCALAR(action, '$.type') = 'targeting_change' THEN 1.0 ELSE 0.0 END,
        CASE WHEN JSON_EXTRACT_SCALAR(action, '$.type') = 'creative_update' THEN 1.0 ELSE 0.0 END,
        
        -- Action magnitude (normalized)
        COALESCE(CAST(JSON_EXTRACT_SCALAR(action, '$.magnitude') AS FLOAT64) / 100.0, 0.0),
        
        -- Action direction
        CASE WHEN JSON_EXTRACT_SCALAR(action, '$.direction') = 'increase' THEN 1.0 ELSE -1.0 END
        
      ] as action_vector,
      
      -- Next state vector (same as state vector but from next step)
      LEAD([
        COALESCE(ctr / 0.1, 0.0),
        COALESCE(cpc / 10.0, 0.0),
        COALESCE(roas / 500.0, 0.0),
        COALESCE(spend / CAST(daily_budget AS FLOAT64), 0.0),
        COALESCE(LOG(impressions + 1) / 20.0, 0.0),
        COALESCE(LOG(clicks + 1) / 15.0, 0.0),
        COALESCE(LOG(conversions + 1) / 10.0, 0.0),
        CASE WHEN campaign_type = 'simulation' THEN 1.0 ELSE 0.0 END,
        CASE WHEN campaign_type = 'real' THEN 1.0 ELSE 0.0 END,
        CASE WHEN platform = 'meta' THEN 1.0 ELSE 0.0 END,
        CASE WHEN platform = 'google' THEN 1.0 ELSE 0.0 END,
        EXTRACT(HOUR FROM timestamp) / 24.0,
        EXTRACT(DAYOFWEEK FROM timestamp) / 7.0,
        CASE WHEN age_range LIKE '%18-24%' THEN 1.0 ELSE 0.0 END,
        CASE WHEN age_range LIKE '%25-34%' THEN 1.0 ELSE 0.0 END,
        CASE WHEN age_range LIKE '%35-44%' THEN 1.0 ELSE 0.0 END
      ]) OVER (
        PARTITION BY episode_id 
        ORDER BY step_number
      ) as next_state_vector,
      
      -- Metadata for debugging and analysis
      JSON_OBJECT(
        'campaign_id', campaign_id,
        'campaign_type', campaign_type,
        'platform', platform,
        'timestamp', timestamp,
        'raw_state', state,
        'raw_action', action
      ) as metadata
      
    FROM episode_sequences
  )
  SELECT 
    state_vector,
    action_vector,
    reward,
    next_state_vector,
    done,
    episode_id,
    step_number,
    metadata
  FROM feature_vectors
  WHERE state_vector IS NOT NULL
    AND action_vector IS NOT NULL
  ORDER BY episode_id, step_number
);

-- =============================================================================
-- BENCHMARK PORTAL EXPORT FUNCTIONS
-- =============================================================================

-- Function to generate benchmark report data
CREATE OR REPLACE TABLE FUNCTION `gaelp_campaigns.generate_benchmark_report`(
  start_date DATE,
  end_date DATE,
  campaign_types ARRAY<STRING>
)
RETURNS TABLE<
  agent_id STRING,
  campaign_type STRING,
  platform STRING,
  total_campaigns INT64,
  total_spend FLOAT64,
  total_conversions FLOAT64,
  avg_roas FLOAT64,
  avg_ctr FLOAT64,
  avg_cpc FLOAT64,
  success_rate FLOAT64,
  risk_score FLOAT64,
  performance_grade STRING,
  benchmark_percentile FLOAT64,
  trend_direction STRING,
  metadata JSON
>
AS (
  WITH campaign_performance AS (
    SELECT 
      c.agent_id,
      c.campaign_type,
      c.platform,
      c.campaign_id,
      
      -- Aggregate metrics from daily performance
      SUM(dp.total_impressions) as total_impressions,
      SUM(dp.total_clicks) as total_clicks,
      SUM(dp.total_conversions) as total_conversions,
      SUM(dp.total_spend) as total_spend,
      AVG(dp.daily_ctr) as avg_ctr,
      AVG(dp.daily_cpc) as avg_cpc,
      AVG(dp.daily_roas) as avg_roas,
      
      -- Calculate success metrics
      COUNT(DISTINCT dp.date) as active_days,
      COUNTIF(dp.daily_roas > 150) / COUNT(*) as success_rate,
      
      -- Risk indicators
      STDDEV(dp.daily_roas) as roas_volatility,
      COUNTIF(dp.daily_spend > 100) / COUNT(*) as high_spend_rate
      
    FROM `gaelp_campaigns.campaigns` c
    JOIN `gaelp_campaigns.daily_campaign_performance` dp 
      ON c.campaign_id = dp.campaign_id
    WHERE dp.date BETWEEN start_date AND end_date
      AND c.campaign_type IN UNNEST(campaign_types)
    GROUP BY c.agent_id, c.campaign_type, c.platform, c.campaign_id
  ),
  agent_aggregates AS (
    SELECT 
      agent_id,
      campaign_type,
      platform,
      COUNT(*) as total_campaigns,
      SUM(total_spend) as total_spend,
      SUM(total_conversions) as total_conversions,
      AVG(avg_roas) as avg_roas,
      AVG(avg_ctr) as avg_ctr,
      AVG(avg_cpc) as avg_cpc,
      AVG(success_rate) as success_rate,
      
      -- Risk score calculation
      (
        COALESCE(AVG(roas_volatility) / 100, 0) * 0.4 +
        COALESCE(AVG(high_spend_rate), 0) * 0.3 +
        COALESCE(1 - AVG(success_rate), 0) * 0.3
      ) as risk_score,
      
      -- Trend analysis (last 7 days vs previous 7 days)
      AVG(CASE 
        WHEN dp.date >= DATE_SUB(end_date, INTERVAL 7 DAY) 
        THEN cp.avg_roas 
      END) as recent_roas,
      AVG(CASE 
        WHEN dp.date BETWEEN DATE_SUB(end_date, INTERVAL 14 DAY) 
                          AND DATE_SUB(end_date, INTERVAL 8 DAY)
        THEN cp.avg_roas 
      END) as previous_roas
      
    FROM campaign_performance cp
    JOIN `gaelp_campaigns.daily_campaign_performance` dp 
      ON cp.campaign_id = dp.campaign_id
    GROUP BY agent_id, campaign_type, platform
  ),
  performance_rankings AS (
    SELECT 
      *,
      -- Performance grading
      CASE 
        WHEN avg_roas >= 300 THEN 'A+'
        WHEN avg_roas >= 250 THEN 'A'
        WHEN avg_roas >= 200 THEN 'B+'
        WHEN avg_roas >= 150 THEN 'B'
        WHEN avg_roas >= 100 THEN 'C'
        WHEN avg_roas >= 50 THEN 'D'
        ELSE 'F'
      END as performance_grade,
      
      -- Percentile ranking within campaign type
      PERCENT_RANK() OVER (
        PARTITION BY campaign_type 
        ORDER BY avg_roas
      ) * 100 as benchmark_percentile,
      
      -- Trend direction
      CASE 
        WHEN recent_roas > previous_roas * 1.1 THEN 'Strong Upward'
        WHEN recent_roas > previous_roas * 1.05 THEN 'Upward'
        WHEN recent_roas < previous_roas * 0.9 THEN 'Strong Downward'
        WHEN recent_roas < previous_roas * 0.95 THEN 'Downward'
        ELSE 'Stable'
      END as trend_direction
      
    FROM agent_aggregates
  )
  SELECT 
    agent_id,
    campaign_type,
    platform,
    total_campaigns,
    total_spend,
    total_conversions,
    avg_roas,
    avg_ctr,
    avg_cpc,
    success_rate,
    risk_score,
    performance_grade,
    benchmark_percentile,
    trend_direction,
    
    JSON_OBJECT(
      'recent_roas', recent_roas,
      'previous_roas', previous_roas,
      'data_period', JSON_OBJECT(
        'start_date', start_date,
        'end_date', end_date
      ),
      'generated_at', CURRENT_TIMESTAMP()
    ) as metadata
    
  FROM performance_rankings
  ORDER BY avg_roas DESC
);

-- =============================================================================
-- EXTERNAL SYSTEM INTEGRATION PROCEDURES
-- =============================================================================

-- Procedure to export training data to Cloud Storage
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.export_training_data_to_gcs`(
  IN agent_id STRING,
  IN start_date DATE,
  IN end_date DATE,
  IN gcs_bucket STRING
)
BEGIN
  DECLARE export_uri STRING;
  DECLARE file_count INT64;
  
  SET export_uri = CONCAT('gs://', gcs_bucket, '/training_data/', agent_id, '/', 
                         FORMAT_DATE('%Y%m%d', start_date), '_', 
                         FORMAT_DATE('%Y%m%d', end_date), '/*.parquet');
  
  -- Export training data in Parquet format
  EXPORT DATA OPTIONS(
    uri = export_uri,
    format = 'PARQUET',
    overwrite = TRUE
  ) AS
  SELECT * FROM `gaelp_campaigns.prepare_training_data`(agent_id, start_date, end_date);
  
  -- Log the export operation
  INSERT INTO `gaelp_campaigns.data_lifecycle_log` (
    operation_type, table_name, 
    status, completed_at, triggered_by
  ) VALUES (
    'export', 'training_data',
    'completed', CURRENT_TIMESTAMP(), 'api'
  );
  
END;

-- Procedure to sync performance data with external platforms
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.sync_external_platform_data`(
  IN platform_name STRING,
  IN sync_date DATE
)
BEGIN
  DECLARE sync_count INT64 DEFAULT 0;
  
  -- Create temporary table for external data
  CREATE TEMP TABLE external_metrics AS
  SELECT 
    campaign_id,
    TIMESTAMP(sync_date) as timestamp,
    'daily' as granularity,
    'actual' as metric_type,
    impressions,
    clicks,
    conversions,
    spend,
    platform_name as data_source
  FROM `gaelp_campaigns.external_platform_sync_staging`
  WHERE platform = platform_name
    AND sync_date = sync_date;
  
  -- Merge with existing performance metrics
  MERGE `gaelp_campaigns.performance_metrics` target
  USING external_metrics source
  ON target.campaign_id = source.campaign_id
    AND target.timestamp = source.timestamp
    AND target.granularity = source.granularity
  WHEN MATCHED THEN
    UPDATE SET 
      impressions = source.impressions,
      clicks = source.clicks,
      conversions = source.conversions,
      spend = source.spend,
      ctr = SAFE_DIVIDE(source.clicks, source.impressions),
      cpc = SAFE_DIVIDE(source.spend, source.clicks),
      roas = SAFE_DIVIDE(source.conversions * 100, source.spend),
      data_source = source.data_source
  WHEN NOT MATCHED THEN
    INSERT (
      campaign_id, timestamp, granularity, metric_type,
      impressions, clicks, conversions, spend,
      ctr, cpc, roas, data_source, created_at
    ) VALUES (
      source.campaign_id, source.timestamp, source.granularity, source.metric_type,
      source.impressions, source.clicks, source.conversions, source.spend,
      SAFE_DIVIDE(source.clicks, source.impressions),
      SAFE_DIVIDE(source.spend, source.clicks),
      SAFE_DIVIDE(source.conversions * 100, source.spend),
      source.data_source, CURRENT_TIMESTAMP()
    );
  
  SET sync_count = @@row_count;
  
  -- Log sync operation
  INSERT INTO `gaelp_campaigns.data_lifecycle_log` (
    operation_type, table_name, records_affected,
    status, completed_at, triggered_by, policy_name
  ) VALUES (
    'sync', 'performance_metrics', sync_count,
    'completed', CURRENT_TIMESTAMP(), 'scheduler', platform_name
  );
  
  -- Clean up staging data
  DELETE FROM `gaelp_campaigns.external_platform_sync_staging`
  WHERE platform = platform_name AND sync_date = sync_date;
  
END;

-- =============================================================================
-- STAGING TABLE FOR EXTERNAL DATA SYNC
-- =============================================================================

CREATE OR REPLACE TABLE `gaelp_campaigns.external_platform_sync_staging` (
  platform STRING NOT NULL,
  campaign_id STRING NOT NULL,
  sync_date DATE NOT NULL,
  
  -- Raw metrics from external platform
  impressions INT64,
  clicks INT64,
  conversions FLOAT64,
  spend FLOAT64,
  
  -- Platform-specific data
  external_campaign_id STRING,
  platform_metadata JSON,
  
  -- Sync metadata
  synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  sync_batch_id STRING,
  api_response_code INT64,
  
  -- Data quality
  validation_status STRING DEFAULT 'pending',
  validation_notes STRING
)
PARTITION BY sync_date
CLUSTER BY platform, campaign_id
OPTIONS (
  description = "Staging table for external platform data before processing",
  partition_expiration_days = 7
);