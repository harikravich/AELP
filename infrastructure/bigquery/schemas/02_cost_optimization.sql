-- =============================================================================
-- GAELP Cost Optimization and Data Lifecycle Management
-- =============================================================================

-- =============================================================================
-- MATERIALIZED VIEWS FOR COST OPTIMIZATION
-- =============================================================================

-- Daily campaign performance aggregation (heavily queried)
CREATE MATERIALIZED VIEW `gaelp_campaigns.daily_campaign_performance`
PARTITION BY DATE(date)
CLUSTER BY campaign_id, agent_id
AS
SELECT 
  campaign_id,
  agent_id,
  DATE(timestamp) as date,
  campaign_type,
  platform,
  
  -- Aggregated metrics
  SUM(impressions) as total_impressions,
  SUM(clicks) as total_clicks,
  SUM(conversions) as total_conversions,
  SUM(spend) as total_spend,
  
  -- Calculated KPIs
  SAFE_DIVIDE(SUM(clicks), SUM(impressions)) as daily_ctr,
  SAFE_DIVIDE(SUM(spend), SUM(clicks)) as daily_cpc,
  SAFE_DIVIDE(SUM(conversions) * 100, SUM(spend)) as daily_roas,
  
  -- Statistical measures
  COUNT(*) as data_points,
  STDDEV(ctr) as ctr_variance,
  MIN(timestamp) as first_update,
  MAX(timestamp) as last_update,
  
  -- Cost attribution
  SUM(spend) / COUNT(DISTINCT campaign_id) as attributed_spend,
  
  CURRENT_TIMESTAMP() as materialized_at
  
FROM `gaelp_campaigns.performance_metrics`
WHERE granularity = 'hourly'
  AND metric_type = 'actual'
GROUP BY 
  campaign_id, agent_id, DATE(timestamp), campaign_type, platform;

-- Agent performance leaderboard (frequently accessed)
CREATE MATERIALIZED VIEW `gaelp_campaigns.agent_leaderboard`
CLUSTER BY performance_tier, campaign_type
AS
WITH recent_performance AS (
  SELECT 
    agent_id,
    campaign_type,
    AVG(daily_roas) as avg_roas_30d,
    SUM(total_spend) as total_spend_30d,
    COUNT(DISTINCT campaign_id) as campaigns_30d,
    SUM(total_conversions) as total_conversions_30d
  FROM `gaelp_campaigns.daily_campaign_performance`
  WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  GROUP BY agent_id, campaign_type
),
all_time_performance AS (
  SELECT 
    agent_id,
    campaign_type,
    AVG(daily_roas) as avg_roas_all_time,
    SUM(total_spend) as total_spend_all_time,
    COUNT(DISTINCT campaign_id) as campaigns_all_time,
    MIN(date) as first_campaign_date,
    MAX(date) as last_campaign_date
  FROM `gaelp_campaigns.daily_campaign_performance`
  GROUP BY agent_id, campaign_type
)
SELECT 
  rp.agent_id,
  rp.campaign_type,
  
  -- Recent performance
  rp.avg_roas_30d,
  rp.total_spend_30d,
  rp.campaigns_30d,
  rp.total_conversions_30d,
  
  -- All-time performance
  atp.avg_roas_all_time,
  atp.total_spend_all_time,
  atp.campaigns_all_time,
  atp.first_campaign_date,
  atp.last_campaign_date,
  
  -- Performance rankings
  ROW_NUMBER() OVER (
    PARTITION BY rp.campaign_type 
    ORDER BY rp.avg_roas_30d DESC
  ) as roas_rank_30d,
  
  ROW_NUMBER() OVER (
    PARTITION BY rp.campaign_type 
    ORDER BY atp.avg_roas_all_time DESC
  ) as roas_rank_all_time,
  
  -- Performance classification
  CASE 
    WHEN rp.avg_roas_30d >= 300 THEN 'Elite'
    WHEN rp.avg_roas_30d >= 200 THEN 'Advanced'
    WHEN rp.avg_roas_30d >= 150 THEN 'Intermediate'
    WHEN rp.avg_roas_30d >= 100 THEN 'Beginner'
    ELSE 'Training'
  END as performance_tier,
  
  -- Trend analysis
  CASE 
    WHEN rp.avg_roas_30d > atp.avg_roas_all_time * 1.1 THEN 'Improving'
    WHEN rp.avg_roas_30d < atp.avg_roas_all_time * 0.9 THEN 'Declining'
    ELSE 'Stable'
  END as performance_trend,
  
  CURRENT_TIMESTAMP() as materialized_at

FROM recent_performance rp
JOIN all_time_performance atp 
  ON rp.agent_id = atp.agent_id 
  AND rp.campaign_type = atp.campaign_type;

-- =============================================================================
-- DATA ARCHIVAL AND LIFECYCLE TABLES
-- =============================================================================

-- Cold storage table for old performance metrics
CREATE OR REPLACE TABLE `gaelp_campaigns.performance_metrics_archive` (
  -- Same schema as main table but optimized for storage
  campaign_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  granularity STRING NOT NULL,
  metric_type STRING NOT NULL,
  
  -- Compressed metrics (store as JSON for flexibility)
  compressed_metrics JSON NOT NULL,
  
  -- Archival metadata
  archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  original_partition_date DATE,
  compression_ratio FLOAT64,
  
  -- Checksum for data integrity
  data_checksum STRING
)
PARTITION BY DATE(original_partition_date)
CLUSTER BY campaign_id
OPTIONS (
  description = "Archived performance metrics older than 1 year",
  partition_expiration_days = 2555 -- 7 years for compliance
);

-- Table for tracking data lifecycle operations
CREATE OR REPLACE TABLE `gaelp_campaigns.data_lifecycle_log` (
  operation_id STRING DEFAULT GENERATE_UUID(),
  operation_type STRING NOT NULL, -- 'archive', 'delete', 'compress', 'export'
  table_name STRING NOT NULL,
  partition_date DATE,
  
  -- Operation details
  records_affected INT64,
  bytes_processed INT64,
  compression_achieved FLOAT64,
  cost_savings_usd FLOAT64,
  
  -- Execution tracking
  status STRING DEFAULT 'pending',
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  completed_at TIMESTAMP,
  error_message STRING,
  
  -- Metadata
  triggered_by STRING, -- 'scheduler', 'manual', 'policy'
  policy_name STRING
)
PARTITION BY DATE(started_at)
CLUSTER BY operation_type, status
OPTIONS (
  description = "Log of data lifecycle management operations"
);

-- =============================================================================
-- COST MONITORING TABLES
-- =============================================================================

-- Table for tracking BigQuery costs by query and user
CREATE OR REPLACE TABLE `gaelp_campaigns.query_cost_tracking` (
  query_id STRING NOT NULL,
  project_id STRING NOT NULL,
  user_email STRING,
  
  -- Query details
  query_text STRING,
  query_type STRING, -- 'SELECT', 'INSERT', 'CREATE_VIEW', etc.
  tables_accessed ARRAY<STRING>,
  
  -- Cost metrics
  bytes_processed INT64,
  bytes_billed INT64,
  slot_ms INT64,
  estimated_cost_usd FLOAT64,
  
  -- Performance metrics
  creation_time TIMESTAMP,
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  duration_ms INT64,
  
  -- Optimization flags
  uses_partition_filter BOOL,
  uses_clustering BOOL,
  uses_materialized_view BOOL,
  optimization_opportunities ARRAY<STRING>
)
PARTITION BY DATE(creation_time)
CLUSTER BY user_email, query_type
OPTIONS (
  description = "Query cost and performance tracking for optimization",
  partition_expiration_days = 90
);

-- Table for cost budgets and alerts
CREATE OR REPLACE TABLE `gaelp_campaigns.cost_budgets` (
  budget_id STRING DEFAULT GENERATE_UUID(),
  budget_name STRING NOT NULL,
  
  -- Budget scope
  scope_type STRING NOT NULL, -- 'project', 'dataset', 'table', 'user'
  scope_value STRING NOT NULL,
  
  -- Budget limits
  monthly_budget_usd FLOAT64 NOT NULL,
  daily_budget_usd FLOAT64,
  alert_threshold_percent FLOAT64 DEFAULT 80.0,
  
  -- Current usage
  current_month_spend FLOAT64 DEFAULT 0.0,
  current_day_spend FLOAT64 DEFAULT 0.0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  
  -- Alert configuration
  alert_emails ARRAY<STRING>,
  auto_pause_enabled BOOL DEFAULT FALSE,
  
  -- Status
  active BOOL DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  created_by STRING
)
CLUSTER BY scope_type, active
OPTIONS (
  description = "Cost budgets and alerting configuration"
);

-- =============================================================================
-- DATA LIFECYCLE MANAGEMENT PROCEDURES
-- =============================================================================

-- Procedure to archive old performance metrics
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.archive_old_performance_metrics`(
  IN cutoff_date DATE
)
BEGIN
  DECLARE records_archived INT64 DEFAULT 0;
  DECLARE bytes_saved INT64 DEFAULT 0;
  DECLARE operation_id STRING DEFAULT GENERATE_UUID();
  
  -- Log operation start
  INSERT INTO `gaelp_campaigns.data_lifecycle_log` (
    operation_id, operation_type, table_name, partition_date,
    status, triggered_by
  ) VALUES (
    operation_id, 'archive', 'performance_metrics', cutoff_date,
    'running', 'scheduler'
  );
  
  -- Archive data with compression
  INSERT INTO `gaelp_campaigns.performance_metrics_archive` (
    campaign_id,
    timestamp,
    granularity,
    metric_type,
    compressed_metrics,
    original_partition_date,
    data_checksum
  )
  SELECT 
    campaign_id,
    timestamp,
    granularity,
    metric_type,
    
    -- Compress metrics into JSON
    JSON_OBJECT(
      'impressions', impressions,
      'clicks', clicks,
      'conversions', conversions,
      'spend', spend,
      'ctr', ctr,
      'cpc', cpc,
      'roas', roas,
      'platform_metrics', platform_metrics
    ) as compressed_metrics,
    
    DATE(timestamp) as original_partition_date,
    
    -- Simple checksum
    TO_HEX(MD5(CONCAT(
      campaign_id, 
      CAST(timestamp AS STRING),
      CAST(impressions AS STRING),
      CAST(spend AS STRING)
    ))) as data_checksum
    
  FROM `gaelp_campaigns.performance_metrics`
  WHERE DATE(timestamp) <= cutoff_date;
  
  -- Get count of archived records
  SET records_archived = @@row_count;
  
  -- Delete archived data from main table
  DELETE FROM `gaelp_campaigns.performance_metrics`
  WHERE DATE(timestamp) <= cutoff_date;
  
  -- Update operation log
  UPDATE `gaelp_campaigns.data_lifecycle_log`
  SET 
    status = 'completed',
    completed_at = CURRENT_TIMESTAMP(),
    records_affected = records_archived,
    compression_achieved = 0.6 -- Estimated 60% compression
  WHERE operation_id = operation_id;
  
END;

-- Procedure to clean up old simulation data
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.cleanup_old_simulation_data`(
  IN retention_days INT64
)
BEGIN
  DECLARE cutoff_timestamp TIMESTAMP;
  DECLARE records_deleted INT64 DEFAULT 0;
  
  SET cutoff_timestamp = TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL retention_days DAY);
  
  -- Delete old simulation data
  DELETE FROM `gaelp_campaigns.simulation_data`
  WHERE simulation_timestamp < cutoff_timestamp;
  
  SET records_deleted = @@row_count;
  
  -- Log the cleanup operation
  INSERT INTO `gaelp_campaigns.data_lifecycle_log` (
    operation_type, table_name, records_affected,
    status, completed_at, triggered_by
  ) VALUES (
    'delete', 'simulation_data', records_deleted,
    'completed', CURRENT_TIMESTAMP(), 'scheduler'
  );
  
END;

-- =============================================================================
-- COST OPTIMIZATION PROCEDURES
-- =============================================================================

-- Procedure to update query cost tracking
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.update_query_costs`()
BEGIN
  -- This would typically pull from INFORMATION_SCHEMA.JOBS_BY_PROJECT
  -- For now, we'll create a placeholder procedure
  
  INSERT INTO `gaelp_campaigns.query_cost_tracking` (
    query_id,
    project_id,
    user_email,
    query_text,
    query_type,
    bytes_processed,
    bytes_billed,
    estimated_cost_usd,
    creation_time,
    start_time,
    end_time,
    duration_ms,
    uses_partition_filter,
    uses_clustering
  )
  SELECT 
    job_id as query_id,
    project_id,
    user_email,
    query as query_text,
    statement_type as query_type,
    total_bytes_processed as bytes_processed,
    total_bytes_billed as bytes_billed,
    (total_bytes_billed / 1024 / 1024 / 1024 / 1024) * 5.0 as estimated_cost_usd, -- $5 per TB
    creation_time,
    start_time,
    end_time,
    TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) as duration_ms,
    
    -- Check for optimization patterns
    REGEXP_CONTAINS(query, r'WHERE.*_PARTITIONTIME') as uses_partition_filter,
    FALSE as uses_clustering -- Would need more sophisticated analysis
    
  FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
  WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    AND job_type = 'QUERY'
    AND state = 'DONE'
    AND error_result IS NULL;
    
END;

-- Procedure to check cost budgets and send alerts
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.check_cost_budgets`()
BEGIN
  DECLARE done BOOL DEFAULT FALSE;
  DECLARE budget_id STRING;
  DECLARE budget_name STRING;
  DECLARE monthly_budget FLOAT64;
  DECLARE current_spend FLOAT64;
  DECLARE threshold_percent FLOAT64;
  DECLARE alert_emails ARRAY<STRING>;
  
  -- Cursor for active budgets
  DECLARE budget_cursor CURSOR FOR
    SELECT 
      b.budget_id,
      b.budget_name,
      b.monthly_budget_usd,
      b.current_month_spend,
      b.alert_threshold_percent,
      b.alert_emails
    FROM `gaelp_campaigns.cost_budgets` b
    WHERE b.active = TRUE;
  
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
  
  OPEN budget_cursor;
  
  budget_loop: LOOP
    FETCH budget_cursor INTO 
      budget_id, budget_name, monthly_budget, 
      current_spend, threshold_percent, alert_emails;
    
    IF done THEN
      LEAVE budget_loop;
    END IF;
    
    -- Check if threshold exceeded
    IF current_spend >= (monthly_budget * threshold_percent / 100) THEN
      -- Log alert (in practice, this would trigger external alerting)
      INSERT INTO `gaelp_campaigns.data_lifecycle_log` (
        operation_type, table_name, 
        records_affected, -- Use this field to store spend amount
        status, triggered_by, policy_name
      ) VALUES (
        'alert', 'cost_budgets',
        CAST(current_spend AS INT64),
        'completed', 'scheduler', budget_name
      );
    END IF;
    
  END LOOP;
  
  CLOSE budget_cursor;
  
END;