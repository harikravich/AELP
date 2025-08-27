-- =============================================================================
-- GAELP Monitoring and Alerting System
-- =============================================================================

-- =============================================================================
-- REAL-TIME MONITORING VIEWS
-- =============================================================================

-- Live campaign health dashboard
CREATE OR REPLACE VIEW `gaelp_campaigns.live_campaign_health` AS
WITH latest_metrics AS (
  SELECT 
    campaign_id,
    timestamp,
    impressions,
    clicks,
    conversions,
    spend,
    ctr,
    cpc,
    roas,
    ROW_NUMBER() OVER (
      PARTITION BY campaign_id 
      ORDER BY timestamp DESC
    ) as rn
  FROM `gaelp_campaigns.performance_metrics`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR)
    AND metric_type = 'actual'
),
campaign_health AS (
  SELECT 
    c.campaign_id,
    c.campaign_name,
    c.agent_id,
    c.status,
    c.campaign_type,
    c.platform,
    
    -- Latest metrics
    lm.timestamp as last_update,
    lm.impressions,
    lm.clicks,
    lm.conversions,
    lm.spend,
    lm.ctr,
    lm.cpc,
    lm.roas,
    
    -- Health indicators
    CASE 
      WHEN lm.timestamp < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 MINUTE) THEN 'Stale'
      WHEN lm.roas < 50 THEN 'Critical'
      WHEN lm.roas < 100 THEN 'Warning'
      WHEN lm.spend > CAST(JSON_EXTRACT_SCALAR(c.budget_config, '$.daily_limit') AS FLOAT64) * 0.9 THEN 'Budget Alert'
      ELSE 'Healthy'
    END as health_status,
    
    -- Performance flags
    ARRAY(
      SELECT flag FROM UNNEST([
        IF(lm.ctr < 0.005, 'Low CTR', NULL),
        IF(lm.cpc > 5.0, 'High CPC', NULL),
        IF(lm.roas < 100, 'Poor ROAS', NULL),
        IF(lm.spend = 0 AND c.status = 'active', 'No Spend', NULL),
        IF(lm.conversions = 0 AND lm.spend > 10, 'No Conversions', NULL)
      ]) AS flag
      WHERE flag IS NOT NULL
    ) as performance_flags,
    
    -- Budget utilization
    SAFE_DIVIDE(
      lm.spend, 
      CAST(JSON_EXTRACT_SCALAR(c.budget_config, '$.daily_limit') AS FLOAT64)
    ) as daily_budget_utilization,
    
    -- Data freshness
    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), lm.timestamp, MINUTE) as minutes_since_update
    
  FROM `gaelp_campaigns.campaigns` c
  LEFT JOIN latest_metrics lm ON c.campaign_id = lm.campaign_id AND lm.rn = 1
  WHERE c.status IN ('active', 'paused')
)
SELECT 
  *,
  -- Overall health score (0-100)
  GREATEST(0, LEAST(100, 
    50 + -- Base score
    (CASE WHEN roas > 150 THEN 20 ELSE (roas - 100) / 5 END) + -- ROAS component
    (CASE WHEN ctr > 0.02 THEN 15 ELSE ctr * 750 END) + -- CTR component
    (CASE WHEN daily_budget_utilization BETWEEN 0.7 AND 0.95 THEN 15 ELSE 0 END) - -- Budget efficiency
    (CASE WHEN minutes_since_update > 60 THEN 30 ELSE 0 END) -- Data freshness penalty
  )) as health_score
FROM campaign_health;

-- Agent performance anomaly detection
CREATE OR REPLACE VIEW `gaelp_campaigns.agent_anomaly_detection` AS
WITH agent_baselines AS (
  SELECT 
    agent_id,
    campaign_type,
    AVG(daily_roas) as baseline_roas,
    STDDEV(daily_roas) as roas_stddev,
    AVG(total_spend) as baseline_daily_spend,
    STDDEV(total_spend) as spend_stddev,
    COUNT(*) as baseline_days
  FROM `gaelp_campaigns.daily_campaign_performance`
  WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
                 AND DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  GROUP BY agent_id, campaign_type
  HAVING COUNT(*) >= 10 -- Need sufficient baseline data
),
recent_performance AS (
  SELECT 
    agent_id,
    campaign_type,
    date,
    AVG(daily_roas) as daily_roas,
    SUM(total_spend) as daily_spend
  FROM `gaelp_campaigns.daily_campaign_performance`
  WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  GROUP BY agent_id, campaign_type, date
)
SELECT 
  rp.agent_id,
  rp.campaign_type,
  rp.date,
  rp.daily_roas,
  rp.daily_spend,
  
  -- Baseline comparisons
  ab.baseline_roas,
  ab.baseline_daily_spend,
  
  -- Z-scores for anomaly detection
  SAFE_DIVIDE(
    rp.daily_roas - ab.baseline_roas, 
    ab.roas_stddev
  ) as roas_z_score,
  SAFE_DIVIDE(
    rp.daily_spend - ab.baseline_daily_spend, 
    ab.spend_stddev
  ) as spend_z_score,
  
  -- Anomaly flags
  CASE 
    WHEN ABS(SAFE_DIVIDE(rp.daily_roas - ab.baseline_roas, ab.roas_stddev)) > 2 THEN 'ROAS Anomaly'
    WHEN ABS(SAFE_DIVIDE(rp.daily_spend - ab.baseline_daily_spend, ab.spend_stddev)) > 2 THEN 'Spend Anomaly'
    ELSE 'Normal'
  END as anomaly_type,
  
  -- Severity assessment
  CASE 
    WHEN ABS(SAFE_DIVIDE(rp.daily_roas - ab.baseline_roas, ab.roas_stddev)) > 3 THEN 'Critical'
    WHEN ABS(SAFE_DIVIDE(rp.daily_roas - ab.baseline_roas, ab.roas_stddev)) > 2 THEN 'High'
    WHEN ABS(SAFE_DIVIDE(rp.daily_spend - ab.baseline_daily_spend, ab.spend_stddev)) > 2 THEN 'Medium'
    ELSE 'Low'
  END as severity
  
FROM recent_performance rp
JOIN agent_baselines ab 
  ON rp.agent_id = ab.agent_id 
  AND rp.campaign_type = ab.campaign_type
WHERE ABS(SAFE_DIVIDE(rp.daily_roas - ab.baseline_roas, ab.roas_stddev)) > 1.5
   OR ABS(SAFE_DIVIDE(rp.daily_spend - ab.baseline_daily_spend, ab.spend_stddev)) > 1.5;

-- =============================================================================
-- ALERTING FUNCTIONS
-- =============================================================================

-- Function to evaluate alert conditions
CREATE OR REPLACE FUNCTION `gaelp_campaigns.evaluate_alert_conditions`(
  campaign_metrics STRUCT<
    campaign_id STRING,
    roas FLOAT64,
    spend FLOAT64,
    ctr FLOAT64,
    minutes_since_update INT64,
    daily_budget FLOAT64
  >
) RETURNS ARRAY<STRUCT<
  alert_type STRING,
  severity STRING,
  message STRING,
  threshold_value FLOAT64,
  actual_value FLOAT64
>>
LANGUAGE js AS """
  const alerts = [];
  const m = campaign_metrics;
  
  // ROAS alerts
  if (m.roas < 50) {
    alerts.push({
      alert_type: 'POOR_ROAS',
      severity: 'CRITICAL',
      message: `ROAS of ${m.roas.toFixed(1)}% is critically low`,
      threshold_value: 50.0,
      actual_value: m.roas
    });
  } else if (m.roas < 100) {
    alerts.push({
      alert_type: 'LOW_ROAS',
      severity: 'WARNING',
      message: `ROAS of ${m.roas.toFixed(1)}% is below target`,
      threshold_value: 100.0,
      actual_value: m.roas
    });
  }
  
  // Spend alerts
  const spend_ratio = m.daily_budget > 0 ? m.spend / m.daily_budget : 0;
  if (spend_ratio > 0.95) {
    alerts.push({
      alert_type: 'BUDGET_EXHAUSTED',
      severity: 'HIGH',
      message: `Daily budget ${(spend_ratio * 100).toFixed(1)}% consumed`,
      threshold_value: 0.95,
      actual_value: spend_ratio
    });
  } else if (spend_ratio > 0.8) {
    alerts.push({
      alert_type: 'BUDGET_WARNING',
      severity: 'MEDIUM',
      message: `Daily budget ${(spend_ratio * 100).toFixed(1)}% consumed`,
      threshold_value: 0.8,
      actual_value: spend_ratio
    });
  }
  
  // CTR alerts
  if (m.ctr < 0.005) {
    alerts.push({
      alert_type: 'LOW_CTR',
      severity: 'MEDIUM',
      message: `CTR of ${(m.ctr * 100).toFixed(3)}% is below benchmark`,
      threshold_value: 0.005,
      actual_value: m.ctr
    });
  }
  
  // Data freshness alerts
  if (m.minutes_since_update > 120) {
    alerts.push({
      alert_type: 'STALE_DATA',
      severity: 'HIGH',
      message: `No data updates for ${m.minutes_since_update} minutes`,
      threshold_value: 120,
      actual_value: m.minutes_since_update
    });
  } else if (m.minutes_since_update > 60) {
    alerts.push({
      alert_type: 'DATA_DELAY',
      severity: 'MEDIUM',
      message: `Data delay of ${m.minutes_since_update} minutes`,
      threshold_value: 60,
      actual_value: m.minutes_since_update
    });
  }
  
  return alerts;
""";

-- =============================================================================
-- ALERT PROCESSING PROCEDURES
-- =============================================================================

-- Procedure to process and send alerts
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.process_alerts`()
BEGIN
  DECLARE done BOOL DEFAULT FALSE;
  DECLARE campaign_id STRING;
  DECLARE alert_conditions ARRAY<STRUCT<alert_type STRING, severity STRING, message STRING, threshold_value FLOAT64, actual_value FLOAT64>>;
  
  -- Cursor for campaigns that need alerting
  DECLARE alert_cursor CURSOR FOR
    SELECT 
      ch.campaign_id,
      `gaelp_campaigns.evaluate_alert_conditions`(
        STRUCT(
          ch.campaign_id,
          ch.roas,
          ch.spend,
          ch.ctr,
          ch.minutes_since_update,
          CAST(JSON_EXTRACT_SCALAR(c.budget_config, '$.daily_limit') AS FLOAT64)
        )
      ) as alerts
    FROM `gaelp_campaigns.live_campaign_health` ch
    JOIN `gaelp_campaigns.campaigns` c ON ch.campaign_id = c.campaign_id
    WHERE ch.health_status IN ('Critical', 'Warning', 'Budget Alert')
      OR ARRAY_LENGTH(ch.performance_flags) > 0;
  
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
  
  OPEN alert_cursor;
  
  alert_loop: LOOP
    FETCH alert_cursor INTO campaign_id, alert_conditions;
    
    IF done THEN
      LEAVE alert_loop;
    END IF;
    
    -- Process each alert condition
    IF ARRAY_LENGTH(alert_conditions) > 0 THEN
      -- Insert alerts into safety events table
      INSERT INTO `gaelp_campaigns.safety_events` (
        event_id,
        campaign_id,
        event_type,
        severity,
        category,
        description,
        detected_value,
        threshold_config,
        detection_method,
        detector_id,
        confidence_score,
        action_taken,
        action_details
      )
      SELECT 
        GENERATE_UUID() as event_id,
        campaign_id,
        alert.alert_type as event_type,
        alert.severity,
        'performance' as category,
        alert.message as description,
        JSON_OBJECT('value', alert.actual_value) as detected_value,
        JSON_OBJECT('threshold', alert.threshold_value) as threshold_config,
        'rule_based' as detection_method,
        'alert_processor_v1' as detector_id,
        1.0 as confidence_score,
        CASE 
          WHEN alert.severity = 'CRITICAL' THEN 'pause_campaign'
          WHEN alert.severity = 'HIGH' THEN 'notify_admin'
          ELSE 'monitor'
        END as action_taken,
        JSON_OBJECT(
          'alert_type', alert.alert_type,
          'auto_generated', TRUE,
          'notification_sent', TRUE
        ) as action_details
      FROM UNNEST(alert_conditions) as alert;
      
    END IF;
    
  END LOOP;
  
  CLOSE alert_cursor;
  
END;

-- =============================================================================
-- MONITORING DASHBOARDS AND REPORTS
-- =============================================================================

-- System health summary view
CREATE OR REPLACE VIEW `gaelp_campaigns.system_health_summary` AS
WITH health_metrics AS (
  SELECT 
    COUNT(*) as total_campaigns,
    COUNTIF(health_status = 'Healthy') as healthy_campaigns,
    COUNTIF(health_status = 'Warning') as warning_campaigns,
    COUNTIF(health_status = 'Critical') as critical_campaigns,
    COUNTIF(health_status = 'Stale') as stale_campaigns,
    
    -- Performance aggregates
    AVG(roas) as avg_system_roas,
    SUM(spend) as total_system_spend,
    SUM(conversions) as total_system_conversions,
    
    -- Data quality metrics
    AVG(minutes_since_update) as avg_data_latency,
    COUNTIF(minutes_since_update > 60) as delayed_campaigns,
    
    -- Budget metrics
    AVG(daily_budget_utilization) as avg_budget_utilization,
    COUNTIF(daily_budget_utilization > 0.9) as high_budget_utilization_campaigns
    
  FROM `gaelp_campaigns.live_campaign_health`
  WHERE last_update >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
),
recent_issues AS (
  SELECT 
    COUNT(*) as total_safety_events_24h,
    COUNTIF(severity = 'critical') as critical_events_24h,
    COUNTIF(resolved_at IS NULL) as unresolved_events
  FROM `gaelp_campaigns.safety_events`
  WHERE detected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
),
agent_status AS (
  SELECT 
    COUNT(DISTINCT agent_id) as active_agents,
    COUNT(DISTINCT CASE WHEN anomaly_type != 'Normal' THEN agent_id END) as anomalous_agents
  FROM `gaelp_campaigns.agent_anomaly_detection`
  WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
)
SELECT 
  -- Campaign health
  hm.total_campaigns,
  hm.healthy_campaigns,
  hm.warning_campaigns,
  hm.critical_campaigns,
  hm.stale_campaigns,
  SAFE_DIVIDE(hm.healthy_campaigns, hm.total_campaigns) * 100 as health_percentage,
  
  -- Performance metrics
  hm.avg_system_roas,
  hm.total_system_spend,
  hm.total_system_conversions,
  
  -- Data quality
  hm.avg_data_latency,
  hm.delayed_campaigns,
  SAFE_DIVIDE(hm.delayed_campaigns, hm.total_campaigns) * 100 as delayed_percentage,
  
  -- Budget management
  hm.avg_budget_utilization,
  hm.high_budget_utilization_campaigns,
  
  -- Safety and issues
  ri.total_safety_events_24h,
  ri.critical_events_24h,
  ri.unresolved_events,
  
  -- Agent status
  ag.active_agents,
  ag.anomalous_agents,
  SAFE_DIVIDE(ag.anomalous_agents, ag.active_agents) * 100 as anomaly_rate,
  
  -- Overall system status
  CASE 
    WHEN hm.critical_campaigns > 0 OR ri.critical_events_24h > 0 THEN 'Critical'
    WHEN hm.warning_campaigns > hm.total_campaigns * 0.2 THEN 'Degraded'
    WHEN hm.delayed_campaigns > hm.total_campaigns * 0.1 THEN 'Performance Issues'
    ELSE 'Healthy'
  END as overall_system_status,
  
  CURRENT_TIMESTAMP() as report_generated_at

FROM health_metrics hm
CROSS JOIN recent_issues ri
CROSS JOIN agent_status ag;

-- =============================================================================
-- SCHEDULED MONITORING JOBS
-- =============================================================================

-- Create table for alert subscription management
CREATE OR REPLACE TABLE `gaelp_campaigns.alert_subscriptions` (
  subscription_id STRING DEFAULT GENERATE_UUID(),
  user_email STRING NOT NULL,
  alert_types ARRAY<STRING> NOT NULL, -- Types of alerts to receive
  severity_levels ARRAY<STRING> NOT NULL, -- Minimum severity levels
  
  -- Scope filters
  agent_ids ARRAY<STRING>, -- Specific agents to monitor (NULL = all)
  campaign_types ARRAY<STRING>, -- Specific campaign types (NULL = all)
  platforms ARRAY<STRING>, -- Specific platforms (NULL = all)
  
  -- Delivery preferences
  delivery_method STRING DEFAULT 'email', -- 'email', 'slack', 'webhook'
  delivery_config JSON, -- Method-specific configuration
  
  -- Throttling
  max_alerts_per_hour INT64 DEFAULT 10,
  quiet_hours_start TIME, -- Start of quiet hours (no alerts)
  quiet_hours_end TIME, -- End of quiet hours
  
  -- Status
  active BOOL DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  last_alert_sent TIMESTAMP
)
CLUSTER BY user_email, active
OPTIONS (
  description = "Alert subscription configuration for users and systems"
);

-- Procedure to send alert notifications (placeholder for external integration)
CREATE OR REPLACE PROCEDURE `gaelp_campaigns.send_alert_notifications`()
BEGIN
  -- This procedure would integrate with external notification systems
  -- For now, we'll just log the alerts that would be sent
  
  INSERT INTO `gaelp_campaigns.data_lifecycle_log` (
    operation_type, table_name, records_affected,
    status, completed_at, triggered_by
  )
  SELECT 
    'notification' as operation_type,
    'alert_notifications' as table_name,
    COUNT(*) as records_affected,
    'completed' as status,
    CURRENT_TIMESTAMP() as completed_at,
    'scheduler' as triggered_by
  FROM `gaelp_campaigns.safety_events`
  WHERE detected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
    AND resolved_at IS NULL;
    
END;