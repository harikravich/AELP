-- =============================================================================
-- GAELP Ad Campaign Learning Platform - Core Campaign Tables
-- =============================================================================

-- Create dataset for ad campaign learning data
CREATE SCHEMA IF NOT EXISTS `gaelp_campaigns`
OPTIONS (
  description = "GAELP Ad Campaign Learning Platform - Campaign data, performance metrics, and agent training data",
  location = "us-central1"
);

-- =============================================================================
-- CAMPAIGNS TABLE
-- =============================================================================
CREATE OR REPLACE TABLE `gaelp_campaigns.campaigns` (
  -- Primary identifiers
  campaign_id STRING NOT NULL,
  agent_id STRING NOT NULL,
  environment_id STRING NOT NULL,
  
  -- Campaign metadata
  campaign_name STRING,
  campaign_type STRING NOT NULL, -- 'simulation', 'real', 'hybrid'
  status STRING NOT NULL, -- 'draft', 'active', 'paused', 'completed', 'failed'
  
  -- Configuration (stored as JSON for flexibility)
  configuration JSON NOT NULL,
  creative_config JSON,
  targeting_config JSON,
  budget_config JSON,
  
  -- Platform integration
  platform STRING, -- 'meta', 'google', 'simulation'
  external_campaign_id STRING, -- Platform-specific ID
  external_metadata JSON,
  
  -- Agent and learning context
  agent_version STRING,
  parent_campaign_id STRING, -- For sim-to-real progression
  simulation_run_id STRING,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  started_at TIMESTAMP,
  ended_at TIMESTAMP,
  
  -- Safety and compliance
  safety_status STRING DEFAULT 'pending', -- 'pending', 'approved', 'rejected', 'flagged'
  compliance_checks JSON,
  
  -- Metadata
  tags ARRAY<STRING>,
  labels JSON,
  notes STRING
)
PARTITION BY DATE(created_at)
CLUSTER BY agent_id, campaign_type, status
OPTIONS (
  description = "Core campaigns table storing campaign configurations and metadata",
  partition_expiration_days = 1095, -- 3 years
  require_partition_filter = TRUE
);

-- =============================================================================
-- PERFORMANCE METRICS TABLE
-- =============================================================================
CREATE OR REPLACE TABLE `gaelp_campaigns.performance_metrics` (
  -- Primary identifiers
  campaign_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  
  -- Granularity and aggregation level
  granularity STRING NOT NULL, -- 'hourly', 'daily', 'episode'
  metric_type STRING NOT NULL, -- 'actual', 'simulated', 'predicted'
  
  -- Core performance metrics
  impressions INT64 DEFAULT 0,
  clicks INT64 DEFAULT 0,
  conversions FLOAT64 DEFAULT 0,
  spend FLOAT64 DEFAULT 0,
  
  -- Calculated metrics
  ctr FLOAT64, -- click-through rate
  cpc FLOAT64, -- cost per click
  cpm FLOAT64, -- cost per mille
  cpa FLOAT64, -- cost per acquisition
  roas FLOAT64, -- return on ad spend
  
  -- Advanced metrics for learning
  reach INT64,
  frequency FLOAT64,
  engagement_rate FLOAT64,
  quality_score FLOAT64,
  
  -- Platform-specific metrics
  platform_metrics JSON,
  
  -- Attribution and conversion data
  view_through_conversions FLOAT64 DEFAULT 0,
  click_through_conversions FLOAT64 DEFAULT 0,
  conversion_value FLOAT64 DEFAULT 0,
  
  -- Simulation-specific data
  persona_responses JSON, -- For simulated campaigns
  confidence_score FLOAT64, -- Model confidence in predictions
  
  -- Data quality
  data_source STRING, -- 'platform_api', 'simulation', 'manual'
  is_estimated BOOL DEFAULT FALSE,
  quality_flags ARRAY<STRING>,
  
  -- Metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY campaign_id, granularity, metric_type
OPTIONS (
  description = "Time-series performance metrics for campaigns with multiple granularities",
  partition_expiration_days = 1095, -- 3 years
  require_partition_filter = TRUE
);

-- =============================================================================
-- AGENT EPISODES TABLE
-- =============================================================================
CREATE OR REPLACE TABLE `gaelp_campaigns.agent_episodes` (
  -- Primary identifiers
  episode_id STRING NOT NULL,
  campaign_id STRING NOT NULL,
  agent_id STRING NOT NULL,
  
  -- Episode context
  episode_number INT64 NOT NULL,
  step_number INT64 NOT NULL,
  
  -- RL state representation
  state JSON NOT NULL, -- Current campaign state
  action JSON NOT NULL, -- Action taken by agent
  reward FLOAT64 NOT NULL,
  next_state JSON,
  done BOOL NOT NULL,
  
  -- Action details
  action_type STRING NOT NULL, -- 'budget_adjust', 'creative_update', 'targeting_change', 'bid_modify'
  action_value JSON, -- Specific action parameters
  
  -- Reward components (for analysis)
  reward_components JSON, -- Breakdown of reward calculation
  immediate_reward FLOAT64,
  future_reward_estimate FLOAT64,
  
  -- Agent decision context
  policy_version STRING,
  exploration_factor FLOAT64,
  confidence_score FLOAT64,
  alternative_actions JSON, -- Other actions considered
  
  -- Environment feedback
  environment_response JSON,
  execution_status STRING, -- 'success', 'failed', 'partial'
  execution_latency_ms INT64,
  
  -- Learning metadata
  q_values ARRAY<FLOAT64>, -- Q-values for different actions
  policy_gradient JSON,
  value_function_estimate FLOAT64,
  
  -- Timestamps
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  action_executed_at TIMESTAMP,
  reward_observed_at TIMESTAMP
)
PARTITION BY DATE(timestamp)
CLUSTER BY campaign_id, agent_id, episode_number
OPTIONS (
  description = "Detailed agent episode data for reinforcement learning training",
  partition_expiration_days = 730, -- 2 years
  require_partition_filter = TRUE
);

-- =============================================================================
-- SIMULATION DATA TABLE
-- =============================================================================
CREATE OR REPLACE TABLE `gaelp_campaigns.simulation_data` (
  -- Primary identifiers
  simulation_id STRING NOT NULL,
  campaign_id STRING NOT NULL,
  persona_id STRING NOT NULL,
  
  -- Simulation context
  simulation_timestamp TIMESTAMP NOT NULL,
  simulation_type STRING NOT NULL, -- 'ab_test', 'creative_eval', 'targeting_test'
  
  -- Creative and targeting inputs
  creative_config JSON NOT NULL,
  targeting_config JSON NOT NULL,
  ad_creative JSON NOT NULL, -- Text, images, video metadata
  
  -- Persona response
  persona_response JSON NOT NULL, -- LLM persona response
  interaction_type STRING, -- 'impression', 'click', 'conversion', 'ignore'
  engagement_score FLOAT64,
  sentiment_score FLOAT64,
  
  -- Response reasoning
  response_reasoning TEXT, -- Why the persona responded this way
  decision_factors JSON, -- Factors that influenced the decision
  
  -- Behavioral modeling
  attention_score FLOAT64,
  relevance_score FLOAT64,
  trust_score FLOAT64,
  fatigue_score FLOAT64, -- Ad fatigue simulation
  
  -- Conversion modeling
  conversion_probability FLOAT64,
  conversion_value_estimate FLOAT64,
  conversion_type STRING, -- 'purchase', 'signup', 'download', etc.
  
  -- Simulation quality
  model_version STRING,
  temperature FLOAT64, -- LLM generation temperature
  tokens_used INT64,
  generation_latency_ms INT64,
  
  -- Metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  batch_id STRING -- For batch processing identification
)
PARTITION BY DATE(simulation_timestamp)
CLUSTER BY campaign_id, persona_id, interaction_type
OPTIONS (
  description = "Simulated user interactions and persona responses to ad campaigns",
  partition_expiration_days = 1095, -- 3 years
  require_partition_filter = TRUE
);

-- =============================================================================
-- PERSONAS TABLE
-- =============================================================================
CREATE OR REPLACE TABLE `gaelp_campaigns.personas` (
  -- Primary identifiers
  persona_id STRING NOT NULL,
  persona_name STRING NOT NULL,
  
  -- Demographic data
  demographics JSON NOT NULL, -- age, gender, location, income, etc.
  
  -- Psychographic data
  interests ARRAY<STRING>,
  behavior_patterns JSON, -- shopping habits, media consumption, etc.
  personality_traits JSON, -- Big Five, other psychological models
  
  -- Digital behavior
  device_usage JSON, -- preferred devices, usage patterns
  platform_preferences ARRAY<STRING>, -- social media, search engines, etc.
  ad_sensitivity JSON, -- how they respond to different ad types
  
  -- Purchase behavior
  purchase_history JSON,
  price_sensitivity FLOAT64,
  brand_loyalty JSON,
  decision_making_style STRING,
  
  -- Simulation parameters
  response_variability FLOAT64, -- How consistent responses are
  fatigue_threshold FLOAT64, -- How quickly they get ad fatigue
  trust_baseline FLOAT64, -- Base trust level
  
  -- LLM configuration
  system_prompt TEXT NOT NULL,
  temperature FLOAT64 DEFAULT 0.7,
  max_tokens INT64 DEFAULT 500,
  
  -- Metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  active BOOL DEFAULT TRUE,
  tags ARRAY<STRING>
)
CLUSTER BY active
OPTIONS (
  description = "User personas for simulation-based campaign testing"
);

-- =============================================================================
-- SAFETY EVENTS TABLE
-- =============================================================================
CREATE OR REPLACE TABLE `gaelp_campaigns.safety_events` (
  -- Primary identifiers
  event_id STRING NOT NULL,
  campaign_id STRING,
  agent_id STRING,
  
  -- Event classification
  event_type STRING NOT NULL, -- 'budget_exceeded', 'inappropriate_creative', 'targeting_violation', 'performance_anomaly'
  severity STRING NOT NULL, -- 'low', 'medium', 'high', 'critical'
  category STRING NOT NULL, -- 'safety', 'compliance', 'performance', 'technical'
  
  -- Event details
  description TEXT NOT NULL,
  detected_value JSON, -- The value that triggered the event
  threshold_config JSON, -- The threshold that was exceeded
  
  -- Detection context
  detection_method STRING, -- 'rule_based', 'ml_model', 'human_review'
  detector_id STRING, -- Specific detector/rule that triggered
  confidence_score FLOAT64,
  
  -- Action taken
  action_taken STRING NOT NULL, -- 'pause_campaign', 'modify_creative', 'adjust_budget', 'human_review'
  action_details JSON,
  auto_resolved BOOL DEFAULT FALSE,
  
  -- Resolution tracking
  resolved_at TIMESTAMP,
  resolved_by STRING, -- user_id or 'system'
  resolution_notes TEXT,
  
  -- Impact assessment
  financial_impact FLOAT64, -- Estimated cost/loss
  performance_impact JSON, -- Impact on campaign metrics
  
  -- Timestamps
  detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(detected_at)
CLUSTER BY severity, event_type, campaign_id
OPTIONS (
  description = "Safety events and incidents for campaign monitoring and compliance",
  partition_expiration_days = 2555, -- 7 years for compliance
  require_partition_filter = TRUE
);

-- =============================================================================
-- CREATE INDEXES FOR PERFORMANCE
-- =============================================================================

-- Note: BigQuery automatically creates indexes and optimizes queries
-- No explicit CREATE INDEX statements needed