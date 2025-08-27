-- =============================================================================
-- GAELP User Journey Database Schema
-- Comprehensive multi-touch attribution and user journey tracking
-- =============================================================================

-- Users table - Core identity resolution
CREATE OR REPLACE TABLE `gaelp.users` (
  -- Primary identifiers
  user_id STRING NOT NULL,
  canonical_user_id STRING NOT NULL, -- After identity resolution
  
  -- Identity resolution data
  device_ids ARRAY<STRING>,
  email_hash STRING,
  phone_hash STRING,
  fingerprint_hash STRING,
  ip_address STRING,
  
  -- Demographics and segments
  age_range STRING,
  gender STRING,
  location_country STRING,
  location_region STRING,
  location_city STRING,
  
  -- Journey state
  current_journey_state STRING NOT NULL DEFAULT 'UNAWARE',
  journey_score FLOAT64 DEFAULT 0.0,
  conversion_probability FLOAT64 DEFAULT 0.0,
  
  -- Timestamps
  first_seen TIMESTAMP NOT NULL,
  last_seen TIMESTAMP NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(first_seen)
CLUSTER BY canonical_user_id, current_journey_state;

-- User journeys - Individual journey instances
CREATE OR REPLACE TABLE `gaelp.user_journeys` (
  journey_id STRING NOT NULL,
  user_id STRING NOT NULL,
  canonical_user_id STRING NOT NULL,
  
  -- Journey metadata
  journey_start TIMESTAMP NOT NULL,
  journey_end TIMESTAMP,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  timeout_at TIMESTAMP NOT NULL, -- 14-day timeout
  
  -- Journey progression
  initial_state STRING NOT NULL DEFAULT 'UNAWARE',
  current_state STRING NOT NULL DEFAULT 'UNAWARE',
  final_state STRING,
  state_progression ARRAY<STRUCT<
    state STRING,
    timestamp TIMESTAMP,
    confidence FLOAT64,
    trigger_event STRING
  >>,
  
  -- Journey outcomes
  converted BOOLEAN NOT NULL DEFAULT FALSE,
  conversion_timestamp TIMESTAMP,
  conversion_value FLOAT64,
  conversion_type STRING,
  
  -- Attribution data
  first_touch_channel STRING,
  last_touch_channel STRING,
  touchpoint_count INT64 DEFAULT 0,
  days_to_conversion INT64,
  
  -- Journey scoring
  journey_score FLOAT64 DEFAULT 0.0,
  engagement_score FLOAT64 DEFAULT 0.0,
  intent_score FLOAT64 DEFAULT 0.0,
  
  -- Metadata
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(journey_start)
CLUSTER BY canonical_user_id, current_state, is_active;

-- Touchpoints - Individual interactions within journeys
CREATE OR REPLACE TABLE `gaelp.journey_touchpoints` (
  touchpoint_id STRING NOT NULL,
  journey_id STRING NOT NULL,
  user_id STRING NOT NULL,
  canonical_user_id STRING NOT NULL,
  
  -- Touchpoint data
  timestamp TIMESTAMP NOT NULL,
  channel STRING NOT NULL,
  campaign_id STRING,
  creative_id STRING,
  placement_id STRING,
  
  -- Interaction details
  interaction_type STRING NOT NULL, -- impression, click, view, engagement
  page_url STRING,
  referrer_url STRING,
  device_type STRING,
  browser STRING,
  os STRING,
  
  -- Content and targeting
  content_category STRING,
  message_variant STRING,
  audience_segment STRING,
  targeting_criteria JSON,
  
  -- State impact
  pre_state STRING,
  post_state STRING,
  state_change_confidence FLOAT64,
  
  -- Engagement metrics
  dwell_time_seconds FLOAT64,
  scroll_depth FLOAT64,
  click_depth INT64,
  engagement_score FLOAT64,
  
  -- Attribution weights
  first_touch_weight FLOAT64 DEFAULT 0.0,
  last_touch_weight FLOAT64 DEFAULT 0.0,
  time_decay_weight FLOAT64 DEFAULT 0.0,
  position_weight FLOAT64 DEFAULT 0.0,
  
  -- Metadata
  session_id STRING,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY canonical_user_id, journey_id, channel;

-- Journey states - State transition tracking
CREATE OR REPLACE TABLE `gaelp.journey_state_transitions` (
  transition_id STRING NOT NULL,
  journey_id STRING NOT NULL,
  user_id STRING NOT NULL,
  canonical_user_id STRING NOT NULL,
  
  -- State transition
  from_state STRING NOT NULL,
  to_state STRING NOT NULL,
  transition_timestamp TIMESTAMP NOT NULL,
  
  -- Transition triggers
  trigger_touchpoint_id STRING,
  trigger_channel STRING,
  trigger_event STRING,
  trigger_confidence FLOAT64,
  
  -- Context
  session_id STRING,
  days_since_journey_start INT64,
  touchpoints_since_last_transition INT64,
  
  -- ML predictions
  predicted_next_state STRING,
  next_state_probability FLOAT64,
  conversion_probability FLOAT64,
  
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(transition_timestamp)
CLUSTER BY canonical_user_id, from_state, to_state;

-- Competitor exposure tracking
CREATE OR REPLACE TABLE `gaelp.competitor_exposures` (
  exposure_id STRING NOT NULL,
  user_id STRING NOT NULL,
  canonical_user_id STRING NOT NULL,
  journey_id STRING,
  
  -- Competitor data
  competitor_name STRING NOT NULL,
  competitor_channel STRING NOT NULL,
  exposure_timestamp TIMESTAMP NOT NULL,
  exposure_type STRING NOT NULL, -- ad, organic, direct
  
  -- Context
  pre_exposure_state STRING,
  post_exposure_state STRING,
  state_impact_score FLOAT64,
  
  -- Competitive intelligence
  competitor_message STRING,
  competitor_offer STRING,
  price_comparison FLOAT64,
  feature_comparison JSON,
  
  -- Impact on journey
  journey_impact_score FLOAT64,
  conversion_probability_change FLOAT64,
  
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(exposure_timestamp)
CLUSTER BY canonical_user_id, competitor_name;

-- Channel history - Historical performance by channel
CREATE OR REPLACE TABLE `gaelp.channel_history` (
  channel_performance_id STRING NOT NULL,
  channel STRING NOT NULL,
  date DATE NOT NULL,
  
  -- Channel metrics
  impressions INT64 DEFAULT 0,
  clicks INT64 DEFAULT 0,
  conversions INT64 DEFAULT 0,
  conversion_value FLOAT64 DEFAULT 0.0,
  
  -- Journey impact
  state_progressions INT64 DEFAULT 0,
  journey_starts INT64 DEFAULT 0,
  journey_completions INT64 DEFAULT 0,
  
  -- Attribution metrics
  first_touch_conversions INT64 DEFAULT 0,
  last_touch_conversions INT64 DEFAULT 0,
  assisted_conversions INT64 DEFAULT 0,
  
  -- Quality scores
  engagement_score FLOAT64 DEFAULT 0.0,
  intent_score FLOAT64 DEFAULT 0.0,
  conversion_rate FLOAT64 DEFAULT 0.0,
  
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY date
CLUSTER BY channel, date;

-- Identity resolution log
CREATE OR REPLACE TABLE `gaelp.identity_resolution_log` (
  resolution_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  
  -- Identity matching
  user_ids ARRAY<STRING> NOT NULL,
  canonical_user_id STRING NOT NULL,
  resolution_method STRING NOT NULL,
  confidence_score FLOAT64 NOT NULL,
  
  -- Matching criteria
  matching_signals ARRAY<STRING>,
  signal_weights JSON,
  
  -- Impact
  journeys_merged INT64 DEFAULT 0,
  touchpoints_merged INT64 DEFAULT 0,
  
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY canonical_user_id;

-- =============================================================================
-- Materialized Views for Analytics
-- =============================================================================

-- Active journeys summary
CREATE OR REPLACE MATERIALIZED VIEW `gaelp.active_journeys_summary` AS
SELECT 
  current_state,
  COUNT(*) as journey_count,
  AVG(journey_score) as avg_journey_score,
  AVG(engagement_score) as avg_engagement_score,
  AVG(intent_score) as avg_intent_score,
  AVG(conversion_probability) as avg_conversion_probability,
  AVG(touchpoint_count) as avg_touchpoint_count,
  AVG(DATE_DIFF(CURRENT_DATE(), DATE(journey_start), DAY)) as avg_journey_age_days
FROM `gaelp.user_journeys`
WHERE is_active = TRUE
  AND timeout_at > CURRENT_TIMESTAMP()
GROUP BY current_state;

-- Channel performance summary
CREATE OR REPLACE MATERIALIZED VIEW `gaelp.channel_performance_summary` AS
SELECT 
  channel,
  DATE_TRUNC(timestamp, DAY) as date,
  COUNT(*) as touchpoints,
  COUNT(DISTINCT canonical_user_id) as unique_users,
  COUNT(DISTINCT journey_id) as unique_journeys,
  AVG(engagement_score) as avg_engagement_score,
  SUM(CASE WHEN post_state != pre_state THEN 1 ELSE 0 END) as state_progressions
FROM `gaelp.journey_touchpoints`
WHERE timestamp >= DATE_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY channel, DATE_TRUNC(timestamp, DAY);

-- Journey funnel analysis
CREATE OR REPLACE MATERIALIZED VIEW `gaelp.journey_funnel` AS
WITH state_counts AS (
  SELECT 
    current_state,
    COUNT(*) as users_in_state,
    COUNT(CASE WHEN converted = TRUE THEN 1 END) as converted_users
  FROM `gaelp.user_journeys`
  WHERE is_active = TRUE OR converted = TRUE
  GROUP BY current_state
)
SELECT 
  current_state,
  users_in_state,
  converted_users,
  SAFE_DIVIDE(converted_users, users_in_state) as conversion_rate,
  LAG(users_in_state) OVER (ORDER BY 
    CASE current_state 
      WHEN 'UNAWARE' THEN 1
      WHEN 'AWARE' THEN 2  
      WHEN 'CONSIDERING' THEN 3
      WHEN 'INTENT' THEN 4
      WHEN 'CONVERTED' THEN 5
    END
  ) as previous_state_users,
  SAFE_DIVIDE(
    users_in_state, 
    LAG(users_in_state) OVER (ORDER BY 
      CASE current_state 
        WHEN 'UNAWARE' THEN 1
        WHEN 'AWARE' THEN 2  
        WHEN 'CONSIDERING' THEN 3
        WHEN 'INTENT' THEN 4
        WHEN 'CONVERTED' THEN 5
      END
    )
  ) as progression_rate
FROM state_counts;

-- =============================================================================
-- Indexes and Constraints
-- =============================================================================

-- Create search indexes for efficient queries
CREATE SEARCH INDEX journey_content_index
ON `gaelp.journey_touchpoints`(ALL COLUMNS);

-- =============================================================================
-- Data Retention Policies
-- =============================================================================

-- 14-day active journey timeout (handled by application)
-- 2 year retention for completed journeys
-- 5 year retention for conversion data
-- 1 year retention for touchpoint details