-- GAELP Persistent User Database Schema
-- CRITICAL INFRASTRUCTURE: Users that NEVER reset between episodes

-- Create the gaelp_users dataset
CREATE SCHEMA IF NOT EXISTS `aura-thrive-platform.gaelp_users`
OPTIONS (
  description = "GAELP Persistent User Database - Solves user reset fundamental flaw",
  location = "US",
  labels = [("system", "gaelp"), ("critical", "true")]
);

-- =====================================================================
-- PERSISTENT USERS TABLE
-- Core table tracking users across episodes with state persistence
-- =====================================================================

CREATE TABLE IF NOT EXISTS `aura-thrive-platform.gaelp_users.persistent_users` (
    -- Identity
    user_id STRING NOT NULL,
    canonical_user_id STRING NOT NULL,
    device_ids ARRAY<STRING>,
    email_hash STRING,
    phone_hash STRING,
    
    -- Journey state (persists across episodes - CRITICAL)
    current_journey_state STRING NOT NULL DEFAULT "UNAWARE",
    awareness_level FLOAT64 DEFAULT 0.0,
    fatigue_score FLOAT64 DEFAULT 0.0,
    intent_score FLOAT64 DEFAULT 0.0,
    
    -- Competitor tracking (persists across episodes)
    competitor_exposures JSON,
    competitor_fatigue JSON,
    
    -- Cross-device tracking
    devices_seen JSON,
    cross_device_confidence FLOAT64 DEFAULT 0.0,
    
    -- Time tracking (NEVER resets)
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    last_episode STRING,
    episode_count INT64 DEFAULT 0,
    
    -- Journey history (accumulates over time)
    journey_history JSON,
    touchpoint_history JSON,
    conversion_history JSON,
    
    -- Timeout management (14-day rule)
    timeout_at TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(last_seen)
CLUSTER BY canonical_user_id, is_active
OPTIONS (
    description = "Persistent users that maintain state across episodes - CRITICAL",
    labels = [("table_type", "persistent_users")]
);

-- =====================================================================
-- JOURNEY SESSIONS TABLE  
-- Individual journey sessions within persistent user lifetimes
-- =====================================================================

CREATE TABLE IF NOT EXISTS `aura-thrive-platform.gaelp_users.journey_sessions` (
    -- Session identity
    session_id STRING NOT NULL,
    user_id STRING NOT NULL,
    canonical_user_id STRING NOT NULL,
    episode_id STRING NOT NULL,
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP,
    
    -- Session state tracking
    session_state_changes JSON,
    session_touchpoints ARRAY<STRING>,
    session_channels ARRAY<STRING>,
    session_devices ARRAY<STRING>,
    
    -- Session outcomes
    converted_in_session BOOLEAN DEFAULT FALSE,
    conversion_value FLOAT64,
    session_engagement FLOAT64 DEFAULT 0.0,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(session_start)
CLUSTER BY canonical_user_id, episode_id
OPTIONS (
    description = "Journey sessions within persistent user lifetimes",
    labels = [("table_type", "journey_sessions")]
);

-- =====================================================================
-- PERSISTENT TOUCHPOINTS TABLE
-- All touchpoints with proper attribution tracking
-- =====================================================================

CREATE TABLE IF NOT EXISTS `aura-thrive-platform.gaelp_users.persistent_touchpoints` (
    -- Touchpoint identity
    touchpoint_id STRING NOT NULL,
    user_id STRING NOT NULL,
    canonical_user_id STRING NOT NULL,
    session_id STRING NOT NULL,
    episode_id STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Touchpoint details
    channel STRING NOT NULL,
    campaign_id STRING,
    creative_id STRING,
    device_type STRING,
    
    -- State impact (tracks changes to persistent user state)
    pre_state STRING NOT NULL DEFAULT "UNAWARE",
    post_state STRING NOT NULL DEFAULT "UNAWARE", 
    state_change_confidence FLOAT64 DEFAULT 0.0,
    
    -- Engagement metrics
    engagement_score FLOAT64 DEFAULT 0.0,
    dwell_time FLOAT64,
    interaction_depth INT64 DEFAULT 0,
    
    -- Multi-touch attribution weights
    attribution_weights JSON,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY canonical_user_id, channel
OPTIONS (
    description = "All touchpoints with state impact and attribution tracking",
    labels = [("table_type", "persistent_touchpoints")]
);

-- =====================================================================
-- COMPETITOR EXPOSURES TABLE
-- Track competitor interactions and impact on user fatigue
-- =====================================================================

CREATE TABLE IF NOT EXISTS `aura-thrive-platform.gaelp_users.competitor_exposures` (
    -- Exposure identity
    exposure_id STRING NOT NULL,
    user_id STRING NOT NULL,
    canonical_user_id STRING NOT NULL,
    session_id STRING,
    episode_id STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Competitor details
    competitor_name STRING NOT NULL,
    competitor_channel STRING NOT NULL,
    exposure_type STRING NOT NULL,
    
    -- Impact tracking
    pre_exposure_state STRING NOT NULL,
    impact_score FLOAT64 DEFAULT 0.0,
    fatigue_increase FLOAT64 DEFAULT 0.0,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY canonical_user_id, competitor_name
OPTIONS (
    description = "Competitor exposures and their impact on user journey state",
    labels = [("table_type", "competitor_exposures")]
);

-- =====================================================================
-- VIEWS FOR ANALYTICS AND MONITORING
-- =====================================================================

-- Active users view
CREATE OR REPLACE VIEW `aura-thrive-platform.gaelp_users.active_persistent_users` AS
SELECT 
    canonical_user_id,
    user_id,
    current_journey_state,
    awareness_level,
    fatigue_score,
    intent_score,
    episode_count,
    ARRAY_LENGTH(device_ids) as device_count,
    DATE_DIFF(CURRENT_DATE(), DATE(first_seen), DAY) as days_active,
    DATE_DIFF(CURRENT_DATE(), DATE(last_seen), DAY) as days_since_last_seen,
    DATE_DIFF(DATE(timeout_at), CURRENT_DATE(), DAY) as days_until_timeout
FROM `aura-thrive-platform.gaelp_users.persistent_users`
WHERE is_active = TRUE
ORDER BY last_seen DESC;

-- User state distribution
CREATE OR REPLACE VIEW `aura-thrive-platform.gaelp_users.user_state_distribution` AS
SELECT 
    current_journey_state,
    COUNT(*) as user_count,
    AVG(awareness_level) as avg_awareness,
    AVG(fatigue_score) as avg_fatigue,
    AVG(intent_score) as avg_intent,
    AVG(episode_count) as avg_episodes
FROM `aura-thrive-platform.gaelp_users.persistent_users`
WHERE is_active = TRUE
GROUP BY current_journey_state
ORDER BY user_count DESC;

-- Episode persistence analysis
CREATE OR REPLACE VIEW `aura-thrive-platform.gaelp_users.episode_persistence_stats` AS
SELECT 
    episode_count,
    COUNT(*) as users_with_episode_count,
    AVG(DATE_DIFF(CURRENT_DATE(), DATE(first_seen), DAY)) as avg_days_active,
    COUNT(CASE WHEN ARRAY_LENGTH(conversion_history) > 0 THEN 1 END) as users_with_conversions
FROM `aura-thrive-platform.gaelp_users.persistent_users`
WHERE is_active = TRUE
GROUP BY episode_count
ORDER BY episode_count;

-- Cross-device analysis
CREATE OR REPLACE VIEW `aura-thrive-platform.gaelp_users.cross_device_analysis` AS
SELECT 
    ARRAY_LENGTH(device_ids) as device_count,
    COUNT(*) as user_count,
    AVG(cross_device_confidence) as avg_confidence,
    AVG(episode_count) as avg_episodes
FROM `aura-thrive-platform.gaelp_users.persistent_users`
WHERE is_active = TRUE AND ARRAY_LENGTH(device_ids) IS NOT NULL
GROUP BY device_count
ORDER BY device_count;

-- Competitor impact analysis
CREATE OR REPLACE VIEW `aura-thrive-platform.gaelp_users.competitor_impact_summary` AS
SELECT 
    competitor_name,
    competitor_channel,
    COUNT(*) as exposure_count,
    COUNT(DISTINCT canonical_user_id) as unique_users_exposed,
    AVG(impact_score) as avg_impact_score,
    AVG(fatigue_increase) as avg_fatigue_increase,
    DATE(timestamp) as exposure_date
FROM `aura-thrive-platform.gaelp_users.competitor_exposures`
WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY competitor_name, competitor_channel, DATE(timestamp)
ORDER BY exposure_date DESC, exposure_count DESC;

-- =====================================================================
-- INDEXES AND PERFORMANCE OPTIMIZATION
-- =====================================================================

-- Create search indexes for common queries
-- (BigQuery automatically optimizes based on clustering, but we can add explicit indexes if needed)

-- =====================================================================
-- DATA RETENTION POLICIES
-- =====================================================================

-- Set up automatic cleanup for old data
-- Inactive users after 1 year
CREATE OR REPLACE TABLE FUNCTION `aura-thrive-platform.gaelp_users.cleanup_inactive_users`()
AS (
  SELECT canonical_user_id
  FROM `aura-thrive-platform.gaelp_users.persistent_users`
  WHERE is_active = FALSE 
  AND DATE_DIFF(CURRENT_DATE(), DATE(updated_at), DAY) > 365
);

-- Archive old touchpoints after 2 years
CREATE OR REPLACE TABLE FUNCTION `aura-thrive-platform.gaelp_users.archive_old_touchpoints`()
AS (
  SELECT *
  FROM `aura-thrive-platform.gaelp_users.persistent_touchpoints`
  WHERE DATE_DIFF(CURRENT_DATE(), DATE(timestamp), DAY) > 730
);

-- =====================================================================
-- MONITORING AND ALERTS
-- =====================================================================

-- Query for monitoring user reset issues (should return 0)
CREATE OR REPLACE VIEW `aura-thrive-platform.gaelp_users.user_reset_detection` AS
SELECT 
    canonical_user_id,
    COUNT(*) as duplicate_first_seen_count,
    STRING_AGG(DISTINCT user_id) as all_user_ids
FROM `aura-thrive-platform.gaelp_users.persistent_users`
WHERE is_active = TRUE
GROUP BY canonical_user_id, DATE(first_seen)
HAVING COUNT(*) > 1
ORDER BY duplicate_first_seen_count DESC;

-- Query for detecting timeout issues
CREATE OR REPLACE VIEW `aura-thrive-platform.gaelp_users.timeout_monitoring` AS
SELECT 
    COUNT(CASE WHEN timeout_at <= CURRENT_TIMESTAMP() AND is_active = TRUE THEN 1 END) as users_past_timeout,
    COUNT(CASE WHEN DATE_DIFF(DATE(timeout_at), CURRENT_DATE(), DAY) <= 1 AND is_active = TRUE THEN 1 END) as users_expiring_soon,
    COUNT(CASE WHEN is_active = TRUE THEN 1 END) as total_active_users,
    COUNT(CASE WHEN is_active = FALSE THEN 1 END) as total_inactive_users
FROM `aura-thrive-platform.gaelp_users.persistent_users`;

-- =====================================================================
-- GRANT PERMISSIONS
-- =====================================================================

-- Grant access to GAELP service accounts
-- GRANT SELECT ON `aura-thrive-platform.gaelp_users.*` TO 'serviceAccount:gaelp-training@aura-thrive-platform.iam.gserviceaccount.com';
-- GRANT INSERT ON `aura-thrive-platform.gaelp_users.*` TO 'serviceAccount:gaelp-training@aura-thrive-platform.iam.gserviceaccount.com';
-- GRANT UPDATE ON `aura-thrive-platform.gaelp_users.*` TO 'serviceAccount:gaelp-training@aura-thrive-platform.iam.gserviceaccount.com';

-- =====================================================================
-- DOCUMENTATION AND COMMENTS
-- =====================================================================

/*
PERSISTENT USER DATABASE SCHEMA - CRITICAL SYSTEM

This schema solves the FUNDAMENTAL FLAW where users were resetting between episodes,
making all RL learning invalid. 

KEY FEATURES:
1. Users NEVER reset between episodes
2. Journey states accumulate over 3-14 days
3. Cross-device identity resolution 
4. 14-day timeout for unconverted users
5. Full BigQuery integration with NO FALLBACKS

TABLES:
- persistent_users: Core user state that persists across episodes
- journey_sessions: Individual sessions within user lifetimes
- persistent_touchpoints: All touchpoints with state impact tracking
- competitor_exposures: Competitor interactions and fatigue tracking

PARTITIONING:
- All tables partitioned by date for performance
- Clustered by canonical_user_id for efficient user queries

MONITORING:
- Views for detecting user reset issues (should be empty)
- Timeout monitoring and cleanup
- Cross-device analysis
- Competitor impact tracking

This schema is CRITICAL for GAELP's RL learning system.
Without persistent users, all learning is invalid.
*/