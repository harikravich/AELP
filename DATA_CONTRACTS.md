# Data Contracts and Schemas

Purpose: canonical schemas and mappings for GA4, Ads, attribution/touchpoints, simulator calibration, and training telemetry.

## Environment
- Project: `aura-thrive-platform`
- Datasets:
  - Users: `gaelp_users` (persistent identity/journeys/touchpoints)
  - Training: `gaelp_training` (episodes/metrics/A\B/safety)
- Storage: GCS bucket for model artifacts/checkpoints

## GA4 Ingestion (Staging Tables)
- Table: `gaelp_training.ga4_events_raw`
  - event_timestamp: TIMESTAMP
  - event_name: STRING (e.g., purchase, sign_up, view_item, page_view)
  - user_pseudo_id: STRING
  - session_id: STRING
  - device_category: STRING (mobile, desktop, tablet)
  - default_channel_group: STRING (organic, paid_search, social, display, email, direct)
  - page_path: STRING
  - campaign: STRING
  - source: STRING
  - medium: STRING
  - value: FLOAT64 (if present)
  - currency: STRING
  - custom_params: JSON

- Table: `gaelp_training.ga4_sessions`
  - session_id: STRING (PK)
  - user_pseudo_id: STRING
  - session_start: TIMESTAMP
  - session_end: TIMESTAMP
  - device_category: STRING
  - default_channel_group: STRING
  - pages_per_session: FLOAT64
  - average_session_duration: FLOAT64 (seconds)
  - bounces: INT64

- Table: `gaelp_training.ga4_aggregates`
  - date: DATE
  - device_category: STRING
  - default_channel_group: STRING
  - hour: INT64
  - sessions: INT64
  - screen_page_views: INT64
  - conversions: INT64
  - users: INT64

Mapping: GA4 API responses → staged via jobs; then transformed to downstream tables below.

## Ads Performance (Google)
- Table: `gaelp_training.ads_campaign_performance`
  - date: DATE
  - campaign_id: STRING
  - campaign_name: STRING
  - impressions: INT64
  - clicks: INT64
  - cost_micros: INT64
  - conversions: FLOAT64
  - conversion_value: FLOAT64
  - ctr: FLOAT64
  - avg_cpc_micros: INT64
  - impression_share: FLOAT64

- Table: `gaelp_training.ads_keyword_performance`
  - date: DATE
  - campaign_id: STRING
  - ad_group_id: STRING
  - keyword: STRING
  - impressions: INT64
  - clicks: INT64
  - cost_micros: INT64
  - conversions: FLOAT64
  - conversion_value: FLOAT64
  - ctr: FLOAT64
  - avg_cpc_micros: INT64

## Persistent Users/Journeys (Users Dataset)
- Table: `gaelp_users.persistent_users`
  - user_id: STRING (PK)
  - canonical_user_id: STRING
  - device_ids: ARRAY<STRING>
  - email_hash: STRING
  - phone_hash: STRING
  - current_journey_state: STRING (enum)
  - awareness_level: FLOAT64
  - fatigue_score: FLOAT64
  - intent_score: FLOAT64
  - last_seen: TIMESTAMP

- Table: `gaelp_users.journey_sessions`
  - session_id: STRING (PK)
  - user_id: STRING
  - start_time: TIMESTAMP
  - end_time: TIMESTAMP
  - device_category: STRING
  - channel: STRING
  - pages_viewed: INT64
  - time_on_site: INT64
  - bounced: BOOL

- Table: `gaelp_users.persistent_touchpoints`
  - touchpoint_id: STRING (PK)
  - user_id: STRING
  - session_id: STRING
  - timestamp: TIMESTAMP
  - touchpoint_type: STRING (impression, click, visit, conversion)
  - channel: STRING
  - source: STRING
  - medium: STRING
  - campaign: STRING
  - ad_group: STRING
  - creative_id: STRING
  - keyword: STRING
  - click_id: STRING
  - attribution_window_days: INT64
  - attributed_value: FLOAT64
  - attribution_model: STRING
  - conversion_value: FLOAT64
  - conversion_type: STRING (trial, purchase, subscription)
  - product_category: STRING
  - is_privacy_restricted: BOOL
  - tracking_method: STRING (client, server, hybrid)
  - data_quality: FLOAT64

- Table: `gaelp_users.competitor_exposures`
  - exposure_id: STRING (PK)
  - user_id: STRING
  - timestamp: TIMESTAMP
  - competitor_name: STRING
  - impression_count: INT64
  - impact_score: FLOAT64

## Attribution / Reward Module (Shared Sim & Live)
- Unified function calculates rewards = conversion_value − spend with delayed credit via MTA.
- Inputs:
  - Touchpoints from `gaelp_users.persistent_touchpoints`
  - Conversion events from GA4/Ads (mapped to touchpoints via session_id, click_id, user_id, lag windows)
- Supported models: Linear, Position-based, Time Decay, Data-driven
- Output: per-decision attributed_reward per step; aggregated at episode end.

## Simulator Calibration (Materialized Views)
- View: `gaelp_training.sim_auction_params`
  - channel, segment, device, hour
  - win_rate_curve: JSON (bid→win_rate)
  - cpc_distribution: JSON (params)
  - position_distribution: JSON (params)

- View: `gaelp_training.sim_user_params`
  - segment, channel, device
  - session_duration_params: JSON
  - revisit_params: JSON
  - fatigue_params: JSON
  - temporal_activity: JSON

- View: `gaelp_training.sim_creative_params`
  - features: JSON (content signals)
  - ctr_cvr_params: JSON

- View: `gaelp_training.sim_conversion_lag_params`
  - segment, channel, creative
  - lag_distribution: JSON

## Training Telemetry (Training Dataset)
- Table: `gaelp_training.training_episodes`
  - timestamp: TIMESTAMP
  - episode: INT64
  - steps: INT64
  - spend: FLOAT64
  - revenue: FLOAT64
  - conversions: INT64
  - win_rate: FLOAT64
  - avg_cpc: FLOAT64
  - epsilon: FLOAT64

- Table: `gaelp_training.training_metrics` (optional granular)
  - timestamp: TIMESTAMP
  - episode: INT64
  - metric_name: STRING (loss, grad_norm, q_overestimation, etc.)
  - metric_value: FLOAT64

- Table: `gaelp_training.ab_test_results`
  - test_id: STRING
  - variant: STRING
  - start_time: TIMESTAMP
  - end_time: TIMESTAMP
  - impressions: INT64
  - clicks: INT64
  - conversions: FLOAT64
  - conversion_rate: FLOAT64
  - lift: FLOAT64
  - p_value: FLOAT64

- Table: `gaelp_training.safety_events`
  - timestamp: TIMESTAMP
  - component: STRING
  - severity: STRING
  - message: STRING
  - metadata: JSON

- Table: `gaelp_training.checkpoints`
  - checkpoint_id: STRING
  - created_at: TIMESTAMP
  - model_version: STRING
  - path: STRING (GCS)
  - metrics_snapshot: JSON

## Normalized Action Schema (for PlatformAdapter)
- action_id: STRING
- timestamp: TIMESTAMP
- platform: STRING (google_ads, meta, tiktok)
- campaign: STRING
- ad_group: STRING
- bid: FLOAT64 (or bid_modifier)
- budget_adjustment: FLOAT64
- creative_id: STRING
- audience/segment: STRING
- placement/channel: STRING
- constraints: JSON (caps, pacing)

## KPI Normalization (Cross-Platform)
- Unified KPIs for training and dashboards:
  - spend, revenue, conversions, CAC, ROAS, win_rate, CTR, CVR, avg_cpc
- Platform-specific metrics mapped to unified schema during ingestion.

