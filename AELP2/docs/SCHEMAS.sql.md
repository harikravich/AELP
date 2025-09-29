# BigQuery Schemas (authoritative DDL)

Note: Replace `${PROJECT}`, `${DATASET}`, `${USERS}` at execution time. Use CREATE OR REPLACE VIEW for views; CREATE TABLE IF NOT EXISTS for tables. Partition time-series tables by DATE(timestamp) or date.

## Experiments (A/B)
```sql
-- Assignments (sticky unit → variant)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.ab_assignments` (
  timestamp TIMESTAMP,
  experiment STRING,
  variant STRING,
  unit_id STRING,
  unit_type STRING,          -- 'ga4_client_id' | 'user_id' | 'cookie' | 'ad_click_id'
  context JSON
) PARTITION BY DATE(timestamp) CLUSTER BY experiment, variant, unit_id;

-- Daily metrics per experiment/variant (aggregated by pipelines)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.ab_metrics_daily` (
  date DATE,
  experiment STRING,
  variant STRING,
  spend FLOAT64,
  clicks INT64,
  conversions INT64,
  revenue FLOAT64,
  cost FLOAT64,
  cac FLOAT64,
  roas FLOAT64
) PARTITION BY date CLUSTER BY experiment, variant;

-- Optional: unified results view (illustrative)
CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.ab_results` AS
SELECT
  m.date, m.experiment, m.variant,
  m.spend, m.clicks, m.conversions, m.revenue, m.cac, m.roas
FROM `${PROJECT}.${DATASET}.ab_metrics_daily` m;
```

## Exploration & RL
```sql
-- Cell coverage/history (Angle×Audience×Channel×LP×Offer)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.explore_cells` (
  cell_key STRING,           -- canonical key: angle|audience|channel|lp|offer
  angle STRING,
  audience STRING,
  channel STRING,
  lp STRING,
  offer STRING,
  last_seen TIMESTAMP,
  spend FLOAT64,
  clicks INT64,
  conversions INT64,
  revenue FLOAT64,
  cac FLOAT64,
  value FLOAT64
) PARTITION BY DATE(last_seen) CLUSTER BY angle, audience, channel;

-- Bandit posteriors per metric
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.bandit_posteriors` (
  ts TIMESTAMP,
  cell_key STRING,
  metric STRING,             -- e.g., 'reward', 'cac', 'ctr', 'cvr'
  mean FLOAT64,
  ci_low FLOAT64,
  ci_high FLOAT64,
  samples INT64
) PARTITION BY DATE(ts) CLUSTER BY metric, cell_key;

-- RL policy snapshots (opaque payload for explain/trace)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.rl_policy_snapshots` (
  ts TIMESTAMP,
  payload JSON
) PARTITION BY DATE(ts);
```

## Creative Publishing & Assets
```sql
-- Queue of publish intents (HITL approved)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.creative_publish_queue` (
  enqueued_at TIMESTAMP,
  run_id STRING,
  platform STRING,           -- 'google_ads' | 'meta' | 'tiktok'
  type STRING,               -- 'rsa' | 'pmax_asset_group' | 'video'
  campaign_id STRING,
  ad_group_id STRING,
  asset_group_id STRING,
  payload JSON,              -- structured asset/ad spec
  status STRING,             -- 'queued' | 'processing' | 'done' | 'error'
  requested_by STRING
) PARTITION BY DATE(enqueued_at) CLUSTER BY platform, type, status;

-- Log of publish executions
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.creative_publish_log` (
  ts TIMESTAMP,
  run_id STRING,
  platform STRING,
  platform_ids JSON,         -- created asset/ad/group IDs
  status STRING,             -- 'paused_created' | 'error'
  policy_topics JSON,        -- disapproval topics if any
  error STRING
) PARTITION BY DATE(ts) CLUSTER BY platform, status;

-- Optional asset registry
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.creative_assets` (
  asset_id STRING,
  type STRING,               -- 'text' | 'image' | 'video'
  text STRING,
  youtube_id STRING,
  image_uri STRING,
  created_at TIMESTAMP,
  policy_flags JSON,
  meta JSON
) PARTITION BY DATE(created_at) CLUSTER BY type;

-- Generated variants with scores & flags
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.creative_variants_scored` (
  variant_id STRING,
  base_asset_id STRING,
  gen_method STRING,         -- 'llm' | 'human' | 'template'
  text STRING,
  image_uri STRING,
  score_ctr FLOAT64,
  score_cvr FLOAT64,
  policy_flags JSON,
  created_at TIMESTAMP,
  ready BOOL
) PARTITION BY DATE(created_at) CLUSTER BY gen_method, ready;

-- Policy flags detail
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.creative_policy_flags` (
  asset_id STRING,
  reasons ARRAY<STRING>,
  topics JSON,
  blocked BOOL,
  reviewed_by STRING,
  created_at TIMESTAMP
) PARTITION BY DATE(created_at);
```

## Landing Pages & Funnels
```sql
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.lp_tests` (
  test_id STRING,
  created_at TIMESTAMP,
  lp_a STRING,
  lp_b STRING,
  status STRING,             -- 'draft' | 'running' | 'stopped'
  traffic_split FLOAT64,
  primary_metric STRING
) PARTITION BY DATE(created_at);

CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.lp_block_metrics` (
  date DATE,
  lp_url STRING,
  block STRING,              -- 'hero' | 'sticky_cta' | 'faq' | ...
  metric STRING,             -- 'view_time' | 'scroll' | 'click'
  value FLOAT64
) PARTITION BY date CLUSTER BY lp_url, block;

CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.funnel_dropoffs` (
  date DATE,
  lp_url STRING,
  stage STRING,              -- 'view' | 'engage' | 'form_start' | 'form_submit' | 'purchase'
  visitors INT64,
  drop_rate FLOAT64
) PARTITION BY date CLUSTER BY lp_url, stage;
```

## Landing Page Modules (Proof Blocks)
```sql
-- Per-run log for a module execution on an LP (consent required)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.lp_module_runs` (
  run_id STRING,
  slug STRING,               -- e.g., 'insight_preview' | 'scam_check' | 'privacy_check'
  page_url STRING,
  consent_id STRING,
  created_ts TIMESTAMP,
  status STRING,             -- 'queued' | 'running' | 'done' | 'error'
  elapsed_ms INT64,
  error_code STRING
) PARTITION BY DATE(created_ts) CLUSTER BY slug, status;

-- Consent acknowledgements (no raw PII stored)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.consent_logs` (
  consent_id STRING,
  slug STRING,
  page_url STRING,
  consent_text STRING,
  ip_hash STRING,
  user_agent STRING,
  ts TIMESTAMP
) PARTITION BY DATE(ts) CLUSTER BY slug;

-- Sanitized result payloads (short retention recommended)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.module_results` (
  run_id STRING,
  slug STRING,
  summary_text STRING,
  result_json JSON,
  expires_at TIMESTAMP
) PARTITION BY DATE(expires_at) CLUSTER BY slug;
```

## Halo & Interference
```sql
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.halo_experiments` (
  exp_id STRING,
  channel STRING,
  geo STRING,
  start DATE,
  end_date DATE,
  treatment_share FLOAT64,
  status STRING
) PARTITION BY start CLUSTER BY channel, geo;

CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.halo_reads_daily` (
  date DATE,
  exp_id STRING,
  brand_lift FLOAT64,
  ci_low FLOAT64,
  ci_high FLOAT64,
  method STRING             -- 'GeoLift' | 'CausalImpact' | ...
) PARTITION BY date CLUSTER BY exp_id;

CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.channel_interference_scores` (
  date DATE,
  from_channel STRING,
  to_channel STRING,
  cannibalization FLOAT64,
  lift FLOAT64
) PARTITION BY date CLUSTER BY from_channel, to_channel;
```

## Users Dataset (Canvas & Admin)
```sql
CREATE TABLE IF NOT EXISTS `${PROJECT}.${USERS}.canvas_pins` (
  ts TIMESTAMP,
  user_email STRING,
  pin_id STRING,
  payload JSON
) PARTITION BY DATE(ts) CLUSTER BY user_email;
```

## Research & Channel Discovery
```sql
-- Candidate channels discovered by the Research Agent
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.channel_candidates` (
  id STRING,
  name STRING,
  type STRING,               -- 'search' | 'social' | 'native' | 'email' | 'CTV' | 'other'
  status STRING,             -- 'new' | 'triage' | 'pilot_pending' | 'pilot' | 'live' | 'archived'
  audience_fit_notes STRING,
  use_cases ARRAY<STRING>,
  pricing_model STRING,      -- 'CPC' | 'CPM' | 'CPL' | 'CPA'
  typical_cpc FLOAT64,
  min_budget FLOAT64,
  formats ARRAY<STRING>,     -- e.g., ['text','image','video']
  targeting JSON,            -- geo/device/age/interests
  api_available BOOL,
  docs_url STRING,
  auth_type STRING,          -- 'OAuth' | 'API_KEY' | 'CSV'
  export_mode STRING,        -- 'API' | 'CSV' | 'Manual'
  measurability JSON,        -- {utms_supported:bool, offline_supported:bool}
  risk_notes STRING,
  bot_fraud_risk STRING,
  effort_estimate STRING,    -- 'S' | 'M' | 'L'
  integration_steps JSON,
  score_fit FLOAT64,
  score_cost FLOAT64,
  score_measure FLOAT64,
  score_effort FLOAT64,
  score_risk FLOAT64,
  score_total FLOAT64,
  citations ARRAY<JSON>,     -- [{title,url}]
  created_by STRING,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
) PARTITION BY DATE(created_at) CLUSTER BY status, type;

-- Free‑form findings per candidate (research notes with citations)
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.research_findings` (
  ts TIMESTAMP,
  candidate_id STRING,
  summary STRING,
  details JSON,              -- structured bullets
  citations ARRAY<JSON>,     -- [{title,url}]
  source STRING              -- 'perplexity' | 'manual' | 'other'
) PARTITION BY DATE(ts) CLUSTER BY candidate_id, source;
```

## Research (Perplexity‑powered)
```sql
CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.research_angle_candidates` (
  created_at TIMESTAMP,
  use_case STRING,
  angle STRING,
  audience STRING,
  channel STRING,
  lp STRING,
  offer STRING,
  rationale STRING,
  expected_cac_min FLOAT64,
  expected_cac_max FLOAT64,
  sources JSON
) PARTITION BY DATE(created_at) CLUSTER BY use_case, angle;

CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.research_findings` (
  created_at TIMESTAMP,
  question STRING,
  summary STRING,
  sources JSON
) PARTITION BY DATE(created_at);

CREATE TABLE IF NOT EXISTS `${PROJECT}.${DATASET}.creative_briefs` (
  created_at TIMESTAMP,
  use_case STRING,
  angle STRING,
  hook STRING,
  copy_guidelines STRING,
  policy_flags JSON,
  examples JSON
) PARTITION BY DATE(created_at) CLUSTER BY use_case, angle;
```
