# Data Contracts Catalog (BigQuery)

Purpose: list producer → tables/views → consumer with freshness expectations and ownership.

Note: replace `<project>` with `${GOOGLE_CLOUD_PROJECT}`, `<ds>` with `${BIGQUERY_TRAINING_DATASET}`, `<users>` with `${BIGQUERY_USERS_DATASET}`.

## Core Ads Tables (producers: AELP2/pipelines/google_ads_*_to_bq.py)
- `<project>.<ds>.ads_campaign_performance` (date, campaign_id, impressions, clicks, cost_micros, conversions, conversion_value, impression_share)
  - Consumers: Exec/Finance (/exec, /finance), Freshness API, MMM
  - SLA: daily (T+1); on-demand backfill last 14 days
- `<project>.<ds>.ads_ad_performance` (date, ad_id, metrics)
  - Consumers: Creative Center, Experiments, fatigue alerts
- `<project>.<ds>.ads_conversion_actions`, `<project>.<ds>.ads_conversion_action_stats`
  - Consumers: KPI lock, `ads_kpi_daily` view
- `<project>.<ds>.ads_keywords`, `<project>.<ds>.ads_geo_device`, `<project>.<ds>.ads_search_terms`
  - Consumers: Growth Lab, recommendations scanner

## GA4 Tables (producers: ga4_to_bq.py)
- `<project>.<ds>.ga4_daily` (date, device, channel_group, conversions, revenue)
  - Consumers: Journeys, Exec, halo
- `<project>.<ds>.ga4_export_*` (if native export)
  - Consumers: journeys + advanced pathing

## Attribution / Lag
- `<project>.<ds>.ga4_lagged_attribution` → view `<ds>.ga4_lagged_daily`
  - Consumers: Journeys & Exec lag panel; fidelity eval

## Views (create_bq_views.py)
- `<ds>.ads_kpi_daily` (requires KPI IDs)
- `<ds>.training_episodes_daily`, `<ds>.ads_campaign_daily`

## Experiments (A/B)
- `<ds>.ab_experiments` (definition)
- `<ds>.ab_exposures` (logged via API)
- [MISSING] `<ds>.ab_assignments`, `<ds>.ab_metrics_daily`, view `<ds>.ab_results`

## Bandits / RL
- [MISSING] `<ds>.explore_cells` (cell_key, spend, conv, cac, value, last_seen)
- [MISSING] `<ds>.bandit_posteriors` (cell_key, ts, mean, ci_low, ci_high)
- [MISSING] `<ds>.rl_policy_snapshots` (ts, payload)

## Creative Publishing
- `<ds>.creative_changes` (proposals log)
- [MISSING] `<ds>.creative_publish_queue`, `<ds>.creative_publish_log`, `<ds>.creative_assets`, `<ds>.creative_variants_scored`, `<ds>.creative_policy_flags`

## Landing Pages
- [MISSING] `<ds>.lp_tests`, `<ds>.lp_block_metrics`, `<ds>.funnel_dropoffs`

## Journeys & Halo
- [MISSING] `<ds>.halo_experiments`, `<ds>.halo_reads_daily`, `<ds>.channel_interference_scores`

## Value Uploads
- `<ds>.value_uploads_staging`, `<ds>.value_uploads_log`

## Users Dataset
- `<project>.<users>.canvas_pins` (Chat/Canvas)
- Future: audience exports and admin tables

Ownership
- Data producers: Pipelines team (AELP2/pipelines)
- Consumers: Dashboard APIs, Modeling jobs, Ops
- SLAs & Alerts: Ops; surfaced via `ops_alerts` (add if missing)

