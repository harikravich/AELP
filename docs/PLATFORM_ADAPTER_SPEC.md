# Platform Adapter Specification (Scaffold)

Purpose: define a normalized action schema and adapter interface to support Google, Meta, TikTok, etc.

## Normalized Action (input to adapter)
- platform: STRING (google_ads, meta, tiktok)
- campaign: STRING
- ad_group: STRING
- bid: FLOAT64 (or bid_modifier)
- budget_adjustment: FLOAT64
- creative_id: STRING
- audience/segment: STRING
- placement/channel: STRING
- constraints: JSON (caps, pacing)

## Adapter Interface
- apply_action(action) -> result/status
- get_metrics(filters) -> performance snapshot
- health_check() -> bool
- map_creative(creative_id) -> platform creative
- ensure_campaign_structure(campaign, ad_group) -> created/verified

## KPIs and Mapping
- Unified KPIs: spend, revenue, conversions, CAC, ROAS, win_rate, CTR, CVR, avg_cpc
- Responsibility: adapters map platform-native metrics to unified KPIs for the training plane.

