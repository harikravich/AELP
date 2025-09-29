Data Ingestion into BigQuery

Goal: Load historical data from Google Ads and GA4 for calibration and monitoring.

Prereqs
- ADC: gcloud auth application-default login
- Env in .env or AELP2/config/.env.aelp2:
  - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET, BIGQUERY_DATASET_LOCATION (optional)
  - Google Ads: GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_CUSTOMER_ID (10 digits)
  - GA4: GA4_PROPERTY_ID (format: properties/<id>)
  - Optional GA4 export: GA4_EXPORT_DATASET (e.g., analytics_308028264)

Tables Produced
- `${BIGQUERY_TRAINING_DATASET}.ads_campaign_performance` (date-partitioned optional; fields include date, campaign_id, impressions, clicks, cost_micros, conversions, conversion_value, ctr, avg_cpc_micros, impression_share)
- `${BIGQUERY_TRAINING_DATASET}.ga4_aggregates` (date, device_category, default_channel_group, sessions, screen_page_views, conversions, users)
- `${BIGQUERY_TRAINING_DATASET}.ads_ad_performance` (date, customer_id, campaign/ad_group/ad ids, names hashed, impressions/clicks/cost/conversions/value/ctr/cpc)

Additional Ads Tables (comprehensive ingestion)
- `${BIGQUERY_TRAINING_DATASET}.ads_keyword_performance` (partitioned by date)
  - date, customer_id, campaign_id(+hash), ad_group_id(+hash), criterion_id, keyword_text_hash (redacted), match_type, impressions/clicks/cost/conversions/value/ctr/cpc
- `${BIGQUERY_TRAINING_DATASET}.ads_search_terms` (partitioned by date)
  - date, customer_id, campaign_id, ad_group_id, search_term_hash (redacted), impressions/clicks/cost/conversions/value/ctr/cpc
  - Note: raw search terms are hashed by default (`AELP2_REDACT_TEXT=1`)
- `${BIGQUERY_TRAINING_DATASET}.ads_conversion_action_stats` (partitioned by date)
  - date, customer_id, campaign_id, conversion_action_id, conversion_action_name (redacted), conversions, conversion_value, cost_micros
  - Load per customer (use MCC coordinator or per‑ID):
    - python -m AELP2.pipelines.google_ads_conversion_stats_by_action_to_bq --start 2024-07-01 --end 2024-07-31 --customer 7844126439

Commands
- Load date range (both):
  - bash AELP2/scripts/load_history.sh 2024-06-01 2024-08-31
- Ads only:
  - python -m AELP2.pipelines.google_ads_to_bq --start 2024-06-01 --end 2024-08-31
- Enumerate all client accounts under MCC and load campaign performance:
  - python -m AELP2.pipelines.google_ads_mcc_to_bq --start 2024-06-01 --end 2024-08-31
- Ads keywords for a specific customer:
  - python -m AELP2.pipelines.google_ads_keywords_to_bq --start 2024-06-01 --end 2024-08-31 --customer 7844126439
- Ads search terms for a specific customer:
  - python -m AELP2.pipelines.google_ads_search_terms_to_bq --start 2024-06-01 --end 2024-08-31 --customer 7844126439
- Ads ad performance for a specific customer:
  - python -m AELP2.pipelines.google_ads_ad_performance_to_bq --start 2024-06-01 --end 2024-08-31 --customer 7844126439
  - Tip: omit --customer to use GOOGLE_ADS_CUSTOMER_ID from env
- GA4 only (requires GA4_PROPERTY_ID):
  - python -m AELP2.pipelines.ga4_to_bq --start 2024-06-01 --end 2024-08-31
  - or: bash AELP2/scripts/run_ga4_ingestion.sh --last28
  - Auth options:
    - ADC (user): `gcloud auth application-default login --no-launch-browser --scopes=https://www.googleapis.com/auth/analytics.readonly,https://www.googleapis.com/auth/cloud-platform`
    - OAuth client refresh token (reuse your Google Web OAuth client): set
      `GA4_OAUTH_CLIENT_ID`, `GA4_OAUTH_CLIENT_SECRET`, `GA4_OAUTH_REFRESH_TOKEN` (must include `analytics.readonly` scope)

- MCC coordinator (all child accounts; quota-aware serial loads):
  - python -m AELP2.pipelines.ads_mcc_coordinator --start 2024-07-01 --end 2024-07-31 --tasks campaigns,keywords,search_terms,geo_device,adgroups,conversion_actions
  - Optional: export AELP2_ADS_MCC_DELAY_SECONDS=2.0
  - List MCC child accounts only:
    - python -m AELP2.pipelines.google_ads_mcc_to_bq --start 2024-07-01 --end 2024-07-31 --list-only
  - Include ad performance in MCC loads:
    - python -m AELP2.pipelines.ads_mcc_coordinator --start 2024-07-01 --end 2024-07-31 --tasks ad_performance

Backfills
- 3‑year Ads backfill (monthly windows across MCC):
  - bash AELP2/scripts/run_ads_backfill.sh --mcc <MCC_ID> --tasks "campaigns,ad_performance,keywords,search_terms,geo_device,adgroups,conversion_actions"
  - Add `--dry-run` to preview windows; use `--start`/`--end` to customize range

Views
- Create core BQ views for dashboards/subagents:
  - python -m AELP2.pipelines.create_bq_views
  - If `GA4_EXPORT_DATASET` is set and accessible, a `ga4_export_daily` staging view is created

New
- `${BIGQUERY_TRAINING_DATASET}.ads_conversion_actions` (definitions)
  - id, name_hash (redacted), category, type, status, primary_for_goal, customer_id, loaded_at
  - Load:
    - python -m AELP2.pipelines.google_ads_conversion_actions_to_bq

Notes
- For raw GA4 events, enable native GA4→BigQuery export in GA4 Admin; this script provides aggregated metrics only.
- The orchestrator’s calibration reference builder uses `ads_campaign_performance.impression_share` to build `AELP2_CALIBRATION_REF_JSON` on first run.
- No dummy data; missing credentials will produce actionable errors.
