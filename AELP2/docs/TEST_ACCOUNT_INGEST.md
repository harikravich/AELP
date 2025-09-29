# Test Account Ingestion Flow

Prereqs:
- Set `GOOGLE_ADS_LOGIN_CUSTOMER_ID=<MCC_ID>` on the dashboard service.
- Use Sandbox dataset (dataset switcher) to avoid prod writes.

Steps:
1. Open Control page → "Ads Ingestion" block.
2. Enter personal 10-digit customer ID in the `only` field and click Run.
3. The handler spawns `AELP2/scripts/run_ads_ingestion.sh --last14` in background.
4. Verify tables in `${BIGQUERY_TRAINING_DATASET}`: `ads_campaign_performance`, `ads_ad_performance`, `ads_conversion_actions`, `ads_conversion_action_stats`.

Notes:
- Writes are blocked on prod dataset by `resolveDatasetForAction('write')` guard.
- Use MCC limited scopes and test accounts only.
