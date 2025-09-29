# Permissions & Accounts Checklist

Required before T-1 (no later than 2025-09-09):
- Google Ads developer token: Standard access (production)
- OAuth refresh token tied to MCC with access to target `customer_id`
- GA4 Data API access: property scope `analytics.readonly` (or GA4→BQ export enabled)
- GCP roles on project `${GOOGLE_CLOUD_PROJECT}` for service account:
  - BigQuery Data Viewer, Job User (dataset `${BIGQUERY_TRAINING_DATASET}` and `gaelp_users`)
  - (Optional) Storage Object Viewer for asset mirroring
- Env vars set on runner: `GOOGLE_CLOUD_PROJECT`, `BIGQUERY_TRAINING_DATASET`, `GA4_PROPERTY_ID` (if using Data API)

Verification script: `python3 -m AELP2.pipelines.permissions_check`
