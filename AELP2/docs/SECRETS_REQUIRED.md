# Required Secrets and Environment Variables

Set via `.env` or your secret manager. Do NOT commit credentials.

Core:
- GOOGLE_CLOUD_PROJECT
- BIGQUERY_TRAINING_DATASET

Google Ads:
- GOOGLE_ADS_DEVELOPER_TOKEN
- GOOGLE_ADS_CLIENT_ID
- GOOGLE_ADS_CLIENT_SECRET
- GOOGLE_ADS_REFRESH_TOKEN
- GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC)
- GOOGLE_ADS_CUSTOMER_ID (10 digits)

GA4:
- GA4_PROPERTY_ID and credentials for BigQuery export (if used) or OAuth/service account for API

KPI:
- AELP2_KPI_CONVERSION_ACTION_IDS (comma-separated)

Feature Flags / HITL:
- AELP2_ALLOW_GOOGLE_MUTATIONS (0/1)
- AELP2_ALLOW_BANDIT_MUTATIONS (0/1)
- AELP2_ALLOW_VALUE_UPLOADS (0/1)
- GATES_FLAGS_JSON (optional local flags JSON)
- GrowthBook/Unleash/Flagsmith keys if used (see core/safety/feature_flags.py)

MMM / Budget:
- AELP2_CAC_CAP
- AELP2_MMM_USE_UPLIFT (0/1)

YouTube Reach Planner:
- AELP2_YT_LOCATION_ID (default US 2840)
- AELP2_YT_BUDGET (USD)
- AELP2_YT_DURATION_DAYS

Enhanced Conversions / Meta CAPI:
- AELP2_VALUE_UPLOAD_DRY_RUN (1 default)
- AELP2_VALUE_UPLOAD_PAYLOAD_REF (audit reference)

If any are missing at runtime, components will fail fast with actionable errors.
