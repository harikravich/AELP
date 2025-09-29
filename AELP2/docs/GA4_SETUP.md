# GA4 Setup for AELP2 (Data API + BigQuery Export)

Purpose: enable GA4 data for calibration, monitoring, and IRL fidelity checks.

## Option A: Native GA4 → BigQuery Export (Recommended)
1) GA4 Admin → Property → BigQuery Links → Link your project `GOOGLE_CLOUD_PROJECT`.
2) Choose daily export (and streaming if needed).
3) Data lands in dataset `analytics_<property_id>`.
4) Create staging views in `${BIGQUERY_TRAINING_DATASET}` for sessions/events/conversions.

## Option B: GA4 Data API (Aggregates)
Used until native export is active; populates `${BIGQUERY_TRAINING_DATASET}.ga4_aggregates`.

Prereqs
- Enable API: Google Analytics Data API on your GCP project.
- Auth: Application Default Credentials (`gcloud auth application-default login`) or Service Account with key.
- GA4 Property Access: add the caller (user or SA email) as ‘Analyst’ in GA4 Property Access Management.

Env
- `GA4_PROPERTY_ID=properties/<id>`

Run Loader
- `python3 -m AELP2.pipelines.ga4_to_bq --start 2024-06-01 --end 2024-08-31`

Troubleshooting
- 403 ACCESS_TOKEN_SCOPE_INSUFFICIENT: caller lacks GA4 property access or ADC missing; fix as above.
- Quotas: respect Data API limits (batch date ranges if needed).

Security
- Load only aggregates via Data API; for raw events, rely on native export.
- IAM least privilege: dataset-level viewer for readers, editor for loaders.

