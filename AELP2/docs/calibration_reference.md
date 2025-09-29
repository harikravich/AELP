Calibration Reference Builder

Purpose: produce a JSON file of win_rate samples used to validate auction calibration.

Source: BigQuery table `${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.ads_campaign_performance` using column `impression_share` as the win_rate proxy.

Usage (env):
- `GOOGLE_CLOUD_PROJECT`
- `BIGQUERY_TRAINING_DATASET`
- `AELP2_CALIBRATION_REF_JSON` (e.g., `AELP2/.refs/win_rates.json`)
- `AELP2_CALIBRATION_REF_DAYS` (optional, default 30)
- `AELP2_CALIBRATION_MAX_KS`, `AELP2_CALIBRATION_MAX_HIST_MSE`

Behavior:
- If `AELP2_CALIBRATION_REF_JSON` is set but the file does not exist, the orchestrator will attempt to build it via BigQuery and save to that path.
- If BigQuery is not configured or table is missing, calibration fails with a clear error.

CLI run example:
- Ensure ADC is configured: `gcloud auth application-default login`
- Set env vars above and run the orchestrator.

