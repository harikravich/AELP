Internal Pacing → BigQuery → GA4 Tie‑Out

Purpose
- Use your internal pacing file (spend by day) to compute CAC against GA4 enrollments while Meta/Impact APIs are blocked.

Accepted Input
- CSV (export from Excel/Sheets).
- Minimal columns: `date`, `cost` (or `spend`).
- Optional: `platform`, `clicks`, `impressions`, `conversions`, `revenue`.
- Date formats: `YYYY-MM-DD` or `MM/DD/YYYY`.

Steps
1) Save your pacing file as CSV, e.g., `AELP2/data/pacing.csv`.
2) Load pacing to BigQuery:
   - `export GOOGLE_CLOUD_PROJECT=<proj>`
   - `export BIGQUERY_TRAINING_DATASET=gaelp_training`
   - `python3 AELP2/scripts/pacing_to_bq.py --file AELP2/data/pacing.csv`
   - This writes to `<proj>.gaelp_training.pacing_daily` (replaces covered dates).
3) Ensure GA4 enrollments are present:
   - If you already have `<proj>.gaelp_training.ga4_enrollments_daily`, skip.
   - Else run: `python3 AELP2/scripts/ga4_event_to_bq_enrollments.py --event purchase --days 120`
     - Requires `GA4_PROPERTY_ID` and GA4 auth (ADC or OAuth refresh token).
4) Create tie‑out views:
   - `python3 AELP2/scripts/pacing_reconcile_ga4.py`
   - Views created:
     - `pacing_vs_ga4_kpi_daily` (columns: date, cost, ga4_enrollments, ga4_cac)
     - `pacing_vs_ga4_summary` (28/45/90‑day rollups: cost_xx, enr_xx, cac_xx)

Quick Checks
- Last 28 days CAC: `bq query --nouse_legacy_sql "SELECT cac_28 FROM \`${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.pacing_vs_ga4_summary\`"`
- Daily table preview: `bq head -n 10 \`${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.pacing_vs_ga4_kpi_daily\``

Notes
- You can include per‑platform spend in the CSV via `platform` column; the current rollup sums all platforms to a single CAC.
- Safe to rerun: loaders/view creators are idempotent; they replace the date window or view definition.

