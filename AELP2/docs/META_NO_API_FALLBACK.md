Meta Ads → BigQuery (No‑API Fallback)

Use this when developer registration or tokens are blocked. Export from Ads Manager and load the CSV.

Export from Ads Manager (desktop)
- Open Ads Manager for the Aura account (act_192357189581498).
- Set date range (e.g., Aug 1–Sep 15, 2025).
- Breakdown: by Day (so rows are daily).
- Level: Ad.
- Columns: include Ad ID, Ad Set ID, Campaign ID, Impressions, Clicks (all), Amount spent, Purchases, Purchases conversion value (or your key result columns).
- Click Export → CSV.

Load into BigQuery
- Ensure env: `export GOOGLE_CLOUD_PROJECT=<proj>` `export BIGQUERY_TRAINING_DATASET=gaelp_training`
- Run: `python3 AELP2/scripts/meta_csv_to_bq.py --file /path/to/export.csv`
- Destination: `<proj>.gaelp_training.meta_ad_performance` (same table as API path).

Notes
- The loader tolerates common column label differences.
- Safe to rerun: it replaces the date range covered by the CSV.
- Ad names are hashed by default (`AELP2_REDACT_TEXT=1`).

