GA4 ↔ Pacer Reconciliation

Goal
- Reproduce the pacing workbook’s key conversion lines from GA4, side‑by‑side, and compute CAC using GA4 enrollments vs internal subs.

What’s included
- Loader: `AELP2/scripts/pacing_excel_to_bq.py` → tables:
  - `pacing_daily` (date, spend as `cost`, d2c_total_subscribers as `conversions`)
  - `pacing_pacer_daily` (date, spend + Free Trial, App Trial, D2P Starts, Post‑trial Subs, Mobile Subs, D2C Total Subs)
- GA4 builder: `AELP2/scripts/ga4_pacer_to_bq.py` using mapping `AELP2/config/ga4_pacer_mapping.yaml` → `ga4_pacer_daily`.
- Tie‑out views: `pacing_vs_ga4_kpi_daily` and monthly variants already created; add pacer‑level views if desired.

How to run
1) Load the pacing workbook (already done in your env):
   - `python3 AELP2/scripts/pacing_excel_to_bq.py --file "AELP2/data/Pacing 2025-1.xlsx"`
2) Configure GA4 mapping:
   - Edit `AELP2/config/ga4_pacer_mapping.yaml` to match your GA4 event names for:
     - `free_trial_starts`, `app_trial_starts`, `d2p_starts`, `post_trial_subscribers`, `mobile_subscribers`, `d2c_total_subscribers`.
3) Run GA4 extraction (needs GA4_PROPERTY_ID + auth):
   - `export GA4_PROPERTY_ID=properties/XXXXXXXX`
   - Auth: either set `GOOGLE_APPLICATION_CREDENTIALS` to a SA JSON with GA4 read, or `GA4_OAUTH_REFRESH_TOKEN`, `GA4_OAUTH_CLIENT_ID`, `GA4_OAUTH_CLIENT_SECRET`.
   - `python3 AELP2/scripts/ga4_pacer_to_bq.py --days 400`
4) Compare monthly:
   - `bq query --nouse_legacy_sql "SELECT DATE_TRUNC(date, MONTH) m, SUM(spend) spend, SUM(d2c_total_subscribers) subs FROM \`${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.pacing_pacer_daily\` GROUP BY m ORDER BY m"`
   - `bq query --nouse_legacy_sql "SELECT DATE_TRUNC(date, MONTH) m, * EXCEPT(date) FROM \`${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.ga4_pacer_daily\` GROUP BY m ORDER BY m"`

Tips for mapping
- Start with `purchase` for `d2c_total_subscribers` and `mobile_subscribers` with `device_category=mobile`.
- Trials often use custom events like `start_free_trial`, `trial_started`, `trial_begin`; validate by listing top GA4 events (Data API) over the last 90 days.
- If your mobile app is a separate GA4 app stream, the Data API still aggregates at property level. Use `deviceCategory` to split.

Next
- Once mapping ties out, we can wire `/api/bq/kpi` to optionally use `pacing_pacer_daily` vs `ga4_pacer_daily` per metric to compute blended CAC and SoU.

