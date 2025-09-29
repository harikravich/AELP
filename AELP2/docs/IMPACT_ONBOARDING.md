Impact.com (Impact Radius) Onboarding — Advertiser API → BigQuery

Goal
- Ingest daily partner/creator performance (impressions, clicks, actions, payout, revenue) into BigQuery.

What I need from you
- Account type: Advertiser account on impact.com for Aura.
- Credentials (from the UI): `IMPACT_ACCOUNT_SID`, `IMPACT_AUTH_TOKEN`.

Where to find credentials
1) Sign in to impact.com (Advertiser account).
2) Go to Settings → Technical → API Access (or “API” in newer UI).
3) Enable API Access if disabled, then copy:
   - Account SID
   - Auth Token
4) Share securely, or export locally:
   - `export IMPACT_ACCOUNT_SID='...'`
   - `export IMPACT_AUTH_TOKEN='...'`

Run the ingestion
- Ensure BigQuery env:
  - `export GOOGLE_CLOUD_PROJECT=<your-project>`
  - `export BIGQUERY_TRAINING_DATASET=gaelp_training`
- Discover available reports:
  - `python3 AELP2/pipelines/impact_to_bq.py --list`
- Pick a partner/day report (e.g., `adv_partner_performance`) and backfill:
  - `python3 AELP2/pipelines/impact_to_bq.py --report adv_partner_performance --start 2025-08-01 --end 2025-09-15`

Destination schema
- BigQuery table: `<project>.<dataset>.impact_partner_performance` (partitioned by `date`).
- Columns: `date`, `partner_id`, `partner`, `impressions`, `clicks`, `actions`, `payout`, `revenue`.

Notes & troubleshooting
- Permissions: Your impact.com user must have rights to access Reports and API tokens.
- Report IDs vary: use `--list` to discover IDs on your account; `--describe` shows fields.
- Large pulls: If you hit pagination caps, split by month and rerun; the loader is idempotent over the date window.

