# Setup Blockers

This environment is missing some optional tools; features fall back gracefully.

- docker: not found — required to run Robyn and ChannelAttribution containers.
  - Ubuntu (Jammy):
    - `sudo apt-get update`
    - `sudo apt-get install -y docker.io`
    - `sudo systemctl enable --now docker`
    - `sudo usermod -aG docker $USER` (log out/in to use Docker without sudo)
  - macOS/Windows: Install Docker Desktop.
  - Then set `AELP2_ROBYN_ALLOW_RUN=1` and/or `AELP2_CA_ALLOW_RUN=1` and rerun flows.

- BigQuery permissions may be limited.
  - Grant the service account used for BigQuery: `roles/bigquery.user` and `roles/bigquery.dataEditor` on project `${GOOGLE_CLOUD_PROJECT}`.
  - Alternatively set `AELP2_BQ_CREDENTIALS=/path/to/sa.json` for BigQuery.

- google-ads credentials: ensure all GOOGLE_ADS_* env vars are set for Ads ingestion, Reach Planner, and budget canaries.

If you unblock any item, rerun `python3 AELP2/scripts/preflight.py`.
