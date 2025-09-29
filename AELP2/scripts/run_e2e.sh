#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner:
# 1) Ingest Google Ads (MCC) for last 14 days
# 2) Run one AELP2 training episode (writes RL telemetry)
# 3) Reconcile RL vs Ads/GA4 daily metrics
# 4) Run fidelity evaluation

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Required env. Adjust if different in your project.
export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
export BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training}
export BIGQUERY_DATASET_LOCATION=${BIGQUERY_DATASET_LOCATION:-US}

# Use GCE instance service account for all BigQuery writes/reads (no browser auth)
export AELP2_BQ_USE_GCE=1

# Dates
START=$(date -u -d '14 days ago' +%F)
END=$(date -u +%F)

echo "[0/4] GA4 aggregates + lagged attribution (last 28 days)"
bash AELP2/scripts/run_ga4_ingestion.sh --last28 || true
bash AELP2/scripts/run_ga4_attribution.sh || true
python3 -m AELP2.pipelines.create_bq_views || true

echo "[1/4] Ingesting Google Ads for $START..$END"
# Uses env from .google_ads_credentials.env if present (handled by script)
AELP2_BQ_USE_GCE=1 bash AELP2/scripts/run_ads_ingestion.sh \
  --mcc 7762856866 \
  --start "$START" --end "$END" \
  --tasks "campaigns,ad_performance,conversion_actions,conversion_action_stats"

echo "[2/4] Running one AELP2 training episode (writes RL telemetry)"
# Unset GOOGLE_APPLICATION_CREDENTIALS so BigQuery uses GCE SA inside orchestrator
env -u GOOGLE_APPLICATION_CREDENTIALS AELP2_BQ_USE_GCE=1 bash AELP2/scripts/run_aelp2.sh

echo "[3/4] Reconciling RL vs Ads/GA4 daily metrics for $START..$END"
env -u GOOGLE_APPLICATION_CREDENTIALS AELP2_BQ_USE_GCE=1 python3 -m AELP2.pipelines.reconcile_posthoc \
  --start "$START" --end "$END"

echo "[4/4] Running fidelity evaluation for $START..$END"
env -u GOOGLE_APPLICATION_CREDENTIALS AELP2_BQ_USE_GCE=1 bash AELP2/scripts/run_fidelity.sh "$START" "$END"

echo "Done."
