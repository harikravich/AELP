#!/usr/bin/env bash
set -euo pipefail

# Autonomy script for Terminal C (R&D/Migration)
# - Deploys R&D dashboard, runs small training to emit bidding events with explainability, refreshes views
# - Captures Auctions Monitor snapshots; keeps ports behind flags

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
export BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_rnd_20250905}
export AELP2_BQ_USE_GCE=${AELP2_BQ_USE_GCE:-1}
export AELP2_BIDDING_EVENTS_ENABLE=${AELP2_BIDDING_EVENTS_ENABLE:-1}
export AELP2_EXPLAINABILITY_ENABLE=${AELP2_EXPLAINABILITY_ENABLE:-1}
export AELP2_TARGET_WIN_RATE_MIN=${AELP2_TARGET_WIN_RATE_MIN:-0.30}
export AELP2_TARGET_WIN_RATE_MAX=${AELP2_TARGET_WIN_RATE_MAX:-0.60}

echo "[C][$(date -Is)] Ensure dataset + views" | tee -a logs/rnd_status.log
bq --project_id="$GOOGLE_CLOUD_PROJECT" mk --location=US "$GOOGLE_CLOUD_PROJECT:$BIGQUERY_TRAINING_DATASET" >/dev/null 2>&1 || true
python3 -m AELP2.pipelines.create_bq_views || true

echo "[C][$(date -Is)] Deploy R&D dashboard (aelp2-dashboard-rnd)" | tee -a logs/rnd_status.log
REGION=${REGION:-us-central1} SERVICE_NAME=${SERVICE_NAME:-aelp2-dashboard-rnd} \
  GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" BIGQUERY_TRAINING_DATASET="$BIGQUERY_TRAINING_DATASET" \
  bash AELP2/scripts/deploy_dashboard.sh | tee -a logs/rnd_deploy.log || true

echo "[C][$(date -Is)] Small training (5x300, budget 2k) to emit bidding events" | tee -a logs/rnd_status.log
AELP2_EPISODES=5 AELP2_SIM_STEPS=300 AELP2_SIM_BUDGET=2000 AELP2_AOV=100 \
  bash AELP2/scripts/run_quick_fidelity.sh | tee -a logs/rnd_quick_fidelity.log || true

python3 -m AELP2.pipelines.create_bq_views || true

echo "[C][$(date -Is)] Auctions Monitor snapshots" | tee -a logs/rnd_status.log
bq --project_id="$GOOGLE_CLOUD_PROJECT" query --use_legacy_sql=false --format=json \
  "SELECT * FROM \`$GOOGLE_CLOUD_PROJECT.$BIGQUERY_TRAINING_DATASET.bidding_events\` WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 MINUTE) ORDER BY timestamp DESC LIMIT 200" \
  > logs/rnd_bidding_recent.json || true

bq --project_id="$GOOGLE_CLOUD_PROJECT" query --use_legacy_sql=false --format=json \
  "SELECT * FROM \`$GOOGLE_CLOUD_PROJECT.$BIGQUERY_TRAINING_DATASET.bidding_events_per_minute\` ORDER BY minute DESC LIMIT 120" \
  > logs/rnd_bidding_minutely.json || true

echo "[C][$(date -Is)] DONE" | tee -a logs/rnd_status.log

