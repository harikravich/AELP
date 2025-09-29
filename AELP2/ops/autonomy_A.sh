#!/usr/bin/env bash
set -euo pipefail

# Autonomy script for Terminal A (Master/Prod) – Monitoring only
# - Refresh views, headroom, KPI summaries, recent training trends
# - Writes detailed logs for morning review

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
export BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training}
export AELP2_BQ_USE_GCE=${AELP2_BQ_USE_GCE:-1}

echo "[A][$(date -Is)] Refresh views" | tee -a logs/prod_status.log
python3 -m AELP2.pipelines.create_bq_views | tee -a logs/prod_views.log || true

echo "[A][$(date -Is)] Headroom" | tee -a logs/prod_status.log
GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" BIGQUERY_TRAINING_DATASET="$BIGQUERY_TRAINING_DATASET" \
  python3 -m AELP2.scripts.assess_headroom | tee logs/prod_headroom.txt || true

echo "[A][$(date -Is)] KPI daily (28d)" | tee -a logs/prod_status.log
bq --project_id="$GOOGLE_CLOUD_PROJECT" query --use_legacy_sql=false --format=pretty \
  "SELECT DATE(date) d, conversions, revenue, cost, SAFE_DIVIDE(revenue, NULLIF(cost,0)) roas, SAFE_DIVIDE(cost, NULLIF(conversions,0)) cac FROM \`$GOOGLE_CLOUD_PROJECT.$BIGQUERY_TRAINING_DATASET.ads_kpi_daily\` ORDER BY d DESC LIMIT 28" \
  | tee logs/prod_kpi_daily.txt || true

echo "[A][$(date -Is)] Training trend (last 2h, per-minute)" | tee -a logs/prod_status.log
bq --project_id="$GOOGLE_CLOUD_PROJECT" query --use_legacy_sql=false --format=pretty \
  "WITH buckets AS (SELECT TIMESTAMP_TRUNC(timestamp, MINUTE) m, AVG(win_rate) win_rate, SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) roas, SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) cac FROM \`$GOOGLE_CLOUD_PROJECT.$BIGQUERY_TRAINING_DATASET.training_episodes\` WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 120 MINUTE) GROUP BY m) SELECT * FROM buckets ORDER BY m DESC LIMIT 24" \
  | tee logs/prod_training_trend.txt || true

echo "[A][$(date -Is)] DONE" | tee -a logs/prod_status.log

