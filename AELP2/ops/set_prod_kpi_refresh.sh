#!/usr/bin/env bash
set -euo pipefail

# Helper: Create/refresh Prod KPI daily view using locked Ads conversion_action_ids

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
export BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training}
PROJECT=$GOOGLE_CLOUD_PROJECT
DATASET=$BIGQUERY_TRAINING_DATASET
KPI_IDS=${AELP2_KPI_CONVERSION_ACTION_IDS:-6453292723}
export AELP2_BQ_USE_GCE=${AELP2_BQ_USE_GCE:-1}

if [[ -z "$KPI_IDS" ]]; then
  echo "Set AELP2_KPI_CONVERSION_ACTION_IDS (comma-separated Ads conversion_action_ids)" >&2
  exit 2
fi

echo "[set_prod_kpi] Creating ads_kpi_daily in ${PROJECT}.${DATASET} for KPI IDs: ${KPI_IDS}"
export AELP2_KPI_CONVERSION_ACTION_IDS="$KPI_IDS"
python3 -m AELP2.pipelines.create_bq_views

echo "[set_prod_kpi] Preview KPI daily (last 7d)"
bq --project_id="$PROJECT" query --use_legacy_sql=false --format=pretty \
  "SELECT DATE(date) d, conversions, revenue, cost, SAFE_DIVIDE(revenue, NULLIF(cost,0)) roas, SAFE_DIVIDE(cost, NULLIF(conversions,0)) cac FROM \`${PROJECT}.${DATASET}.ads_kpi_daily\` ORDER BY d DESC LIMIT 7"

echo "[set_prod_kpi] Done."
