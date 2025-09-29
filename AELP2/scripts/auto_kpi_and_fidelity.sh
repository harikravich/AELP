#!/usr/bin/env bash
set -euo pipefail

# Auto-select KPI conversion actions from Ads data, refresh views, and run fidelity.
# - Picks top N conversion_action_id by conversion_value over the last 90 days
# - Writes KPI-only views and runs fidelity over the last 14 days

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}
export BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:?Set BIGQUERY_TRAINING_DATASET}

TOP_N=${TOP_N:-3}
START=${START:-$(date -u -d '14 days ago' +%F)}
END=${END:-$(date -u +%F)}

DS="${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}"

if [[ -n "${AELP2_KPI_CONVERSION_ACTION_IDS:-}" ]]; then
  echo "Using pre‑locked KPI conversion_action_ids: ${AELP2_KPI_CONVERSION_ACTION_IDS}"
  export AELP2_FIDELITY_USE_KPI_ONLY=1
else
  echo "Selecting top ${TOP_N} KPI conversion actions from ${DS}.ads_conversion_action_stats (last 90 days)"
  Q="SELECT CAST(conversion_action_id AS STRING) AS id, SUM(conversion_value) AS value
     FROM \`${DS}.ads_conversion_action_stats\`
     WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) AND CURRENT_DATE()
     GROUP BY id
     ORDER BY value DESC
     LIMIT ${TOP_N}"

  # Requires bq CLI and ADC or service account; running on GCE VM should work.
  set +e
  IDS_CSV=$(bq query --use_legacy_sql=false --format=csv --quiet "$Q" 2>/dev/null | awk -F, 'NR>1{print $1}' | paste -sd, -)
  RC=$?
  set -e

  if [[ $RC -ne 0 || -z "${IDS_CSV}" ]]; then
    echo "No KPI conversion_action_ids found or bq query failed. Proceeding without KPI-only filters."
    export AELP2_FIDELITY_USE_KPI_ONLY=0
  else
    echo "KPI conversion_action_ids: ${IDS_CSV}"
    export AELP2_KPI_CONVERSION_ACTION_IDS="${IDS_CSV}"
    export AELP2_FIDELITY_USE_KPI_ONLY=1
  fi
fi

# Refresh views to materialize ads_kpi_daily if KPI IDs are set
echo "Refreshing BigQuery views"
python3 -m AELP2.pipelines.create_bq_views || true

# Optional: compute GA4 lagged attribution for context
echo "Running GA4 lagged attribution (last 28 days)"
bash AELP2/scripts/run_ga4_attribution.sh || true

# Reconcile and run fidelity on the 14-day window
echo "Reconciling RL vs Ads/GA4 for ${START}..${END}"
env -u GOOGLE_APPLICATION_CREDENTIALS AELP2_BQ_USE_GCE=1 python3 -m AELP2.pipelines.reconcile_posthoc --start "$START" --end "$END"

echo "Running fidelity evaluation for ${START}..${END} (KPI-only=${AELP2_FIDELITY_USE_KPI_ONLY:-0})"
env -u GOOGLE_APPLICATION_CREDENTIALS AELP2_BQ_USE_GCE=1 bash AELP2/scripts/run_fidelity.sh "$START" "$END"

echo "auto_kpi_and_fidelity completed."
