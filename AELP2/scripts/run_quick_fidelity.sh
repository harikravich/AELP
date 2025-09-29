#!/usr/bin/env bash
set -euo pipefail

# One-pass: quick training episode (low-noise) + KPI-only fidelity (with GA4 lag-aware credit)
#
# Requirements:
#   - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET set
#   - gcloud/bq installed; VM SA has BigQuery access (AELP2_BQ_USE_GCE=1)
#   - Ads/GA4 ingestion already run recently (optional but recommended)
#
# Optional overrides:
#   AELP2_EPISODES (default 1), AELP2_SIM_STEPS (default 300), AELP2_SIM_BUDGET (default 3000)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" || -z "${BIGQUERY_TRAINING_DATASET:-}" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET" >&2
  exit 2
fi

echo "[prep] Computing AOV from KPI (last 30d) if available..."
set +e
AOV_Q=$(bq query --use_legacy_sql=false --format=csv "SELECT ROUND(SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(conversions),0)),2) FROM \`${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.ads_kpi_daily\` WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()" 2>/dev/null | tail -n +2)
RC=$?
set -e
# Normalize AOV from query (strip quotes/spaces)
AOV_Q_CLEAN=$(echo "${AOV_Q}" | sed 's/[\"\'"'"']//g' | xargs)
if [[ $RC -ne 0 || -z "${AOV_Q_CLEAN}" || "${AOV_Q_CLEAN}" == "null" ]]; then
  AOV_DEFAULT=100
  echo "[prep] AOV not available; defaulting to ${AOV_DEFAULT}"
else
  AOV_DEFAULT="${AOV_Q_CLEAN}"
  echo "[prep] AOV=${AOV_DEFAULT}"
fi

echo "[env] Setting low-noise + floor/guard defaults"
export LOG_LEVEL=${LOG_LEVEL:-WARNING}
export AELP2_HITL_NON_BLOCKING=${AELP2_HITL_NON_BLOCKING:-1}
export AELP2_HITL_ON_GATE_FAIL=${AELP2_HITL_ON_GATE_FAIL:-0}
export AELP2_HITL_ON_GATE_FAIL_FOR_BIDS=${AELP2_HITL_ON_GATE_FAIL_FOR_BIDS:-0}
export AELP2_HITL_MIN_STEP_FOR_APPROVAL=${AELP2_HITL_MIN_STEP_FOR_APPROVAL:-999999}
export AELP2_CALIBRATION_FLOOR_RATIO=${AELP2_CALIBRATION_FLOOR_RATIO:-0.80}
export AELP2_FLOOR_AUTOTUNE_ENABLE=${AELP2_FLOOR_AUTOTUNE_ENABLE:-1}
export AELP2_FLOOR_AUTOTUNE_MIN=${AELP2_FLOOR_AUTOTUNE_MIN:-0.50}
export AELP2_FLOOR_AUTOTUNE_MAX=${AELP2_FLOOR_AUTOTUNE_MAX:-1.00}
export AELP2_FLOOR_AUTOTUNE_STEP=${AELP2_FLOOR_AUTOTUNE_STEP:-0.10}
export AELP2_NOWIN_GUARD_ENABLE=${AELP2_NOWIN_GUARD_ENABLE:-1}
export AELP2_NOWIN_GUARD_STEPS=${AELP2_NOWIN_GUARD_STEPS:-20}
export AELP2_NOWIN_GUARD_FACTOR=${AELP2_NOWIN_GUARD_FACTOR:-2.0}

echo "[env] Aligning ROAS to AOV"
export AELP2_ROAS_BASIS=${AELP2_ROAS_BASIS:-aov}
# Respect pre-set AELP2_AOV; otherwise use computed/default
export AELP2_AOV="${AELP2_AOV:-$AOV_DEFAULT}"

EPISODES=${AELP2_EPISODES:-1}
STEPS=${AELP2_SIM_STEPS:-300}
BUDGET=${AELP2_SIM_BUDGET:-3000}

echo "[train] Running training: episodes=${EPISODES}, steps=${STEPS}, budget=${BUDGET}"
env -u GOOGLE_APPLICATION_CREDENTIALS \
  AELP2_BQ_USE_GCE=1 \
  AELP2_EPISODES="$EPISODES" \
  AELP2_SIM_STEPS="$STEPS" \
  AELP2_SIM_BUDGET="$BUDGET" \
  bash AELP2/scripts/run_aelp2.sh

echo "[fidelity] KPI-only (with GA4 lag-aware)"
bash AELP2/scripts/auto_kpi_and_fidelity.sh

echo "[fidelity] Latest result:"
bq query --use_legacy_sql=false --format=pretty "
SELECT timestamp, passed, mape_roas, rmse_roas, mape_cac, rmse_cac, ks_winrate_vs_impressionshare
FROM \`${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.fidelity_evaluations\`
ORDER BY timestamp DESC
LIMIT 1" || true

echo "Done."
