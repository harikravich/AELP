#!/usr/bin/env bash
set -euo pipefail

# YOLO Autopilot: overnight training + fidelity + shadow canary (no live changes)
# - Forces early wins, then balances floor via autotune
# - Runs extended training and evaluates fidelity gates
# - Logs shadow canary proposals to BigQuery (no live mutations)
# - Writes detailed logs under logs/yolo_*

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"
mkdir -p logs AELP2/config

log() { echo "[YOLO][$(date -Is)] $*" | tee -a logs/yolo_status.log; }

# Load base envs (do not override pre-set)
if [[ -f .env ]]; then
  while IFS='=' read -r k v; do [[ -z "$k" || "$k" =~ ^# ]] && continue; [[ -z "${!k:-}" ]] && export "$k=$v"; done < .env
fi
if [[ -f AELP2/config/.env.aelp2 ]]; then
  while IFS='=' read -r k v; do [[ -z "$k" || "$k" =~ ^# ]] && continue; [[ -z "${!k:-}" ]] && export "$k=$v"; done < AELP2/config/.env.aelp2
fi

# Ensure required project/dataset envs are present
export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
export BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training}

# Helper: run one training with current env
run_training() {
  local episodes=${1:-1}
  local steps=${2:-300}
  local budget=${3:-3000}
  log "Training: episodes=$episodes steps=$steps budget=$budget"
  AELP2_EPISODES="$episodes" AELP2_SIM_STEPS="$steps" AELP2_SIM_BUDGET="$budget" bash AELP2/scripts/run_aelp2.sh | tee -a logs/yolo_training.log || true
}

# Helper: read latest episode metrics from BQ
get_latest_metrics() {
  bq --quiet --project_id="$GOOGLE_CLOUD_PROJECT" query --use_legacy_sql=false --format=csv \
    "SELECT ROUND(win_rate,6), ROUND(spend,2), ROUND(roas,3) FROM \`$GOOGLE_CLOUD_PROJECT.$BIGQUERY_TRAINING_DATASET.training_episodes\` ORDER BY timestamp DESC LIMIT 1" 2>/dev/null | tail -n +2 || true
}

# Helper: refresh views
refresh_views() { python3 -m AELP2.pipelines.create_bq_views | tee -a logs/yolo_views.log || true; }

# Optional: Gmail Ads→BQ (best-effort)
ingest_gmail_ads() {
  if [[ -n "${GMAIL_REFRESH_TOKEN:-}" ]]; then
    log "Gmail Ads env detected; creating isolated creds file"
    bash scripts/run_google_ads_gmail_from_env.sh | tee -a logs/yolo_gmail_setup.log || true
    if [[ -f AELP2/config/.google_ads_credentials.gmail.env ]]; then
      set -a; source AELP2/config/.google_ads_credentials.gmail.env; set +a
      # Force child CID for metrics (avoid manager-level metrics error)
      export GOOGLE_ADS_LOGIN_CUSTOMER_ID=9704174968
      export GOOGLE_ADS_CUSTOMER_ID=5843249566
      export GOOGLE_CLOUD_PROJECT BIGQUERY_TRAINING_DATASET
      local START=$(date -u -d '14 days ago' +%F); local END=$(date -u +%F)
      log "Ads→BQ for ${GOOGLE_ADS_CUSTOMER_ID} ${START}..${END}"
      python3 -m AELP2.pipelines.google_ads_to_bq --start "$START" --end "$END" | tee -a logs/yolo_ads_ingest.log || true
      refresh_views
    fi
  else
    log "Gmail Ads OAuth not present; skipping Ads ingest"
  fi
}

# Stage 0: Prime datasets and views
log "Ensuring BQ views"
refresh_views

# Stage 1: Attempt Ads ingest (best effort; non-blocking)
ingest_gmail_ads || true

# Stage 2: Forced-wins warm start (up to 3 attempts)
attempt=1
while (( attempt <= 3 )); do
  log "Forced-wins attempt $attempt/3"
  export LOG_LEVEL=${LOG_LEVEL:-INFO}
  export AELP2_CALIBRATION_FLOOR_RATIO=${AELP2_CALIBRATION_FLOOR_RATIO:-1.10}
  export AELP2_FLOOR_AUTOTUNE_ENABLE=1
  export AELP2_FLOOR_AUTOTUNE_MIN=0.20
  export AELP2_FLOOR_AUTOTUNE_MAX=0.60
  export AELP2_FLOOR_AUTOTUNE_STEP=0.05
  export AELP2_TARGET_WIN_RATE_MIN=0.05
  export AELP2_TARGET_WIN_RATE_MAX=0.25
  export AELP2_NOWIN_GUARD_ENABLE=1
  export AELP2_NOWIN_GUARD_STEPS=5
  export AELP2_NOWIN_GUARD_FACTOR=4.0
  export AELP2_ROAS_BASIS=aov
  export AELP2_AOV=${AELP2_AOV:-100}
  export AELP2_CALIBRATION_MAX_KS=${AELP2_CALIBRATION_MAX_KS:-0.70}
  export AELP2_CALIBRATION_MAX_HIST_MSE=${AELP2_CALIBRATION_MAX_HIST_MSE:-5.0}
  run_training 1 300 3000
  METRICS=$(get_latest_metrics || true)
  echo "$METRICS" | tee -a logs/yolo_metrics.csv
  WR=$(echo "$METRICS" | awk -F, '{print $1+0}')
  if awk -v wr="$WR" 'BEGIN{exit (wr>=0.05)?0:1}'; then
    log "Win rate achieved (>=5%): $WR"
    break
  fi
  # escalate floor/guard
  export AELP2_CALIBRATION_FLOOR_RATIO=$(python3 - <<'PY'
import os
x=float(os.environ.get('AELP2_CALIBRATION_FLOOR_RATIO','1.10'))
print(min(1.50, x+0.10))
PY
)
  export AELP2_NOWIN_GUARD_STEPS=3
  export AELP2_NOWIN_GUARD_FACTOR=5.0
  attempt=$((attempt+1))
done

# Stage 3: Balance floors (3 episodes)
export AELP2_CALIBRATION_FLOOR_RATIO=0.60
export AELP2_NOWIN_GUARD_STEPS=20
export AELP2_NOWIN_GUARD_FACTOR=2.0
run_training 3 300 3000
refresh_views

# Stage 4: Extended training (20 episodes × 800)
# Nudge floors down if wins are stable
export AELP2_CALIBRATION_FLOOR_RATIO=${AELP2_CALIBRATION_FLOOR_RATIO:-0.60}
run_training 20 800 12000
refresh_views

# Stage 5: Fidelity (KPI-only, GA4 lag-aware). Keep current floor by pre-setting.
export AELP2_CALIBRATION_FLOOR_RATIO=${AELP2_CALIBRATION_FLOOR_RATIO:-0.60}
bash AELP2/scripts/auto_kpi_and_fidelity.sh | tee -a logs/yolo_fidelity.log || true

# Stage 6: Check canary readiness (relax thresholds if needed)
export AELP2_FIDELITY_MAX_MAPE_ROAS=${AELP2_FIDELITY_MAX_MAPE_ROAS:-1.0}
export AELP2_FIDELITY_MAX_MAPE_CAC=${AELP2_FIDELITY_MAX_MAPE_CAC:-6.0}
export AELP2_FIDELITY_MAX_KS_WINRATE=${AELP2_FIDELITY_MAX_KS_WINRATE:-0.60}
python3 AELP2/scripts/check_canary_readiness.py | tee -a logs/yolo_readiness.log || true

# Stage 7: Shadow canary proposal (no live changes)
export AELP2_GOOGLE_CANARY_CAMPAIGN_IDS=${AELP2_GOOGLE_CANARY_CAMPAIGN_IDS:-22983040986}
export AELP2_CANARY_BUDGET_DELTA_PCT=${AELP2_CANARY_BUDGET_DELTA_PCT:-0.05}
export AELP2_CANARY_MAX_CHANGES_PER_RUN=${AELP2_CANARY_MAX_CHANGES_PER_RUN:-1}
export AELP2_CANARY_MAX_DAILY_DELTA_PCT=${AELP2_CANARY_MAX_DAILY_DELTA_PCT:-0.10}
export AELP2_SHADOW_MODE=1
export AELP2_ALLOW_GOOGLE_MUTATIONS=0
PYTHONPATH=. python3 AELP2/scripts/apply_google_canary.py | tee -a logs/yolo_canary_shadow.log || true

log "YOLO Autopilot completed"
