#!/usr/bin/env bash
set -euo pipefail

# Kick off key AELP2 jobs in the background with logs + PID files.
# Logs: /tmp/aelp2_*.log, PIDs: /tmp/aelp2_*.pid

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then set -a; source ./.env; set +a; fi
if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" || -z "${BIGQUERY_TRAINING_DATASET:-}" ]]; then
  echo "Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET in .env" >&2
  exit 2
fi

# Do NOT override GOOGLE_APPLICATION_CREDENTIALS globally to avoid breaking BigQuery perms.
# GA4 loaders handle their own auth (OAuth refresh token, SA JSON, or ADC with scopes).

start_job() {
  local name="$1"; shift
  local log="/tmp/aelp2_${name}.log"
  local pidf="/tmp/aelp2_${name}.pid"
  echo "[start] $name → $log"
  # shellcheck disable=SC2068
  nohup "$@" >"$log" 2>&1 < /dev/null &
  echo $! > "$pidf"
}

# 1) GA4 ingest (last 28 days)
GA4_SA_JSON="/home/hariravichandran/.config/gaelp/ga4-service-account.json"
if [[ -f "$GA4_SA_JSON" ]]; then
  start_job ga4_ingest env GOOGLE_APPLICATION_CREDENTIALS="$GA4_SA_JSON" bash AELP2/scripts/run_ga4_ingestion.sh --last28
else
  start_job ga4_ingest bash AELP2/scripts/run_ga4_ingestion.sh --last28
fi

# 2) GA4 lagged attribution
start_job ga4_attr bash AELP2/scripts/run_ga4_attribution.sh

# 3) Create/refresh BQ views
start_job views python3 -m AELP2.pipelines.create_bq_views

# 4) MMM (KPI-only)
START=$(date -u -d '90 day ago' +%F)
END=$(date -u +%F)
start_job mmm env AELP2_MMM_USE_KPI=1 python3 AELP2/pipelines/mmm_service.py --start "$START" --end "$END"

# 5) Bandit decisions (shadow)
start_job bandit_service python3 -m AELP2.core.optimization.bandit_service --lookback 30

# 6) Budget orchestrator (shadow)
start_job budget_orch python3 -m AELP2.core.optimization.budget_orchestrator --days 14 --top_n 1

# 7) Training (stub fallback)
start_job training_stub python3 AELP2/scripts/training_stub.py --episodes 50 --steps 400 --budget 5000

echo "Background jobs started. Use AELP2/scripts/check_long_jobs.sh to check status."
