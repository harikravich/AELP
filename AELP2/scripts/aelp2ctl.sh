#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    line="${line#export }"
    key="${line%%=*}"; val="${line#*=}"
    [[ -z "$key" ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

ensure_pydeps(){
  python3 -m pip install --user -q google-cloud-bigquery google-auth google-api-core >/dev/null 2>&1 || true
}

usage(){
  cat <<'USAGE'
Usage: aelp2ctl <command> [args]

Core:
  start                          Start dev server on :3000
  stop                           Stop dev server on :3000
  health                         Print server health JSON
  doctor                         Env doctor (project/datasets/bq)

Data:
  ads --last14 [--mcc ID] [--tasks LIST]
  ga4 --last28 | --start YYYY-MM-DD --end YYYY-MM-DD
  ga4-attr [--start YYYY-MM-DD --end YYYY-MM-DD]   (defaults last 28)
  views [--kpi-ids CSV]           Create/refresh BQ views; set KPI IDs via env
  kpi-lock --ids CSV              Create ads_kpi_daily with these conversion_action_ids
  fidelity [--start YYYY-MM-DD --end YYYY-MM-DD]

RL / Auctions:
  train [--episodes N --steps N --budget N]
  auctions-seed                   Seed synthetic auctions data for monitor
  kpi-suggest                     Show top conversion_action_ids (last 90d)
  reconcile-ga4 [--event NAME]    Create GA4 enrollments + Ads↔GA4 reconciliation views

One-shot:
  warmup [--mcc ID] [--kpi-ids CSV]
    Runs: ga4 --last28, ads --last14, views (with KPI if provided), ga4-attr, train (5x400 @ $5000), auctions-seed

Examples:
  aelp2ctl start
  aelp2ctl ads --last14 --mcc 7762856866
  aelp2ctl kpi-lock --ids 6453292723,1234567890
  aelp2ctl warmup --mcc 7762856866 --kpi-ids 6453292723,1234567890
USAGE
}

cmd=${1:-}
shift || true

case "${cmd}" in
  start)
    bash AELP2/scripts/quickstart_pilot.sh ;;
  stop)
    if lsof -ti:3000 >/dev/null 2>&1; then kill -9 $(lsof -ti:3000); echo "Stopped :3000"; else echo "No process on :3000"; fi ;;
  health)
    curl -fsS http://127.0.0.1:3000/api/connections/health | jq . || curl -sS http://127.0.0.1:3000/api/connections/health ;;
  doctor)
    bash AELP2/scripts/env_doctor.sh ;;
  ads)
    bash AELP2/scripts/run_ads_ingestion.sh "$@" ;;
  ga4)
    bash AELP2/scripts/run_ga4_ingestion.sh "$@" ;;
  ga4-attr)
    bash AELP2/scripts/run_ga4_attribution.sh "$@" ;;
  views)
    ensure_pydeps
    # Allow passing KPI IDs to set env and create ads_kpi_daily
    KPI_IDS=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --kpi-ids) KPI_IDS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 2;;
      esac
    done
    if [[ -n "$KPI_IDS" ]]; then export AELP2_KPI_CONVERSION_ACTION_IDS="$KPI_IDS"; fi
    python3 -m AELP2.pipelines.create_bq_views ;;
  kpi-lock)
    ensure_pydeps
    # Create ads_kpi_daily directly in BQ (mirrors control route) using provided IDs
    IDS=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --ids) IDS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 2;;
      esac
    done
    [[ -n "$IDS" ]] || { echo "Provide --ids 'id1,id2'" >&2; exit 2; }
    export AELP2_KPI_CONVERSION_ACTION_IDS="$IDS"
    python3 -m AELP2.pipelines.create_bq_views ;;
  fidelity)
    bash AELP2/scripts/run_fidelity.sh "$@" ;;
  train)
    ensure_pydeps
    EP=5; ST=400; BU=5000
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --episodes) EP="$2"; shift 2;;
        --steps) ST="$2"; shift 2;;
        --budget) BU="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 2;;
      esac
    done
    python3 AELP2/scripts/training_stub.py --episodes "$EP" --steps "$ST" --budget "$BU" ;;
  auctions-seed)
    ensure_pydeps
    python3 AELP2/scripts/seed_auctions_stub.py ;;
  reconcile-ga4)
    ensure_pydeps
    EV=""; EXP="${GA4_EXPORT_DATASET:-}"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --event) EV="$2"; shift 2;;
        --export-dataset) EXP="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 2;;
      esac
    done
    if [[ -n "$EXP" ]]; then export GA4_EXPORT_DATASET="$EXP"; fi
    if [[ -n "$EV" && -z "$EXP" ]]; then
      # Fallback: pull event counts via GA4 Data API into BQ table, using GCE ADC for BigQuery writes
      echo "[reconcile-ga4] GA4 export not provided; loading '$EV' via GA4 Data API into ga4_enrollments_daily… (GA4 auth from GOOGLE_APPLICATION_CREDENTIALS if set)"
      python3 AELP2/scripts/ga4_event_to_bq_enrollments.py --event "$EV" || true
    fi
    if [[ -n "$EV" ]]; then python3 AELP2/scripts/ga4_reconcile_ads.py --event "$EV" ${EXP:+--export-dataset "$EXP"};
    else python3 AELP2/scripts/ga4_reconcile_ads.py; fi ;;
  kpi-suggest)
    ensure_pydeps
    DS="${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}"
    Q=$(cat <<'SQL'
SELECT CAST(s.conversion_action_id AS STRING) id, ANY_VALUE(a.name) name, SUM(s.conversion_value) value, SUM(s.conversions) conv
FROM `PROJECT.DATASET.ads_conversion_action_stats` s
LEFT JOIN `PROJECT.DATASET.ads_conversion_actions` a
  ON a.conversion_action_id = s.conversion_action_id
WHERE DATE(s.date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) AND CURRENT_DATE()
GROUP BY id
ORDER BY value DESC
LIMIT 10
SQL
)
    Q="${Q//PROJECT.DATASET/$DS}"
    bq --project_id "${GOOGLE_CLOUD_PROJECT}" query --use_legacy_sql=false --format=prettyjson "$Q" || true ;;
  warmup)
    ensure_pydeps
    MCC="${GOOGLE_ADS_LOGIN_CUSTOMER_ID:-}"
    KPI_IDS=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --mcc) MCC="$2"; shift 2;;
        --kpi-ids) KPI_IDS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 2;;
      esac
    done
    echo "[1/6] GA4 last 28d"
    bash AELP2/scripts/run_ga4_ingestion.sh --last28 || true
    echo "[2/6] Ads last 14d"
    if [[ -n "$MCC" ]]; then bash AELP2/scripts/run_ads_ingestion.sh --mcc "$MCC" --last14 --tasks "ad_performance,conversion_action_stats,conversion_actions" || true;
    else echo "(skip Ads: no MCC provided)"; fi
    echo "[3/6] Views + KPI"
    if [[ -n "$KPI_IDS" ]]; then AELP2_KPI_CONVERSION_ACTION_IDS="$KPI_IDS" python3 -m AELP2.pipelines.create_bq_views || true; else python3 -m AELP2.pipelines.create_bq_views || true; fi
    echo "[4/6] GA4 attribution last 28d"
    bash AELP2/scripts/run_ga4_attribution.sh || true
    echo "[5/6] RL training seed"
    python3 AELP2/scripts/training_stub.py --episodes 5 --steps 400 --budget 5000 || true
    echo "[6/6] Auctions seed"
    python3 AELP2/scripts/seed_auctions_stub.py || true
    echo "Warmup completed." ;;
  *)
    usage; exit 2 ;;
esac
