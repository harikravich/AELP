#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

echo "==== Morning Report ($(date -Is)) ===="

section() { echo; echo "----- $1 -----"; }

section "Prod: Headroom"
[[ -f logs/prod_headroom.txt ]] && tail -n +1 logs/prod_headroom.txt || echo "(no prod_headroom.txt)"

section "Prod: KPI Daily (last 28d)"
[[ -f logs/prod_kpi_daily.txt ]] && tail -n +1 logs/prod_kpi_daily.txt || echo "(no prod_kpi_daily.txt)"

section "Prod: Training Trend (2h)"
[[ -f logs/prod_training_trend.txt ]] && tail -n +1 logs/prod_training_trend.txt || echo "(no prod_training_trend.txt)"

section "Cloud Run Services"
if command -v gcloud >/dev/null 2>&1; then
  REGION=${REGION:-us-central1}
  echo "Hari:"; gcloud run services describe aelp2-dashboard-hari --region "$REGION" --format='value(status.url, status.latestReadyRevisionName, status.conditions[-1].message)' || echo "(describe failed)"
  echo "R&D:"; gcloud run services describe aelp2-dashboard-rnd --region "$REGION" --format='value(status.url, status.latestReadyRevisionName, status.conditions[-1].message)' || echo "(describe failed)"
else
  echo "(gcloud not available; see logs/prod_status.log)"
fi

section "Sandbox: Fidelity (latest)"
[[ -f logs/sandbox_fidelity.txt ]] && tail -n +1 logs/sandbox_fidelity.txt || echo "(no sandbox_fidelity.txt)"

section "Sandbox: Episodes Daily"
[[ -f logs/sandbox_episodes_daily.txt ]] && tail -n +1 logs/sandbox_episodes_daily.txt || echo "(no sandbox_episodes_daily.txt)"

section "Sandbox: Bidding Events (minutely JSON sample)"
[[ -f logs/sandbox_bidding_minutely.json ]] && jq '.[0:5]' logs/sandbox_bidding_minutely.json || echo "(no sandbox_bidding_minutely.json)"

section "R&D: Bidding Recent (JSON sample)"
[[ -f logs/rnd_bidding_recent.json ]] && jq '.[0:5]' logs/rnd_bidding_recent.json || echo "(no rnd_bidding_recent.json)"

section "R&D: Bidding Minutely (JSON sample)"
[[ -f logs/rnd_bidding_minutely.json ]] && jq '.[0:5]' logs/rnd_bidding_minutely.json || echo "(no rnd_bidding_minutely.json)"

echo
echo "==== End of Report ===="
