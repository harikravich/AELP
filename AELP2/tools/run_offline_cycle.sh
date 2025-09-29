#!/usr/bin/env bash
set -euo pipefail

# Offline-only creative search + simulator validation.
#
# Usage:
#   # ensure .env has META_ACCESS_TOKEN, META_ACCOUNT_ID (read-only) and GCP vars if needed
#   ./AELP2/tools/run_offline_cycle.sh [--candidates 300]

cd "$(dirname "$0")/../.."

CAND=300
while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidates) CAND="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

if [[ -f .env ]]; then
  set -a; source <(sed -n 's/^export //p' .env); set +a
fi

export OFFLINE_MODE=1
mkdir -p AELP2/reports AELP2/outputs/creative_candidates

echo "[1/4] Simulator fidelity v2.1 (read-only Graph fetch)"
python3 AELP2/tools/sim_fidelity_campaigns_temporal_v2.py || echo "(warn) fidelity v2.1 failed; check META_* envs"

echo "[2/4] Ad-level offline accuracy (requires historical creative outcomes)"
python3 AELP2/tools/ad_level_accuracy.py || true

echo "[2b] Fetch creative objects (read-only) and enrich"
python3 AELP2/tools/fetch_meta_creatives.py || true
python3 AELP2/tools/enrich_creatives_with_objects.py || true
python3 AELP2/tools/ad_level_calibration_v22.py || true
python3 AELP2/tools/ad_level_calibration_v23.py || true
python3 AELP2/tools/eval_ablation.py || true
python3 AELP2/tools/tune_generator_priors.py || true

echo "[3/4] Generate ${CAND} creative candidates (Aura products incl. Balance)"
python3 AELP2/tools/offline_creative_search.py || true

echo "[4/4] Summaries"
echo "- Fidelity:    AELP2/reports/sim_fidelity_campaigns_temporal_v2.json"
echo "- Ad-level:    AELP2/reports/ad_level_accuracy.json|csv"
echo "- Leaderboard: AELP2/reports/creative_leaderboard.json (top candidates)"
