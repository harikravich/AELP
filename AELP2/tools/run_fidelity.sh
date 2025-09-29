#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export META_ACCESS_TOKEN=... META_ACCOUNT_ID=...
#   export GOOGLE_CLOUD_PROJECT=... BIGQUERY_TRAINING_DATASET=...
#   /opt/aelp/venvs/aelp-heavy/bin/python -c 'import sys; print(sys.version)'
#   ./AELP2/tools/run_fidelity.sh 14

DAYS=${1:-14}

if [[ -f .env ]]; then
  set -a; source <(sed -n 's/^export //p' .env); set +a
fi

if [[ -z "${META_ACCESS_TOKEN:-}" || -z "${META_ACCOUNT_ID:-}" ]]; then
  echo "Missing META_ACCESS_TOKEN or META_ACCOUNT_ID (set in env or .env)" >&2
  exit 1
fi

VENV=${VENV:-/opt/aelp/venvs/aelp-heavy}
source "$VENV/bin/activate"

PYTHONPATH=. AELP2_FIDELITY_DAYS="$DAYS" python AELP2/tools/sim_fidelity_eval.py

echo "Report: AELP2/reports/sim_fidelity.json"

