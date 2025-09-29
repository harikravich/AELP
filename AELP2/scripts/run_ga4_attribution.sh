#!/usr/bin/env bash
set -euo pipefail

# Run GA4 lagged attribution importer over a date window.
# Defaults to last 28 days when no args provided.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Load env without overriding already-set vars
if [[ -f .env ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    line=${line#export } # drop optional export
    key=${line%%=*}
    val=${line#*=}
    [[ -z "$key" || -z "$val" ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

START="${1:-}"
END="${2:-}"
if [[ -z "$START" || -z "$END" ]]; then
  START=$(date -u -d '28 days ago' +%F)
  END=$(date -u +%F)
fi

echo "Running GA4 lagged attribution for $START..$END"
python3 -m AELP2.pipelines.ga4_lagged_attribution --start "$START" --end "$END"
echo "Done"
