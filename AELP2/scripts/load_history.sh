#!/usr/bin/env bash
set -euo pipefail

# Load historical data from Google Ads and GA4 into BigQuery
# Usage: AELP2/scripts/load_history.sh 2024-06-01 2024-08-31

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <start YYYY-MM-DD> <end YYYY-MM-DD>"
  exit 1
fi

START="$1"
END="$2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

set -a
[[ -f .env ]] && source .env || true
[[ -f AELP2/config/.env.aelp2 ]] && source AELP2/config/.env.aelp2 || true
set +a

echo "Loading Google Ads performance into BigQuery..."
python3 -m AELP2.pipelines.google_ads_to_bq --start "$START" --end "$END"

echo "Loading GA4 aggregates into BigQuery..."
if [[ -z "${GA4_PROPERTY_ID:-}" ]]; then
  echo "Skipping GA4 aggregates load (GA4_PROPERTY_ID not set)."
else
  python3 -m AELP2.pipelines.ga4_to_bq --start "$START" --end "$END"
fi

echo "Done."
