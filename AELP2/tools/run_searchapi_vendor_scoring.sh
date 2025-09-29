#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source .env
#   bash AELP2/tools/run_searchapi_vendor_scoring.sh GB,US 365 200

COUNTRIES=${1:-GB}
DAYS=${2:-365}
MAXPQ=${3:-200}

if [[ -z "${SEARCHAPI_API_KEY:-}" ]]; then
  echo "Missing SEARCHAPI_API_KEY in environment. Edit .env and retry." >&2
  exit 2
fi

python3 AELP2/tools/fetch_searchapi_meta.py \
  --filters AELP2/config/bigspy_filters.yaml \
  --countries "$COUNTRIES" \
  --days "$DAYS" \
  --max-per-query "$MAXPQ"

python3 AELP2/tools/import_vendor_meta_creatives.py --src AELP2/vendor_imports
python3 AELP2/tools/build_features_from_creative_objects.py
python3 AELP2/tools/score_vendor_creatives.py

echo "Done. See AELP2/reports/vendor_scores.json"

