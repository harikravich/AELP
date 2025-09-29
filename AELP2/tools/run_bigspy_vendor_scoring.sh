#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source .env
#   bash AELP2/tools/run_bigspy_vendor_scoring.sh AELP2/secrets/bigspy_cookies.txt

COOKIES=${1:-}
if [[ -z "$COOKIES" ]]; then
  echo "Usage: $0 PATH_TO_COOKIES_TXT" >&2
  exit 2
fi

python3 AELP2/tools/bigspy_auto_export.py --cookies "$COOKIES" --filters AELP2/config/bigspy_filters.yaml --max 800
python3 AELP2/tools/import_vendor_meta_creatives.py --src AELP2/vendor_imports
python3 AELP2/tools/build_features_from_creative_objects.py
python3 AELP2/tools/score_vendor_creatives.py

echo "Done. See AELP2/reports/vendor_scores.json"

