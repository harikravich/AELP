#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$PWD}
OUT=${OUT:-artifacts/meta}

if [[ -z "${META_ACCESS_TOKEN:-}" || -z "${META_ACCOUNT_ID:-}" ]]; then
  echo "META_ACCESS_TOKEN or META_ACCOUNT_ID missing; source your .env.local first" >&2
  exit 1
fi

mkdir -p "$OUT"
PYTHONPATH="$ROOT" python3 tools/meta/fetch_ads_and_insights.py \
  --start 2025-09-25 --end 2025-10-01 --limit 50 --date-preset last_90d --outdir "$OUT"

echo "OK: wrote $(wc -l < "$OUT/insights.csv") insight rows to $OUT"

