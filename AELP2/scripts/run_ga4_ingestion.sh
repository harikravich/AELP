#!/usr/bin/env bash
set -euo pipefail

# GA4 Aggregates Loader + View Refresh
#
# Usage examples:
#   bash AELP2/scripts/run_ga4_ingestion.sh --last28
#   bash AELP2/scripts/run_ga4_ingestion.sh --start 2024-08-01 --end 2024-08-31
#
# Requires:
#   - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
#   - GA4_PROPERTY_ID (format: properties/<id>)

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

START=""
END=""
LAST28=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --last28) LAST28=1; shift;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" || -z "${BIGQUERY_TRAINING_DATASET:-}" ]]; then
  echo "Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET" >&2
  exit 3
fi

if [[ -z "${GA4_PROPERTY_ID:-}" ]]; then
  echo "Missing GA4_PROPERTY_ID (format: properties/<id>)" >&2
  exit 4
fi

if [[ $LAST28 -eq 1 ]]; then
  START=$(date -u -d '28 days ago' +%F)
  END=$(date -u +%F)
fi

if [[ -z "$START" || -z "$END" ]]; then
  echo "Provide --start and --end or use --last28" >&2
  exit 5
fi

echo "Loading GA4 aggregates for $START..$END (property: $GA4_PROPERTY_ID)"
python3 -m AELP2.pipelines.ga4_to_bq --start "$START" --end "$END"

echo "Refreshing BQ views (including ga4_daily if present)"
python3 -m AELP2.pipelines.create_bq_views

echo "Done."
