#!/usr/bin/env bash
set -euo pipefail

# Backfill Ads data for the last 3 years (or custom range) in monthly chunks.
# Uses MCC coordinator to iterate across child accounts per window.
#
# Usage examples:
#   bash AELP2/scripts/run_ads_backfill.sh --mcc 7762856866 --tasks "campaigns,ad_performance,keywords,search_terms,geo_device,adgroups,conversion_actions,conversion_action_stats"
#   bash AELP2/scripts/run_ads_backfill.sh --mcc 7762856866 --start 2022-01-01 --end 2024-08-31 --dry-run
#   bash AELP2/scripts/run_ads_backfill.sh --mcc 7762856866 --start 2025-06-01 --end $(date -u +%F) --tasks "conversion_action_stats" --skip 7941505199
#
# Env required: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET, Google Ads creds (see run_ads_ingestion.sh)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ -f .google_ads_credentials.env ]]; then set -a; source .google_ads_credentials.env; set +a; fi

MCC_ID="${GOOGLE_ADS_LOGIN_CUSTOMER_ID:-}"
TASKS="campaigns,ad_performance,keywords,search_terms,geo_device,adgroups,conversion_actions,conversion_action_stats"
START=""
END=""
DRY_RUN=0
ONLY_IDS=""
SKIP_IDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mcc) MCC_ID="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY_IDS="$2"; shift 2;;
    --skip) SKIP_IDS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

if [[ -z "$MCC_ID" ]]; then
  echo "Provide --mcc <id> or set GOOGLE_ADS_LOGIN_CUSTOMER_ID."
  exit 3
fi
export GOOGLE_ADS_LOGIN_CUSTOMER_ID="$MCC_ID"

# Default to last 36 months (inclusive)
if [[ -z "$END" ]]; then END=$(date -u +%F); fi
if [[ -z "$START" ]]; then START=$(date -u -d '36 months ago' +%F); fi

echo "Backfill Ads (MCC=$MCC_ID) $START..$END in monthly chunks"

# Compute month windows
current=$(date -u -d "${START}" +%Y-%m-01)
end_month=$(date -u -d "$(date -u -d "${END}" +%Y-%m-01)" +%Y-%m-01)

windows=()
while [[ "$current" < "$end_month" || "$current" == "$end_month" ]]; do
  y=$(date -u -d "$current" +%Y)
  m=$(date -u -d "$current" +%m)
  win_start="$current"
  win_end=$(date -u -d "$current +1 month -1 day" +%F)
  # Cap last window to END if within same month
  if [[ $(date -u -d "$win_end" +%F) > $(date -u -d "$END" +%F) ]]; then
    win_end=$END
  fi
  windows+=("$win_start:$win_end")
  current=$(date -u -d "$current +1 month" +%Y-%m-01)
done

echo "Planned windows (${#windows[@]}):"
printf '%s\n' "${windows[@]}"

if [[ $DRY_RUN -eq 1 ]]; then exit 0; fi

sleep_between_windows="${AELP2_BACKFILL_SLEEP_BETWEEN_WINDOWS:-5}"
echo "Running tasks: $TASKS"

for w in "${windows[@]}"; do
  s="${w%%:*}"; e="${w##*:}"
  echo "\n=== Window $s .. $e ==="
  PY_ARGS=(--start "$s" --end "$e" --tasks "$TASKS")
  if [[ -n "$ONLY_IDS" ]]; then PY_ARGS+=(--only "$ONLY_IDS"); fi
  if [[ -n "$SKIP_IDS" ]]; then PY_ARGS+=(--skip "$SKIP_IDS"); fi
  python3 -m AELP2.pipelines.ads_mcc_coordinator "${PY_ARGS[@]}" || true
  echo "Sleeping ${sleep_between_windows}s before next window..."
  sleep "$sleep_between_windows"
done

echo "Backfill complete."
