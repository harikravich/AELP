#!/usr/bin/env bash
set -euo pipefail

# Orchestrate Google Ads MCC ingestions with sane quoting and env handling.
#
# Usage examples:
#   bash AELP2/scripts/run_ads_ingestion.sh --mcc 7762856866 --start 2024-07-01 --end 2024-07-31 \
#     --tasks "campaigns,keywords,search_terms,geo_device,adgroups,conversion_actions,ad_performance"
#
#   bash AELP2/scripts/run_ads_ingestion.sh --mcc 7762856866 --last14 --tasks "ad_performance"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Load project env without overriding already-set vars
if [[ -f .env ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    line="${line#export }"
    key="${line%%=*}"; val="${line#*=}"
    [[ -z "$key" ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

# Load Google Ads creds if present
if [[ -f .google_ads_credentials.env ]]; then
  set -a; source .google_ads_credentials.env; set +a
fi

# Defaults
MCC_ID="${GOOGLE_ADS_LOGIN_CUSTOMER_ID:-}"
START=""
END=""
TASKS="campaigns,keywords,search_terms,geo_device,adgroups,conversion_actions,conversion_action_stats,ad_performance"
ONLY_IDS=""
SKIP_IDS=""
LAST14=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mcc) MCC_ID="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --last14) LAST14=1; shift;;
    --only) ONLY_IDS="$2"; shift 2;;
    --skip) SKIP_IDS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

# Validate required env
REQUIRED_VARS=(GOOGLE_CLOUD_PROJECT BIGQUERY_TRAINING_DATASET GOOGLE_ADS_DEVELOPER_TOKEN GOOGLE_ADS_CLIENT_ID GOOGLE_ADS_CLIENT_SECRET GOOGLE_ADS_REFRESH_TOKEN)
for v in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "Missing required env var: $v" >&2
    exit 3
  fi
done

if [[ -z "$MCC_ID" ]]; then
  echo "Missing MCC ID. Provide --mcc <id> or set GOOGLE_ADS_LOGIN_CUSTOMER_ID." >&2
  exit 4
fi

export GOOGLE_ADS_LOGIN_CUSTOMER_ID="$MCC_ID"

echo "Listing child accounts under MCC $MCC_ID ..."
python3 -m AELP2.pipelines.google_ads_mcc_to_bq --start "${START:-2024-01-01}" --end "${END:-2024-01-31}" --list-only || true

if [[ $LAST14 -eq 1 ]]; then
  START=$(date -u -d '14 days ago' +%F)
  END=$(date -u +%F)
fi

if [[ -z "$START" || -z "$END" ]]; then
  echo "Provide --start and --end (YYYY-MM-DD) or use --last14" >&2
  exit 5
fi

echo "Running MCC coordinator $START..$END with tasks: $TASKS"
export AELP2_ADS_MCC_DELAY_SECONDS="${AELP2_ADS_MCC_DELAY_SECONDS:-2.0}"
PY_ARGS=(--start "$START" --end "$END" --tasks "$TASKS")
if [[ -n "$ONLY_IDS" ]]; then PY_ARGS+=(--only "$ONLY_IDS"); fi
if [[ -n "$SKIP_IDS" ]]; then PY_ARGS+=(--skip "$SKIP_IDS"); fi
python3 -m AELP2.pipelines.ads_mcc_coordinator "${PY_ARGS[@]}"

echo "Done."
