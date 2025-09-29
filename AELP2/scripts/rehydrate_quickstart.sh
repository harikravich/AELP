#!/usr/bin/env bash
set -euo pipefail

# Quick rehydrate and run: loads .env, optional container/live flags, then runs orchestrator.
# Usage:
#   bash AELP2/scripts/rehydrate_quickstart.sh [--containers] [--live] [--env PATH_TO_ENV]

CONTAINERS=0
LIVE=0
ENV_FILE=".env"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --containers)
      CONTAINERS=1; shift ;;
    --live)
      LIVE=1; shift ;;
    --env)
      ENV_FILE=${2:-.env}; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

echo "Loading env from $ENV_FILE"
set -a; source "$ENV_FILE"; set +a

# Defaults/toggles
export AELP2_GX_USE_BQ=${AELP2_GX_USE_BQ:-1}
export AELP2_GX_CLICK_IMP_TOLERANCE=${AELP2_GX_CLICK_IMP_TOLERANCE:-0.05}
export AELP2_GX_MAX_VIOLATION_ROWS_PCT=${AELP2_GX_MAX_VIOLATION_ROWS_PCT:-0.10}

if [[ ${CONTAINERS} -eq 1 ]]; then
  export AELP2_ROBYN_ALLOW_RUN=1
  export AELP2_CA_ALLOW_RUN=1
  echo "Containers enabled: Robyn + ChannelAttribution"
fi

if [[ -n "${GA4_SERVICE_ACCOUNT_JSON:-}" && -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$GA4_SERVICE_ACCOUNT_JSON"
fi

if [[ ${LIVE} -eq 1 ]]; then
  export AELP2_ALLOW_GOOGLE_MUTATIONS=1
  export AELP2_ALLOW_VALUE_UPLOADS=1
  export AELP2_VALUE_UPLOAD_DRY_RUN=0
  echo "Live mutations enabled (be careful)."
else
  export AELP2_ALLOW_GOOGLE_MUTATIONS=${AELP2_ALLOW_GOOGLE_MUTATIONS:-0}
  export AELP2_ALLOW_VALUE_UPLOADS=${AELP2_ALLOW_VALUE_UPLOADS:-0}
  export AELP2_VALUE_UPLOAD_DRY_RUN=${AELP2_VALUE_UPLOAD_DRY_RUN:-1}
fi

echo "Project: ${GOOGLE_CLOUD_PROJECT:-unset}"
echo "Training dataset: ${BIGQUERY_TRAINING_DATASET:-unset}  | Users dataset: ${BIGQUERY_USERS_DATASET:-unset}"

python3 AELP2/scripts/e2e_orchestrator.py --auto

