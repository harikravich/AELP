#!/usr/bin/env bash

# Re-exec with bash if invoked via sh/dash so pipefail works
if [ -z "${BASH_VERSION:-}" ]; then
  if command -v bash >/dev/null 2>&1; then
    exec bash "$0" "$@"
  else
    echo "[deploy] ERROR: bash is required. Run with: bash $0 ..." >&2
    exit 1
  fi
fi

set -euo pipefail

# Build and deploy the AELP2 dashboard to Cloud Run with one command.
#
# Usage examples:
#   bash AELP2/scripts/deploy_dashboard.sh \
#     --project aura-thrive-platform --region us-central1 --service aelp2-dashboard \
#     --dataset gaelp_training --users_dataset gaelp_users --ga4 properties/308028264
#
# Optional (reads from your shell env or .env if present):
#   --use-ads-env   # applies GOOGLE_ADS_* + GOOGLE_ADS_LOGIN_CUSTOMER_ID
#   --use-openai    # applies OPENAI_API_KEY
#   --tail-logs     # tails Cloud Run logs after deploy

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Defaults
PROJECT=""
REGION="us-central1"
SERVICE="aelp2-dashboard"
DATASET="gaelp_training"
USERS_DATASET="gaelp_users"
GA4_PROP=""
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
USE_ADS_ENV=0
USE_OPENAI=0
TAIL_LOGS=0

# Load .env if present (non‑destructive)
if [[ -f .env ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    line="${line#export }"
    key="${line%%=*}"; val="${line#*=}"
    [[ -z "$key" ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

die() { echo "[deploy] ERROR: $*" >&2; exit 1; }
log() { echo "[deploy] $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --service) SERVICE="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --users_dataset) USERS_DATASET="$2"; shift 2;;
    --ga4) GA4_PROP="$2"; shift 2;;
    --tag) IMAGE_TAG="$2"; shift 2;;
    --use-ads-env) USE_ADS_ENV=1; shift;;
    --use-openai) USE_OPENAI=1; shift;;
    --slim) export SLIM=1; shift;;
    --tail-logs) TAIL_LOGS=1; shift;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -z "$PROJECT" ]] && PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
[[ -z "$PROJECT" ]] && die "--project or GOOGLE_CLOUD_PROJECT required"

[[ -z "$GA4_PROP" ]] && GA4_PROP="${GA4_PROPERTY_ID:-}"
[[ -z "$GA4_PROP" ]] && log "GA4 property not provided; GA4 buttons may be disabled"

log "Project=$PROJECT  Region=$REGION  Service=$SERVICE  Tag=$IMAGE_TAG"
log "Datasets: training=$DATASET users=$USERS_DATASET  GA4=$GA4_PROP"

command -v gcloud >/dev/null 2>&1 || die "gcloud not found"

# Build container using Cloud Build config (full by default; --slim for UI‑only)
CB_CONFIG="AELP2/apps/dashboard/cloudbuild.yaml"
if [[ "${SLIM:-0}" -eq 1 ]]; then CB_CONFIG="AELP2/apps/dashboard/cloudbuild.slim.yaml"; fi
log "Building container image via Cloud Build ($CB_CONFIG)…"
SRC_DIR="."
if [[ "${SLIM:-0}" -eq 1 ]]; then SRC_DIR="AELP2/apps/dashboard"; fi
gcloud builds submit "$SRC_DIR" \
  --project "$PROJECT" \
  --config "$CB_CONFIG" \
  --substitutions _SERVICE="$SERVICE",_IMAGE_TAG="$IMAGE_TAG"

IMAGE="gcr.io/$PROJECT/$SERVICE:$IMAGE_TAG"

# Deploy to Cloud Run with core env
log "Deploying $SERVICE to Cloud Run (image: $IMAGE)…"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --image "$IMAGE" \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT="$PROJECT",BIGQUERY_TRAINING_DATASET="$DATASET",BIGQUERY_USERS_DATASET="$USERS_DATASET",GA4_PROPERTY_ID="$GA4_PROP",NODE_ENV=production

# Optionally apply Ads and Chat envs from shell/.env
ENV_UPDATE_ARGS=()
if [[ $USE_ADS_ENV -eq 1 ]]; then
  for v in GOOGLE_ADS_DEVELOPER_TOKEN GOOGLE_ADS_CLIENT_ID GOOGLE_ADS_CLIENT_SECRET GOOGLE_ADS_REFRESH_TOKEN GOOGLE_ADS_LOGIN_CUSTOMER_ID; do
    [[ -z "${!v:-}" ]] && die "--use-ads-env set but $v is not in your environment (.env or shell)"
    ENV_UPDATE_ARGS+=("$v=${!v}")
  done
fi
if [[ $USE_OPENAI -eq 1 ]]; then
  [[ -z "${OPENAI_API_KEY:-}" ]] && die "--use-openai set but OPENAI_API_KEY is not in your environment"
  ENV_UPDATE_ARGS+=("OPENAI_API_KEY=${OPENAI_API_KEY}")
fi

if [[ ${#ENV_UPDATE_ARGS[@]} -gt 0 ]]; then
  log "Updating service env: ${#ENV_UPDATE_ARGS[@]} keys"
  gcloud run services update "$SERVICE" \
    --project "$PROJECT" \
    --region "$REGION" \
    --update-env-vars "$(IFS=, ; echo "${ENV_UPDATE_ARGS[*]}")"
fi

# Show URL and latest revision
URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')
REV=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.latestReadyRevisionName)')
log "Live URL: $URL"
log "Revision: $REV"

if [[ $TAIL_LOGS -eq 1 ]]; then
  log "Tailing Cloud Run logs (Ctrl+C to stop)…"
  # Prefer GA command for Cloud Run log tailing; fall back to Logging API if unavailable.
  if gcloud run services logs tail --help >/dev/null 2>&1; then
    gcloud run services logs tail "$SERVICE" --project "$PROJECT" --region "$REGION"
  else
    # Fallback for older gcloud versions
    gcloud logging read \
      --project "$PROJECT" \
      --format='value(textPayload)' \
      --limit 100 \
      --freshness=1h \
      "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE" || true
  fi
fi

log "Done."
