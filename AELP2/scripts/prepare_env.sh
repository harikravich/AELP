#!/usr/bin/env bash

# Prepare environment for AELP2 dashboard and optionally deploy to Cloud Run.
# - Detects existing creds (GA4 service account, Ads profiles) and sets env.
# - Can write a consolidated .env and/or kick off deploy (slim or full).
#
# Examples:
#   # Prepare env (slim, no Ads) and write .env; do not deploy
#   bash AELP2/scripts/prepare_env.sh --project aura-thrive-platform --dataset gaelp_training \
#     --users-dataset gaelp_users --ga4 properties/308028264 --write-dotenv
#
#   # Prepare env using Aura MCC Ads from repo .env and deploy slim
#   bash AELP2/scripts/prepare_env.sh --ads aura --deploy slim --public
#
#   # Prepare env using Gmail Ads profile env file and deploy full (includes Python)
#   bash AELP2/scripts/prepare_env.sh --ads gmail \
#     --ads-env AELP2/config/.google_ads_credentials.gmail.env \
#     --deploy full --public

set -euo pipefail

# Re-exec with bash if invoked via sh/dash
if [ -z "${BASH_VERSION:-}" ]; then
  if command -v bash >/dev/null 2>&1; then exec bash "$0" "$@"; else echo "bash required" >&2; exit 1; fi
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

PROJECT="${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}"
DATASET="${BIGQUERY_TRAINING_DATASET:-gaelp_training}"
USERS_DATASET="${BIGQUERY_USERS_DATASET:-gaelp_users}"
GA4_PROP="${GA4_PROPERTY_ID:-properties/308028264}"
ADS_PROFILE="none"         # none|aura|gmail
ADS_ENV_FILE=""            # optional env file for gmail profile
OPENAI_KEY="${OPENAI_API_KEY:-}"  # pass or inherit
USE_NEXTAUTH=0
ALLOWED_DOMAIN=""
WRITE_DOTENV=0
FORCE=0
DEPLOY=""                  # ''|slim|full
PUBLIC=0                    # add invoker binding
TAIL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --users-dataset|--users_dataset) USERS_DATASET="$2"; shift 2;;
    --ga4|--ga4-property) GA4_PROP="$2"; shift 2;;
    --ads) ADS_PROFILE="$2"; shift 2;;
    --ads-env) ADS_ENV_FILE="$2"; shift 2;;
    --openai-key) OPENAI_KEY="$2"; shift 2;;
    --nextauth) USE_NEXTAUTH=1; shift;;
    --allowed-domain) ALLOWED_DOMAIN="$2"; shift 2;;
    --write-dotenv) WRITE_DOTENV=1; shift;;
    --force) FORCE=1; shift;;
    --deploy) DEPLOY="$2"; shift 2;;
    --public) PUBLIC=1; shift;;
    --tail-logs|--tail) TAIL=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

EXTRA_ARGS=()
if [[ "$USE_NEXTAUTH" -eq 1 ]]; then EXTRA_ARGS+=(--nextauth); fi
if [[ -n "$ALLOWED_DOMAIN" ]]; then EXTRA_ARGS+=(--allowed-domain "$ALLOWED_DOMAIN"); fi
if [[ -n "$OPENAI_KEY" ]]; then EXTRA_ARGS+=(--openai-key "$OPENAI_KEY"); fi

# Choose Ads env source
case "$ADS_PROFILE" in
  none) ;;
  aura)
    # Use repo .env for Aura MCC. Validate presence of LOGIN_CUSTOMER_ID.
    if command -v rg >/dev/null 2>&1; then
      if ! rg -n "^GOOGLE_ADS_LOGIN_CUSTOMER_ID=" -S "$ROOT_DIR/.env" >/dev/null 2>&1; then
        echo "[prepare] WARNING: .env missing GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC). You can still deploy slim without --use-ads-env." >&2
      fi
    else
      if ! grep -q "^GOOGLE_ADS_LOGIN_CUSTOMER_ID=" "$ROOT_DIR/.env" 2>/dev/null; then
        echo "[prepare] WARNING: .env missing GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC). You can still deploy slim without --use-ads-env." >&2
      fi
    fi
    ;;
  gmail)
    [[ -n "$ADS_ENV_FILE" ]] || ADS_ENV_FILE="$ROOT_DIR/AELP2/config/.google_ads_credentials.gmail.env"
    if [[ ! -f "$ADS_ENV_FILE" ]]; then
      echo "[prepare] ERROR: Gmail Ads env file not found: $ADS_ENV_FILE" >&2
      exit 3
    fi
    EXTRA_ARGS+=(--ads-env "$ADS_ENV_FILE")
    ;;
  *) echo "[prepare] ERROR: --ads must be one of none|aura|gmail" >&2; exit 2;;
esac

SETUP_BIN="$ROOT_DIR/AELP2/scripts/setup_env.sh"
[[ -x "$SETUP_BIN" ]] || { echo "[prepare] Missing $SETUP_BIN" >&2; exit 1; }

# Write .env if requested
if [[ "$WRITE_DOTENV" -eq 1 ]]; then
  "$SETUP_BIN" write-dotenv "$ROOT_DIR/.env" --force \
    --project "$PROJECT" --dataset "$DATASET" --users-dataset "$USERS_DATASET" --ga4 "$GA4_PROP" \
    ${EXTRA_ARGS[@]:-}
fi

# Export env to this process for optional deploy
EXPORTS=$("$SETUP_BIN" print-exports --project "$PROJECT" --dataset "$DATASET" --users-dataset "$USERS_DATASET" --ga4 "$GA4_PROP" "${EXTRA_ARGS[@]}" )
eval "$EXPORTS"

SERVICE="aelp2-dashboard"
REGION="${REGION:-us-central1}"

# Optional: make service public (invoker binding)
if [[ "$PUBLIC" -eq 1 ]]; then
  if command -v gcloud >/dev/null 2>&1; then
    gcloud beta run services add-iam-policy-binding "$SERVICE" \
      --project "$PROJECT" --region "$REGION" \
      --member='allUsers' --role='roles/run.invoker' || true
  else
    echo "[prepare] gcloud not found; skipping public binding" >&2
  fi
fi

if [[ -n "$DEPLOY" ]]; then
  DEPLOY_ARGS=( --project "$PROJECT" --region "$REGION" --service "$SERVICE" \
                --dataset "$DATASET" --users_dataset "$USERS_DATASET" --ga4 "$GA4_PROP" )
  if [[ "$DEPLOY" == "slim" ]]; then DEPLOY_ARGS+=(--slim); fi
  if [[ "$TAIL" -eq 1 ]]; then DEPLOY_ARGS+=(--tail-logs); fi

  case "$ADS_PROFILE" in
    aura)
      # Only apply --use-ads-env if LOGIN_CUSTOMER_ID is present
      if [[ -n "${GOOGLE_ADS_LOGIN_CUSTOMER_ID:-}" ]]; then DEPLOY_ARGS+=(--use-ads-env); fi
      ;;
    gmail)
      # For gmail env file, we applied OPENAI only; Ads vars will be injected via services update only if needed later
      ;;
  esac

  # Include OpenAI if present
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then DEPLOY_ARGS+=(--use-openai); fi

  "$ROOT_DIR/AELP2/scripts/deploy_dashboard.sh" "${DEPLOY_ARGS[@]}"
fi

echo "[prepare] Done. Current effective env:"
echo "  GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
echo "  BIGQUERY_TRAINING_DATASET=$BIGQUERY_TRAINING_DATASET"
echo "  BIGQUERY_USERS_DATASET=$BIGQUERY_USERS_DATASET"
echo "  GA4_PROPERTY_ID=${GA4_PROPERTY_ID:-}"
if [[ -n "${GOOGLE_ADS_CUSTOMER_ID:-}" ]]; then echo "  GOOGLE_ADS_CUSTOMER_ID=$GOOGLE_ADS_CUSTOMER_ID"; fi
if [[ -n "${GOOGLE_ADS_LOGIN_CUSTOMER_ID:-}" ]]; then echo "  GOOGLE_ADS_LOGIN_CUSTOMER_ID=$GOOGLE_ADS_LOGIN_CUSTOMER_ID"; fi
