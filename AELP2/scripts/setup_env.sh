#!/usr/bin/env bash
set -euo pipefail

# Unified environment setup for AELP2 dashboard and services.
# - Supports profiles (slim/full) and Ads account variants (aura MCC vs gmail standalone).
# - Can print export lines for copy/paste, write a .env file, or apply to current shell when sourced.
#
# Usage:
#   # Print pasteable exports for slim dashboard (read-only BQ)
#   bash AELP2/scripts/setup_env.sh print-exports \
#     --project aura-thrive-platform --dataset gaelp_training --users-dataset gaelp_users \
#     --ga4 properties/308028264
#
#   # Apply to current shell (must source)
#   source AELP2/scripts/setup_env.sh apply --profile slim --project aura-thrive-platform \
#     --dataset gaelp_training --users-dataset gaelp_users --ga4 properties/308028264
#
#   # Include OpenAI and NextAuth (optional)
#   bash AELP2/scripts/setup_env.sh print-exports --profile full --openai-key "$OPENAI_API_KEY" \
#     --nextauth --allowed-domain aura.com
#
#   # Load Ads creds from env file (e.g., created by scripts/google_ads_gmail_setup.py)
#   bash AELP2/scripts/setup_env.sh print-exports --ads-env AELP2/config/.google_ads_credentials.gmail.env
#
#   # Write a consolidated .env safely (non-destructive if exists; use --force to overwrite)
#   bash AELP2/scripts/setup_env.sh write-dotenv .env \
#     --project aura-thrive-platform --dataset gaelp_training --users-dataset gaelp_users \
#     --ga4 properties/308028264 --openai-key "$OPENAI_API_KEY"

CMD="${1:-print-exports}"; shift || true

# Defaults
PROFILE="slim"                       # slim|full
PROJECT="aura-thrive-platform"
DATASET="gaelp_training"
USERS_DATASET="gaelp_users"
SANDBOX_DATASET=""                   # optional
GA4_PROP="properties/308028264"
OPENAI_KEY=""

USE_NEXTAUTH=0
ALLOWED_DOMAIN=""
NEXTAUTH_SECRET_VAL=""

ADS_ENV_FILE=""                      # optional: path to env file with GOOGLE_ADS_* vars

FORCE=0

die(){ echo "[env-setup] ERROR: $*" >&2; exit 2; }
log(){ echo "[env-setup] $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --users-dataset|--users_dataset) USERS_DATASET="$2"; shift 2;;
    --sandbox-dataset) SANDBOX_DATASET="$2"; shift 2;;
    --ga4|--ga4-property|--ga4-property-id) GA4_PROP="$2"; shift 2;;
    --openai-key) OPENAI_KEY="$2"; shift 2;;
    --nextauth) USE_NEXTAUTH=1; shift;;
    --allowed-domain) ALLOWED_DOMAIN="$2"; shift 2;;
    --nextauth-secret) NEXTAUTH_SECRET_VAL="$2"; shift 2;;
    --ads-env) ADS_ENV_FILE="$2"; shift 2;;
    --force) FORCE=1; shift;;
    *) die "Unknown arg: $1";;
  esac
done

gen_nextauth_secret(){
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import secrets; print(secrets.token_urlsafe(32))
PY
  elif command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 48 | tr -d '\n'
  else
    echo "$(date +%s)-$RANDOM-$RANDOM" | sha256sum | awk '{print $1}'
  fi
}

emit_exports(){
  local out=()
  out+=("export GOOGLE_CLOUD_PROJECT=$PROJECT")
  out+=("export BIGQUERY_TRAINING_DATASET=$DATASET")
  out+=("export BIGQUERY_USERS_DATASET=$USERS_DATASET")
  if [[ -n "$SANDBOX_DATASET" ]]; then out+=("export BIGQUERY_SANDBOX_DATASET=$SANDBOX_DATASET"); fi
  out+=("export GA4_PROPERTY_ID=$GA4_PROP")
  out+=("export NODE_ENV=production")

  if [[ -n "$OPENAI_KEY" ]]; then out+=("export OPENAI_API_KEY=$OPENAI_KEY"); fi

  if [[ $USE_NEXTAUTH -eq 1 ]]; then
    local sec="$NEXTAUTH_SECRET_VAL"; [[ -z "$sec" ]] && sec="$(gen_nextauth_secret)"
    out+=("export NEXTAUTH_SECRET=$sec")
    if [[ -n "$ALLOWED_DOMAIN" ]]; then out+=("export ALLOWED_EMAIL_DOMAIN=$ALLOWED_DOMAIN"); fi
  fi

  load_ads_from_file(){
    local f="$1"
    # shellcheck disable=SC2002
    while IFS='=' read -r k v; do
      [[ -z "$k" || "$k" =~ ^# ]] && continue
      case "$k" in
        GOOGLE_ADS_DEVELOPER_TOKEN|GOOGLE_ADS_CLIENT_ID|GOOGLE_ADS_CLIENT_SECRET|GOOGLE_ADS_REFRESH_TOKEN|GOOGLE_ADS_CUSTOMER_ID|GOOGLE_ADS_LOGIN_CUSTOMER_ID)
          out+=("export $k=$v");;
      esac
    done < <(cat "$f")
  }

  if [[ -n "$ADS_ENV_FILE" ]]; then
    [[ -f "$ADS_ENV_FILE" ]] || die "--ads-env file not found: $ADS_ENV_FILE"
    load_ads_from_file "$ADS_ENV_FILE"
  elif [[ -f .env ]]; then
    # Opportunistically load Ads vars from repo .env when present
    load_ads_from_file .env
  fi

  printf '%s\n' "${out[@]}"
}

write_dotenv(){
  local path="$1"
  if [[ -f "$path" && $FORCE -ne 1 ]]; then
    die "$path exists. Use --force to overwrite."
  fi
  emit_exports | sed 's/^export //g' > "$path"
  echo "Wrote $path"
}

apply_exports(){
  # To affect current shell, this script must be sourced.
  # Detect if sourced by comparing $0 and ${BASH_SOURCE[0]}.
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    die "Use: source AELP2/scripts/setup_env.sh apply [args]"
  fi
  eval "$(emit_exports)"
  echo "Applied environment to current shell."
}

case "$CMD" in
  print-exports)
    emit_exports
    ;;
  write-dotenv)
    write_dotenv "${1:-.env}"
    ;;
  apply)
    apply_exports
    ;;
  *)
    die "Unknown command: $CMD (use print-exports | write-dotenv | apply)"
    ;;
esac
