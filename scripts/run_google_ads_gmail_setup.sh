#!/usr/bin/env bash
set -euo pipefail

echo "==============================================="
echo "Google Ads Gmail Setup (Interactive, no flags)"
echo "==============================================="

# Defaults
DEFAULT_ENV_OUT="AELP2/config/.google_ads_credentials.gmail.env"
if [ ! -d "AELP2/config" ]; then
  DEFAULT_ENV_OUT=".google_ads_credentials.gmail.env"
fi
DEFAULT_CUSTOMER_ID="9704174968"
DEFAULT_CAMPAIGN_NAME="Aura Sandbox - Identity Protection (Gmail)"
DEFAULT_DAILY_BUDGET="50"
DEFAULT_FINAL_URL="https://your-sandbox-landing.example.com/?stream=gmail"
DEFAULT_UTM_SUFFIX="utm_source=google&utm_medium=cpc&utm_campaign={campaignid}&utm_content={creative}&utm_term={keyword}&stream=gmail"

read -r -p "Developer token: " DEV_TOKEN
read -r -p "OAuth Client ID (reuse Aura project): " CLIENT_ID
read -r -p "OAuth Client Secret: " CLIENT_SECRET

read -r -p "Google Ads Customer ID (no dashes) [${DEFAULT_CUSTOMER_ID}]: " CUSTOMER_ID
CUSTOMER_ID=${CUSTOMER_ID:-$DEFAULT_CUSTOMER_ID}

read -r -p "MCC Login Customer ID (optional; blank for standalone): " LOGIN_CUSTOMER_ID || true
read -r -p "Env file output path [${DEFAULT_ENV_OUT}]: " ENV_OUT
ENV_OUT=${ENV_OUT:-$DEFAULT_ENV_OUT}

echo ""
read -r -p "Create a paused Search campaign now? (y/N): " CREATE
CREATE=${CREATE:-N}

CREATE_ARGS=()
if [[ "$CREATE" =~ ^[Yy]$ ]]; then
  read -r -p "Campaign name [${DEFAULT_CAMPAIGN_NAME}]: " CAMPAIGN_NAME
  CAMPAIGN_NAME=${CAMPAIGN_NAME:-$DEFAULT_CAMPAIGN_NAME}
  read -r -p "Daily budget USD [${DEFAULT_DAILY_BUDGET}]: " DAILY_BUDGET
  DAILY_BUDGET=${DAILY_BUDGET:-$DEFAULT_DAILY_BUDGET}
  read -r -p "Final URL [${DEFAULT_FINAL_URL}]: " FINAL_URL
  FINAL_URL=${FINAL_URL:-$DEFAULT_FINAL_URL}
  read -r -p "UTM suffix [${DEFAULT_UTM_SUFFIX}]: " UTM_SUFFIX
  UTM_SUFFIX=${UTM_SUFFIX:-$DEFAULT_UTM_SUFFIX}
  CREATE_ARGS=(
    --create-campaign
    --campaign-name "$CAMPAIGN_NAME"
    --daily-budget "$DAILY_BUDGET"
    --final-url "$FINAL_URL"
    --utm-suffix "$UTM_SUFFIX"
  )
fi

echo ""
echo "Installing prerequisites if needed..."
python3 - <<'PY'
import sys
import subprocess
def ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg], stdout=subprocess.DEVNULL)
for p in ('requests',):
    ensure(p)
print('OK')
PY

CMD=(
  python3 scripts/google_ads_gmail_setup.py
  --developer-token "$DEV_TOKEN"
  --client-id "$CLIENT_ID"
  --client-secret "$CLIENT_SECRET"
  --customer-id "$CUSTOMER_ID"
  --env-out "$ENV_OUT"
)

if [ -n "${LOGIN_CUSTOMER_ID:-}" ]; then
  CMD+=(--login-customer-id "$LOGIN_CUSTOMER_ID")
fi

if [ ${#CREATE_ARGS[@]} -gt 0 ]; then
  CMD+=("${CREATE_ARGS[@]}")
fi

echo ""
echo "Running setup..."
"${CMD[@]}"

echo ""
echo "Done. To use these Gmail creds in your shell:"
echo "  set -a; source $ENV_OUT; set +a"
echo "  unset GOOGLE_ADS_LOGIN_CUSTOMER_ID  # keep unset for standalone Gmail"

