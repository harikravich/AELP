#!/usr/bin/env bash
set -euo pipefail

# Zero-prompt wrapper to create a Gmail child account under your manager.
# Reads creds from .env and maps GMAIL_* -> GOOGLE_ADS_* for this run.

ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

# Map Gmail creds if provided
export GOOGLE_ADS_DEVELOPER_TOKEN="${GOOGLE_ADS_DEVELOPER_TOKEN:-}"  # must be set in .env
export GOOGLE_ADS_CLIENT_ID="${GMAIL_CLIENT_ID:-${GOOGLE_ADS_CLIENT_ID:-}}"
export GOOGLE_ADS_CLIENT_SECRET="${GMAIL_CLIENT_SECRET:-${GOOGLE_ADS_CLIENT_SECRET:-}}"
export GOOGLE_ADS_REFRESH_TOKEN="${GMAIL_REFRESH_TOKEN:-${GOOGLE_ADS_REFRESH_TOKEN:-}}"
export GOOGLE_ADS_LOGIN_CUSTOMER_ID="${GOOGLE_ADS_LOGIN_CUSTOMER_ID:-9704174968}"

echo "Using manager (login_customer_id): $GOOGLE_ADS_LOGIN_CUSTOMER_ID"

# Ensure google-ads is installed
python3 - <<'PY' || true
try:
    import google.ads.googleads
    print("google-ads OK")
except Exception as e:
    import sys, subprocess
    print("Installing google-ads...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-ads"], stdout=subprocess.DEVNULL)
    print("google-ads installed")
PY

python3 scripts/create_gmail_child_account.py

