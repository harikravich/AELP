#!/usr/bin/env bash
set -euo pipefail

# Meta onboarding helper: writes creds to AELP2/config/.meta_credentials.env
# Expects env vars: META_APP_ID, META_APP_SECRET, META_ACCESS_TOKEN, META_ACCOUNT_ID (optional)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_DIR="$ROOT_DIR/AELP2/config"
mkdir -p "$CONFIG_DIR"
OUT="$CONFIG_DIR/.meta_credentials.env"

echo "[META] Creating credentials file at $OUT"
{
  echo "export META_APP_ID=\"${META_APP_ID:-}\""
  echo "export META_APP_SECRET=\"${META_APP_SECRET:-}\""
  echo "export META_ACCESS_TOKEN=\"${META_ACCESS_TOKEN:-}\""
  echo "export META_ACCOUNT_ID=\"${META_ACCOUNT_ID:-}\""
} > "$OUT"

if [[ -z "${META_APP_ID:-}" || -z "${META_APP_SECRET:-}" || -z "${META_ACCESS_TOKEN:-}" ]]; then
  echo "[META] WARNING: One or more required env vars missing. Fill $OUT before ingestion."
else
  echo "[META] Credentials captured."
fi
echo "[META] Done."

