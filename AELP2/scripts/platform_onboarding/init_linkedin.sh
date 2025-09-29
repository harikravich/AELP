#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_DIR="$ROOT_DIR/AELP2/config"
mkdir -p "$CONFIG_DIR"
OUT="$CONFIG_DIR/.linkedin_credentials.env"

echo "[LINKEDIN] Creating credentials file at $OUT"
{
  echo "export LINKEDIN_CLIENT_ID=\"${LINKEDIN_CLIENT_ID:-}\""
  echo "export LINKEDIN_CLIENT_SECRET=\"${LINKEDIN_CLIENT_SECRET:-}\""
  echo "export LINKEDIN_ACCESS_TOKEN=\"${LINKEDIN_ACCESS_TOKEN:-}\""
  echo "export LINKEDIN_ACCOUNT_ID=\"${LINKEDIN_ACCOUNT_ID:-}\""
} > "$OUT"

if [[ -z "${LINKEDIN_CLIENT_ID:-}" || -z "${LINKEDIN_CLIENT_SECRET:-}" || -z "${LINKEDIN_ACCESS_TOKEN:-}" ]]; then
  echo "[LINKEDIN] WARNING: One or more required env vars missing. Fill $OUT before ingestion."
else
  echo "[LINKEDIN] Credentials captured."
fi
echo "[LINKEDIN] Done."

