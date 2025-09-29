#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_DIR="$ROOT_DIR/AELP2/config"
mkdir -p "$CONFIG_DIR"
OUT="$CONFIG_DIR/.tiktok_credentials.env"

echo "[TIKTOK] Creating credentials file at $OUT"
{
  echo "export TIKTOK_APP_ID=\"${TIKTOK_APP_ID:-}\""
  echo "export TIKTOK_SECRET=\"${TIKTOK_SECRET:-}\""
  echo "export TIKTOK_ACCESS_TOKEN=\"${TIKTOK_ACCESS_TOKEN:-}\""
  echo "export TIKTOK_ADVERTISER_ID=\"${TIKTOK_ADVERTISER_ID:-}\""
} > "$OUT"

if [[ -z "${TIKTOK_APP_ID:-}" || -z "${TIKTOK_SECRET:-}" || -z "${TIKTOK_ACCESS_TOKEN:-}" ]]; then
  echo "[TIKTOK] WARNING: One or more required env vars missing. Fill $OUT before ingestion."
else
  echo "[TIKTOK] Credentials captured."
fi
echo "[TIKTOK] Done."

