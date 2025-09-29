#!/usr/bin/env bash
set -euo pipefail
# Usage: source vars.env; ./create_secrets.sh
: "${PROJECT_ID:?Set PROJECT_ID}"
secrets=(SEARCHAPI_API_KEY META_ACCESS_TOKEN META_ADLIBRARY_ACCESS_TOKEN GOOGLE_ADS_DEVELOPER_TOKEN)
for s in "${secrets[@]}"; do
  echo "Ensuring secret: $s"
  gcloud secrets describe "$s" --project "$PROJECT_ID" >/dev/null 2>&1 || gcloud secrets create "$s" --replication-policy="automatic" --project "$PROJECT_ID"
done
echo "Secrets ensured (values not set by script)."

