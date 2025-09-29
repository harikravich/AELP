#!/usr/bin/env bash
set -euo pipefail
# Usage: source vars.env; ./create_buckets.sh
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET_PREFIX:?Set BUCKET_PREFIX}"
tiers=(dev stage prod)
suffixes=(reports artifacts)
for t in "${tiers[@]}"; do
  for s in "${suffixes[@]}"; do
    b="${BUCKET_PREFIX}-${t}-${s}"
    echo "Ensuring bucket: gs://$b"
    gsutil mb -p "$PROJECT_ID" -l "$REGION" -b on "gs://$b" 2>/dev/null || true
    # 30-day auto-delete temp artifacts (only for dev/stage reports)
    if [[ "$t" != "prod" ]]; then
      echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":30}}]}' > /tmp/lc.json
      gsutil lifecycle set /tmp/lc.json "gs://$b" || true
    fi
  done
done
echo "Buckets ensured."

