#!/usr/bin/env bash
set -euo pipefail

# Create Cloud Scheduler jobs for nightly pipelines (if Cloud Scheduler API enabled).
# Requires: gcloud configured with project and permissions.

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT" >&2; exit 2
fi

REGION=${REGION:-us-central1}

echo "Creating Cloud Scheduler job: aelp2-nightly"
gcloud scheduler jobs create http aelp2-nightly \
  --location="$REGION" \
  --schedule="15 2 * * *" \
  --uri="https://example.com/placeholder" \
  --http-method=POST || true

echo "Note: Replace the URI with a Cloud Run Jobs/Workflows trigger or use cron on the VM."

