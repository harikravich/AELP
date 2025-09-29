#!/usr/bin/env bash
set -euo pipefail
# Usage: source ../scripts/gcp/vars.env; ./schedule_jobs.sh
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"

# Example: schedule meta_to_bq (dev)
gcloud scheduler jobs create http meta-to-bq-dev \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --schedule "0 */4 * * *" \
  --uri "https://cloudbuild.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/triggers/META_TO_BQ_DEV:run" \
  --http-method POST \
  --oauth-service-account-email aelp-ci@$PROJECT_ID.iam.gserviceaccount.com || true

echo "Scheduler stubs created (adjust trigger IDs)."

