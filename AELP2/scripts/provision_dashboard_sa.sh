#!/usr/bin/env bash
set -euo pipefail

# Provision a service account for the AELP2 dashboard and grant BigQuery read perms.
# Usage:
#   GOOGLE_CLOUD_PROJECT=aura-thrive-platform bash AELP2/scripts/provision_dashboard_sa.sh

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT" >&2; exit 2
fi

SA_ID="aelp2-dashboard-sa"
SA_EMAIL="$SA_ID@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com"

echo "Creating service account: $SA_EMAIL (idempotent)"
gcloud iam service-accounts create "$SA_ID" \
  --display-name="AELP2 Dashboard Service Account" \
  --project="$GOOGLE_CLOUD_PROJECT" || true

echo "Granting BigQuery read roles at project scope (dataViewer + jobUser)"
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/bigquery.dataViewer" >/dev/null
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/bigquery.jobUser" >/dev/null

echo "Service account ready: $SA_EMAIL"
echo "Export for deploy:"
echo "  export AELP2_DASHBOARD_SA=$SA_EMAIL"

