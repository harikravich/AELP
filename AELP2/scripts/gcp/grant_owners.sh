#!/usr/bin/env bash
set -euo pipefail
# Usage: source vars.env; ./grant_owners.sh
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${ADMINS:?Set ADMINS (space-separated emails)}"
for U in $ADMINS; do
  echo "Granting roles/owner to $U on $PROJECT_ID"
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="user:$U" --role="roles/owner" --quiet
done
echo "Done."

