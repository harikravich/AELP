#!/usr/bin/env bash
set -euo pipefail
# Usage: source vars.env; ./create_datasets.sh
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${DATASETS:?Set DATASETS}"
for DS in $DATASETS; do
  echo "Creating dataset $PROJECT_ID:$DS (if not exists)"
  bq --project_id="$PROJECT_ID" mk --location=US "$PROJECT_ID:$DS" >/dev/null 2>&1 || true
done
echo "Datasets ensured: $DATASETS"

