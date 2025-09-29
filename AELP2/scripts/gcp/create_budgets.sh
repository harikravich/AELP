#!/usr/bin/env bash
set -euo pipefail
# Usage: source vars.env; ./create_budgets.sh
: "${BILLING_ACCOUNT:?Set BILLING_ACCOUNT}"
NAME="AELP2 Prod"
gcloud beta billing budgets create \
  --display-name="$NAME" \
  --billing-account="$BILLING_ACCOUNT" \
  --amount-units=60000 \
  --threshold-rule-percent=0.5 \
  --threshold-rule-percent=0.8 \
  --threshold-rule-percent=1.0 \
  --all-updates-rule-pubsub-topic="" || true
echo "Budget ensured (check in Cloud Billing)."

