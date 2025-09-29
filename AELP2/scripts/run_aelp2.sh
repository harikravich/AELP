#!/usr/bin/env bash
set -euo pipefail

# AELP2 full run script
# - Loads env from .env (repo root) or AELP2/config/.env.aelp2
# - Validates required variables
# - Optionally builds calibration reference (orchestrator can also do it)
# - Runs the production orchestrator

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  # Load repo root .env without overriding already-set vars
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
  # Augment from AELP2/config/.env.aelp2 for any missing AELP2_* vars
  if [[ -f AELP2/config/.env.aelp2 ]]; then
    while IFS='=' read -r key val; do
      [[ -z "$key" || "$key" =~ ^# ]] && continue
      if [[ "$key" == AELP2_* && -z "${!key:-}" ]]; then export "$key=$val"; fi
    done < AELP2/config/.env.aelp2
  fi
elif [[ -f AELP2/config/.env.aelp2 ]]; then
  # Load AELP2 defaults without overriding already-set vars
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < AELP2/config/.env.aelp2
else
  echo "No env file found. Create .env at repo root or AELP2/config/.env.aelp2"
  exit 2
fi

echo "Validating environment..."
python3 - << 'PY'
import os, sys

required = [
  'GOOGLE_CLOUD_PROJECT',
  'BIGQUERY_TRAINING_DATASET',
  'AELP2_MIN_WIN_RATE',
  'AELP2_MAX_CAC',
  'AELP2_MIN_ROAS',
  'AELP2_MAX_SPEND_VELOCITY',
  'AELP2_APPROVAL_TIMEOUT',
  'AELP2_SIM_BUDGET',
  'AELP2_SIM_STEPS',
  'AELP2_EPISODES',
]
missing = [v for v in required if not os.getenv(v)]
if missing:
  print(f"Missing required env vars: {', '.join(missing)}")
  sys.exit(3)

# Calibration gating vars (optional but recommended)
if os.getenv('AELP2_CALIBRATION_REF_JSON'):
  for v in ['AELP2_CALIBRATION_MAX_KS', 'AELP2_CALIBRATION_MAX_HIST_MSE']:
    if not os.getenv(v):
      print(f"ERROR: {v} required when AELP2_CALIBRATION_REF_JSON is set")
      sys.exit(4)
print("Environment OK")
PY

echo "Starting AELP2 orchestrator..."
python3 -m AELP2.core.orchestration.production_orchestrator --episodes "${AELP2_EPISODES}" --steps "${AELP2_SIM_STEPS}" ${AELP2_LOG_ARGS:-}
