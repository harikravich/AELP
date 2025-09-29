#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Load env without overriding pre-set vars
if [[ -f .env ]]; then
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

START=${1:-$(date -u -d '14 days ago' +%F)}
END=${2:-$(date -u +%F)}

echo "Running fidelity evaluation for $START..$END"
python3 -m AELP2.pipelines.fidelity_evaluation --start "$START" --end "$END"
