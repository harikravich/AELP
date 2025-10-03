#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/bg_stop.sh .runs/<name>.pid" >&2
  exit 1
fi
pidf="$1"
pid=$(cat "$pidf" || true)
if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
  kill "$pid" && echo "Stopped pid=$pid from $pidf"
else
  echo "No live process for pidfile: $pidf"
fi

