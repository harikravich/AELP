#!/usr/bin/env bash
set -euo pipefail

check() {
  local name="$1"
  local pidf="/tmp/aelp2_${name}.pid"
  local log="/tmp/aelp2_${name}.log"
  if [[ -f "$pidf" ]]; then
    local pid; pid=$(cat "$pidf" 2>/dev/null || true)
    if [[ -n "${pid}" && -d "/proc/${pid}" ]]; then
      echo "[$name] RUNNING (pid=$pid)"
    else
      echo "[$name] FINISHED (pid=${pid:-?})"
    fi
  else
    echo "[$name] NOT STARTED"
  fi
  if [[ -f "$log" ]]; then
    echo "--- tail $log"; tail -n 10 "$log" || true; echo
  fi
}

for j in ga4_ingest ga4_attr views mmm bandit_service budget_orch training_stub; do
  check "$j"
done

