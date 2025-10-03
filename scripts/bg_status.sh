#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
shopt -s nullglob
for f in .runs/*.pid; do
  [[ -f $f ]] || continue
  pid=$(cat "$f" || true)
  if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
    echo "RUNNING  $f  pid=$pid"
  else
    echo "STOPPED  $f  pid=${pid:-?}"
  fi
done
