#!/usr/bin/env bash
set -euo pipefail

ROOT_CANDIDATES=(
  /home/harikravich_gmail_com/AELP
  /home/hariravichandran/AELP
  "$PWD"
)
for r in "${ROOT_CANDIDATES[@]}"; do
  if [ -d "$r" ] && [ -f "$r/README.md" -o -d "$r/pipelines" ]; then
    ROOT="$r"; break
  fi
done

echo "User: $(whoami)"
echo "Groups: $(id -nG)"
echo "Python: $(python3 -V)"
echo "Repo: $ROOT"
ls -ld "$ROOT" || true
ls -l "$ROOT/README.md" 2>/dev/null || true

echo "Env sample (masked):"
for k in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY META_ACCOUNT_ID META_API_VERSION GOOGLE_CLOUD_PROJECT; do
  v=${!k:-}
  if [[ -n "$v" ]]; then
    echo "  $k=SET (len=${#v})"
  else
    echo "  $k=MISSING"
  fi
done

echo "PYTHONPATH check:"
PYTHONPATH="$ROOT" python3 - <<'PY'
import sys, importlib
print('sys.path[0:3]=', sys.path[0:3])
import importlib.util
spec = importlib.util.find_spec('pipelines')
print('pipelines importable:', bool(spec))
PY

echo "OK"
