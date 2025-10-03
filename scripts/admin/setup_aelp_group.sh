#!/usr/bin/env bash
set -euo pipefail

# Idempotently create shared group access to Hari's AELP tree
# Usage: sudo ./scripts/admin/setup_aelp_group.sh bill isotta

GROUP=aelp
ROOT=/home/hariravichandran/AELP

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0 <user1> [<user2> ...]" >&2
  exit 1
fi

echo "Ensuring group: $GROUP"
getent group "$GROUP" >/dev/null || groupadd "$GROUP"

for u in "$@"; do
  echo "Adding user $u to $GROUP"
  id "$u" >/dev/null 2>&1 || useradd -m -s /bin/bash "$u"
  usermod -aG "$GROUP" "$u"
done

echo "Granting group access to $ROOT"
chgrp -R "$GROUP" "$ROOT"

# Allow group traversal on Hari's home and AELP tree
chmod 0750 /home/hariravichandran || true

find "$ROOT" -type d -exec chmod 2750 {} +
find "$ROOT" -type f -exec chmod 0640 {} +

# Optional: set default ACL so new files remain group-readable
if command -v setfacl >/dev/null 2>&1; then
  setfacl -R -m g:$GROUP:rx "$ROOT" || true
  setfacl -R -d -m g:$GROUP:rx "$ROOT" || true
fi

echo "Done. Members of '$GROUP' can now read $ROOT."

