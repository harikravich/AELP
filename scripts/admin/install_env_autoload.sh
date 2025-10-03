#!/usr/bin/env bash
set -euo pipefail

# Autoload .env for members of 'aelp' group at login.
# Usage: sudo ./scripts/admin/install_env_autoload.sh [/srv/aelp/.env.local]

ENV_FILE=${1:-/srv/aelp/.env.local}
GROUP=aelp
TARGET=/etc/profile.d/aelp_env.sh

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0 [env_file]" >&2
  exit 1
fi

cat > "$TARGET" <<EOF
# Autoload GAELP env for aelp group users
if id -nG "$(whoami)" 2>/dev/null | grep -qw $GROUP; then
  if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
  fi
fi
EOF

chmod 0644 "$TARGET"
echo "Installed login autoload at $TARGET (reads $ENV_FILE for group $GROUP)."

