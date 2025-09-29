#!/usr/bin/env bash
set -euo pipefail

# Sets up Nginx reverse proxy on port 80 to the local Next.js server on 127.0.0.1:3001

if ! command -v sudo >/dev/null 2>&1; then echo "sudo required" >&2; exit 1; fi

echo "Installing Nginx…"
sudo apt-get update -y >/dev/null 2>&1 || true
sudo apt-get install -y nginx >/dev/null 2>&1

cat <<'CONF' | sudo tee /etc/nginx/sites-available/aelp2-dashboard >/dev/null
server {
  listen 80 default_server;
  listen [::]:80 default_server;
  server_name _;

  location / {
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_pass http://127.0.0.1:3001;
  }
}
CONF

sudo rm -f /etc/nginx/sites-enabled/default || true
sudo ln -sf /etc/nginx/sites-available/aelp2-dashboard /etc/nginx/sites-enabled/aelp2-dashboard
sudo nginx -t
sudo systemctl restart nginx
echo "Nginx is proxying http://<VM_EXTERNAL_IP>/ to 127.0.0.1:3001"

