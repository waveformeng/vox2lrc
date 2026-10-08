#!/bin/bash
# Deploy the vox2lrc API to a Droplet prepared by droplet-init.sh. Safe to re-run.
#
#   deploy/deploy.sh <droplet-ip> <domain>
#   SSH_KEY=~/.ssh/other_key deploy/deploy.sh ...   # to pick a specific key
#
# <domain> must already have an A record pointing at the Droplet; Caddy gets a
# Let's Encrypt certificate for it on first start. On the first deploy an API
# token is generated on the server and printed once: store it in your backend's
# secrets as the Bearer token.
set -euo pipefail

HOST=${1:?usage: deploy.sh <droplet-ip> <domain>}
DOMAIN=${2:?usage: deploy.sh <droplet-ip> <domain>}
APP=/opt/worker/vox2lrc
SSH=(ssh -o BatchMode=yes ${SSH_KEY:+-i "$SSH_KEY" -o IdentitiesOnly=yes})
cd "$(dirname "$0")/.."

rsync -az --delete -e "${SSH[*]}" \
  --include='/src/***' --include='/deploy/***' \
  --include='/.env.example' --include='/pyproject.toml' --include='/uv.lock' --include='/README.md' --include='/LICENSE' \
  --exclude='*' \
  ./ "root@$HOST:$APP/"

"${SSH[@]}" "root@$HOST" DOMAIN="$DOMAIN" APP="$APP" bash -s <<'REMOTE'
set -euo pipefail

# One-time: Caddy for HTTPS, and open 80 (ACME + redirect) and 443.
if ! command -v caddy >/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y caddy
fi
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null

# Code and venv, owned by the unprivileged service user.
mkdir -p /opt/worker/.cache
chown -R worker:worker "$APP" /opt/worker/.cache
sudo -u worker -H bash -c "cd $APP && UV_PYTHON_DOWNLOADS=never uv sync --frozen --no-dev --extra server"

# Secrets: generated here on first deploy, never sent over the wire from the laptop.
if [ ! -f /etc/vox2lrc.env ]; then
  TOKEN=$(openssl rand -hex 32)
  sed "s/^VOX2LRC_API_TOKEN=.*/VOX2LRC_API_TOKEN=$TOKEN/" "$APP/.env.example" > /etc/vox2lrc.env
  chown root:worker /etc/vox2lrc.env
  chmod 0640 /etc/vox2lrc.env
  echo "=== New API token (shown once; store it in your backend secrets) ==="
  echo "$TOKEN"
  echo "==================================================================="
fi

install -m 0644 "$APP/deploy/vox2lrc.service" /etc/systemd/system/vox2lrc.service
systemctl daemon-reload
systemctl enable --quiet vox2lrc
systemctl restart vox2lrc

sed "s/VOX2LRC_DOMAIN/$DOMAIN/" "$APP/deploy/Caddyfile" > /etc/caddy/Caddyfile
systemctl reload-or-restart caddy

# First start downloads the model (~17 MB) before the API answers.
for _ in $(seq 60); do
  if curl -sf http://127.0.0.1:8000/healthz; then echo; echo "vox2lrc is up"; exit 0; fi
  sleep 2
done
echo "vox2lrc did not become healthy; recent logs:" >&2
journalctl -u vox2lrc -n 40 --no-pager >&2
exit 1
REMOTE

echo "Test: curl -H \"Authorization: Bearer \$TOKEN\" -F file=@vocals.mp3 -F language=en https://$DOMAIN/v1/transcribe"
