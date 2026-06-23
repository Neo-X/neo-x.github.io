#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found."
  echo "Create it with:"
  echo "  echo 'FTP_PASSWORD=yourpassword' > .env"
  exit 1
fi

source "$ENV_FILE"

FTP_HOST="ftp.fracturedplane.com"
FTP_USER="${FTP_USER:-robot308}"
FTP_PASS="${FTP_PASSWORD:?FTP_PASSWORD not set in .env}"
REMOTE_DIR="/public_html"
LOCAL_DIR="$SCRIPT_DIR/_site"

echo "==> Building site..."
cd "$SCRIPT_DIR"
export PATH="$HOME/.local/share/gem/ruby/3.2.0/bin:$PATH"
bundle exec jekyll build

echo "==> Uploading to $FTP_HOST$REMOTE_DIR ..."
lftp -u "${FTP_USER},${FTP_PASS}" "ftp://${FTP_HOST}" <<LFTP_CMDS
set ftp:ssl-allow yes
set ssl:verify-certificate no
set ftp:passive-mode yes
mirror --reverse --delete --verbose ${LOCAL_DIR}/ ${REMOTE_DIR}/
bye
LFTP_CMDS

echo "==> Done. Visit https://www.fracturedplane.com"
