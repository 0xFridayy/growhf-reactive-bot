#!/usr/bin/env bash
#
# install_alpha_ml_vps.sh — one-shot installer for a VPS dedicated to the
# alpha_ml research pipeline (as opposed to install_okx_bot.sh, which deploys
# the live Telegram bot). Today this means one thing: okx_orderflow_logger.py
# running as a persistent systemd service.
#
# Why this can't just be another GitHub Actions job: the daily search
# (alpha-ml-search.yml) and weekly DDQN retrain (alpha-ml-train.yml) are
# batch jobs and Actions already runs those for free. The order-flow tape
# logger is NOT a batch job — it holds a live OKX WebSocket connection open
# and appends to a local SQLite DB as trades happen. OKX's public REST only
# exposes the last ~500 trades with no historical backfill, so every hour
# this isn't running is tape that can never be recovered. That's the one
# thing that needs a persistent host, which is what this VPS is for.
#
# orderflow.db is NOT committed back to git by this script (or by anything
# else) — it grows unbounded (millions of rows/week) and does not belong in
# a git history. It stays on this VPS's disk; pull it off with scp/rsync
# when you actually want to run a backtest against it. Size the boot disk
# accordingly (the default GCE 10GB disk will fill up in a matter of weeks
# at the observed ~600k rows/day across trades+book_snapshots — 30-50GB is a
# safer starting point).
#
# Usage (run as root on a fresh Ubuntu/Debian VM):
#     curl -fsSL https://raw.githubusercontent.com/0xfridayy/growhf-reactive-bot/main/install_alpha_ml_vps.sh | bash
#   or:
#     sudo ./install_alpha_ml_vps.sh
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Settings (override via environment before running)
# --------------------------------------------------------------------------- #
APP_USER="${APP_USER:-alphaml}"
APP_DIR="${APP_DIR:-/opt/alpha-ml}"
REPO_URL="${REPO_URL:-https://github.com/0xfridayy/growhf-reactive-bot.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
SERVICES=("okx-orderflow-logger")
SERVICE_UNITS=("okx-orderflow-logger.service")

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root (sudo ./install_alpha_ml_vps.sh)."

# --------------------------------------------------------------------------- #
# 1. System packages
# --------------------------------------------------------------------------- #
log "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-pip git

# --------------------------------------------------------------------------- #
# 2. Dedicated service user
# --------------------------------------------------------------------------- #
if id "$APP_USER" &>/dev/null; then
    log "User '$APP_USER' already exists."
else
    log "Creating system user '$APP_USER'..."
    useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

# --------------------------------------------------------------------------- #
# 3. Clone or update the repo
# --------------------------------------------------------------------------- #
if [[ -d "${APP_DIR}/.git" ]]; then
    log "Updating existing checkout in ${APP_DIR}..."
    sudo -u "$APP_USER" git -C "$APP_DIR" fetch --all --prune
    sudo -u "$APP_USER" git -C "$APP_DIR" checkout "$REPO_BRANCH"
    sudo -u "$APP_USER" git -C "$APP_DIR" reset --hard "origin/${REPO_BRANCH}"
else
    log "Cloning ${REPO_URL} into ${APP_DIR}..."
    install -d -o "$APP_USER" -g "$APP_USER" "$APP_DIR"
    sudo -u "$APP_USER" git clone --branch "$REPO_BRANCH" "$REPO_URL" "$APP_DIR"
fi

# --------------------------------------------------------------------------- #
# 4. Python venv + deps
# --------------------------------------------------------------------------- #
# Only the root requirements.txt — the orderflow logger needs requests +
# websocket-client + stdlib sqlite3, nothing from alpha_ml/requirements.txt
# (numpy/pandas/xgboost/torch). Keep this VM lean; the heavy training already
# runs on GitHub Actions' runners for free.
log "Building virtualenv..."
sudo -u "$APP_USER" python3 -m venv "${APP_DIR}/venv"
sudo -u "$APP_USER" "${APP_DIR}/venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

# --------------------------------------------------------------------------- #
# 5. systemd unit
# --------------------------------------------------------------------------- #
log "Installing systemd units..."
for unit in "${SERVICE_UNITS[@]}"; do
    install -m 644 "${APP_DIR}/${unit}" "/etc/systemd/system/${unit}"
done
systemctl daemon-reload
for svc in "${SERVICES[@]}"; do
    systemctl enable "$svc"
    log "Starting ${svc}..."
    systemctl restart "$svc"
done

log "Done. Useful commands:"
for svc in "${SERVICES[@]}"; do
    echo "    systemctl status ${svc}"
    echo "    journalctl -u ${svc} -f"
done
echo "    sqlite3 ${APP_DIR}/orderflow.db 'select count(*) from trades;'"
