#!/usr/bin/env bash
#
# install_okx_bot.sh — one-shot droplet installer for the OKX Telegram bot.
#
# Creates a dedicated system user, lays the repo down under /opt/okx-bot,
# builds a venv, installs the systemd unit, and (optionally) sets up an
# SSH deploy key so the GitHub Actions workflow can push updates.
#
# Usage (run as root on a fresh Ubuntu/Debian droplet):
#     curl -fsSL https://raw.githubusercontent.com/0xfridayy/growhf-reactive-bot/main/install_okx_bot.sh | bash
#   or:
#     sudo ./install_okx_bot.sh
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Settings (override via environment before running)
# --------------------------------------------------------------------------- #
APP_USER="${APP_USER:-okxbot}"
APP_DIR="${APP_DIR:-/opt/okx-bot}"
REPO_URL="${REPO_URL:-https://github.com/0xfridayy/growhf-reactive-bot.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
# Two services, one bot: the Telegram bot (commands + OI/funding flips) and the
# reactive spike screener. Both share this venv + config.json; only the Telegram
# bot polls getUpdates, so there's no getUpdates conflict on the shared token.
SERVICES=("okx-bot" "okx-spike")
SERVICE_UNITS=("okx-bot.service" "okx-spike.service")

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root (sudo ./install_okx_bot.sh)."

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
# 3. Deploy key (so CI can push updates over SSH)
# --------------------------------------------------------------------------- #
SSH_DIR="/home/${APP_USER}/.ssh"
DEPLOY_KEY="${SSH_DIR}/deploy_key"
if [[ ! -f "$DEPLOY_KEY" ]]; then
    log "Generating SSH deploy key..."
    install -d -m 700 -o "$APP_USER" -g "$APP_USER" "$SSH_DIR"
    sudo -u "$APP_USER" ssh-keygen -t ed25519 -N "" -C "okx-bot-deploy@$(hostname)" -f "$DEPLOY_KEY"
    cat > "${SSH_DIR}/config" <<EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ${DEPLOY_KEY}
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
EOF
    chown "$APP_USER:$APP_USER" "${SSH_DIR}/config"
    chmod 600 "${SSH_DIR}/config"
    warn "Add this PUBLIC deploy key to the repo (Settings → Deploy keys, allow write):"
    echo "-----------------------------------------------------------------------"
    cat "${DEPLOY_KEY}.pub"
    echo "-----------------------------------------------------------------------"
else
    log "Deploy key already present at ${DEPLOY_KEY}."
fi

# --------------------------------------------------------------------------- #
# 4. Clone or update the repo
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
# 5. Python venv + deps
# --------------------------------------------------------------------------- #
log "Building virtualenv..."
sudo -u "$APP_USER" python3 -m venv "${APP_DIR}/venv"
sudo -u "$APP_USER" "${APP_DIR}/venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

# --------------------------------------------------------------------------- #
# 6. Config
# --------------------------------------------------------------------------- #
if [[ ! -f "${APP_DIR}/config.json" ]]; then
    warn "No config.json found — copy config.json and fill in telegram_bot_token / telegram_chat_id."
else
    if grep -q "PUT_YOUR" "${APP_DIR}/config.json"; then
        warn "config.json still has placeholder values — edit it before the bot will start."
    fi
fi

# --------------------------------------------------------------------------- #
# 7. systemd units (Telegram bot + spike screener)
# --------------------------------------------------------------------------- #
log "Installing systemd units..."
for unit in "${SERVICE_UNITS[@]}"; do
    install -m 644 "${APP_DIR}/${unit}" "/etc/systemd/system/${unit}"
done
systemctl daemon-reload
for svc in "${SERVICES[@]}"; do
    systemctl enable "$svc"
done

if grep -q "PUT_YOUR" "${APP_DIR}/config.json" 2>/dev/null; then
    warn "Not starting services yet — fill in config.json, then:"
    for svc in "${SERVICES[@]}"; do
        echo "    systemctl start ${svc}"
    done
else
    for svc in "${SERVICES[@]}"; do
        log "Starting ${svc}..."
        systemctl restart "$svc"
    done
fi

log "Done. Useful commands:"
for svc in "${SERVICES[@]}"; do
    echo "    systemctl status ${svc}"
    echo "    journalctl -u ${svc} -f"
done
