# GrowiHF Reactive Signal Bot

Reactive volume + price spike detector for OKX & Hyperliquid perpetuals. Executes small grid positions on signals. Optimized for small accounts ($300–$1000).

## Features

- **Reactive signals**: Detects price spikes (3–5%) + volume surge (3x) confirmation
- **Multi-exchange**: OKX + Hyperliquid monitoring & execution
- **Risk-managed**: Kelly-fraction position sizing, max 2% risk per trade
- **Small-account ready**: Configurable for $280–$300 starting capital
- **Daemon mode**: Runs 24/7 with auto-restart on crash
- **Telegram alerts**: Real-time spike notifications + execution confirmations

## Quick Start

### 1. Setup

```bash
# Clone & install
git clone <repo>
cd crypto-perp-screener
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Edit `config.json`:

```json
{
  "telegram_bot_token": "YOUR_BOT_TOKEN",
  "telegram_chat_id": "YOUR_CHAT_ID",
  "quote_filter": "USDT",
  "poll_interval_seconds": 15,
  "price_thresholds": {
    "60": 3.0,
    "300": 5.0
  },
  "volume_spike_multiplier": 3.0,
  "sizing": {
    "account_equity_usd": 280.0,
    "kelly_fraction": 0.25,
    "max_risk_per_trade": 0.02,
    "max_leverage": 3.0
  }
}
```

**Config parameters:**
- `price_thresholds`: window (seconds) → threshold (%). E.g., 3% spike in 60s
- `volume_spike_multiplier`: current 1m vol / avg 10m vol
- `sizing.kelly_fraction`: position sizing aggressiveness (0.1–0.5 recommended for $300 accounts)
- `sizing.max_risk_per_trade`: max risk as % of equity

### 3. Run

**One-shot test:**
```bash
python growhf_reactive_bot.py
```

**Daemon (Linux/macOS with systemd):**
```bash
sudo cp deploy/growhf.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start growhf
sudo systemctl enable growhf  # auto-start on reboot
```

**Daemon (Docker):**
```bash
docker build -t growhf .
docker run -d --restart unless-stopped -v $(pwd)/config.json:/app/config.json growhf
```

**Windows (Task Scheduler):**
See `deploy/windows_scheduler.md`

## Signals

Bot triggers when both conditions met:

1. **Price spike**: `abs(price_change) > threshold` over window
2. **Volume confirmation**: `current_1m_vol / avg_10m_vol > multiplier`

Example: 4% price move in 60s + 4x volume spike = signal.

## Position Sizing (Small Accounts)

For $280 account with 2% max risk:

- Signal strength = f(vol_ratio, pct_change)
- Position = kelly_frac × base_risk × (1 + signal_strength)
- Range: $5–$20 notional per trade (with 3x leverage = $15–$60 on exchange)

## Logging

All activity logged to:
- `growhf_bot.log` (file)
- stdout (console)

```bash
tail -f growhf_bot.log
```

## Deployment

### Linux/macOS

```bash
# Systemd setup
./deploy/install_systemd.sh

# View logs
journalctl -u growhf -f

# Stop/restart
sudo systemctl stop growhf
sudo systemctl restart growhf
```

### Docker

```bash
docker-compose up -d
docker logs -f growhf-bot
```

### AWS/VPS

```bash
./deploy/install_vps.sh <VPS_IP> <SSH_KEY>
```

## API Keys (Future)

For execution, you'll need:
- **OKX**: API key + secret (for /v5/trade endpoints)
- **Hyperliquid**: API key (for order placement)

Store in `.env`:
```
OKX_API_KEY=your_key
OKX_API_SECRET=your_secret
OKX_PASSPHRASE=your_passphrase
HYPERLIQUID_API_KEY=your_key
```

Currently, bot runs **signals-only mode** (monitoring, no execution). Execution code in progress.

## Troubleshooting

**Bot stops after N hours:**
- Check `growhf_bot.log` for errors
- Verify Telegram token is valid
- Check OKX API rate limits (public endpoints: 100 req/min)

**No signals detected:**
- Increase `price_thresholds` (e.g., 6% instead of 5%)
- Decrease `volume_spike_multiplier` (e.g., 2.5x instead of 3x)
- Verify `quote_filter` matches your market

**High CPU usage:**
- Reduce `poll_interval_seconds` backoff
- Check if system is overloaded

## License

MIT

## Support

Issues & feedback: GitHub Issues or telegram.
