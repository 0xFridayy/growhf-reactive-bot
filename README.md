# GrowiHF Reactive Signal Bot

Reactive volume + price spike detector for OKX & Hyperliquid perpetuals. Executes small grid positions on signals. Optimized for small accounts ($300–$1000).

## Features

- **Reactive signals**: Detects price spikes (3–5%) + volume surge (3x) confirmation
- **Multi-exchange**: OKX + Hyperliquid monitoring & execution
- **Risk-managed**: Kelly-fraction position sizing, max 2% risk per trade
- **Small-account ready**: Configurable for $280–$300 starting capital
- **Daemon mode**: Runs 24/7 with auto-restart on crash
- **Telegram alerts**: Real-time spike notifications + execution confirmations
- **Macro gauges**: `/analyze BTC.D` (a.k.a. BTCDOM), `USDT.D`, `DXY` — none is an OKX
  perp, so they're derived (dominance from CoinGecko supplies × live OKX prices; DXY
  from Yahoo, ~10 min delayed) and returned as candles the normal TPO/regime analytics
  consume. Regime context only — never a sized plan. See `macro_feeds.py`.
- **Market bias**: `/bias` returns a scored TOTAL3 / BTC.D / USDT.D + macro read on
  demand, and the same read auto-posts hourly 19:00–00:00 local. See
  [Market bias](#market-bias-bias) below.

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

## Market bias (`/bias`)

`bias_telegram.py` scores overall crypto risk appetite from **TOTAL3**, **BTC
dominance**, **USDT dominance** and a macro cross-asset panel (DXY / VIX / ES /
US10Y) plus CoinDesk + Cointelegraph headlines. Two ways to get it:

| | |
|---|---|
| **On demand** | text `/bias` to the bot (handled by `okx_tele_bot.py`) |
| **Automatic** | task `Crypto-Bias-Telegram`, hourly 19:00–00:00 local |

Both call `bias_telegram.render_bias()` — one source of truth, so the on-demand
and scheduled messages can never drift apart. Stdlib only, no extra deps.

**Scoring.** Six components each vote in `[-2, +2]`, positive = risk-on for
alts: TOTAL3 momentum, TOTAL3 position in its 72h range, BTC.D trend, USDT.D
trend, alts-vs-BTC, BTC trend. The mean maps to `-100..+100` and a label
(BULLISH / LEAN BULLISH / NEUTRAL / LEAN BEARISH / BEARISH).

The message also states the **divergence read** explicitly, because the
composite score alone can mislead: when BTC.D is rising while alts lag, "risk-on"
is concentrating in BTC and TOTAL3 strength is borrowed. That case prints a
warning rather than letting a high headline score imply alts are bid.

**Method caveat — read this before trusting the deltas.** CoinGecko's free tier
returns only a *current* dominance snapshot, no history. The 4h/24h/72h changes
in BTC.D / USDT.D / TOTAL3 are **reconstructed**: anchored on the live snapshot
and propagated backwards along hourly OKX price paths with supply held fixed
(over a 3–5 day window dominance moves are price-driven, not supply-driven).
Accurate for that window, but it is a model, not measured tape. Every run
appends a real snapshot to `bias_history.csv` (gitignored) so genuine measured
history accumulates for cross-checking.

`TOTAL3` follows the TradingView definition (`TOTAL - BTC - ETH`, which includes
stablecoins); `TOTAL3X` excludes them and is what the scoring actually uses,
since that is the real alt risk-appetite number.

Setup and scheduling live in [`deploy/okx_windows_scheduler.md`](deploy/okx_windows_scheduler.md).
Run `python set_tg_commands.py` once to register the command list with Telegram
so typing `/` autocompletes.

## License

MIT

## Support

Issues & feedback: GitHub Issues or telegram.
