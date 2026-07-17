# GrowiHF Bot — Quick Start (5 Minutes)

## What You Have

✅ **Reactive signal bot** — monitors OKX USDT perps for volume+price spikes  
✅ **Position sizing** — Kelly-fraction sizing for $280–$300 accounts  
✅ **Daemon setup** — runs 24/7 with auto-restart  
✅ **Telegram alerts** — real-time spike notifications  

## Setup (Windows)

### 1. Install Python (if not already)

Download Python 3.10+ from https://www.python.org/downloads/

During install: ✓ Check "Add Python to PATH"

### 2. Run Installer

```bash
cd "C:\Users\jason\Desktop\NeoBDM _ Broker Stalker_files\crypto-perp-screener"
deploy\install_windows.bat
```

This creates a virtual environment and installs dependencies.

### 3. Verify Config

Edit `config.json`:
- `telegram_bot_token`: Get from BotFather (@BotFather on Telegram)
- `telegram_chat_id`: Your Telegram user ID (easiest: message @userinfobot)
- `account_equity_usd`: Set to your actual account size (e.g., 280)

### 4. Test

Run the bot once to verify it works:

```bash
run_bot.bat
```

You should see:
```
2025-07-17 12:34:56 [INFO] GrowiHF Reactive Bot started. Poll=15s, Account=$280
2025-07-17 12:34:56 [INFO] Telegram message sent
```

Press Ctrl+C to stop.

### 5. Always-Run Setup

Choose ONE method:

**Option A: Task Scheduler (Simplest)**
- Follow `deploy\windows_scheduler.md`
- Bot runs 24/7 when you're logged in or not
- Auto-restarts if it crashes

**Option B: Docker (If installed)**
```bash
docker-compose up -d
```

## How It Works

1. **Poll** every 15 seconds: fetches all USDT perp prices from OKX
2. **Detect spike**: checks if price moved 3%+ in 60s OR 5%+ in 300s
3. **Confirm volume**: verifies 1m volume is 3x+ the 10m average
4. **Send alert**: telegram message with pair, price change %, position size

Example alert:
```
🚀 UP spike: BTC-USDT
+4.23% over 60s
Last: 65,432
1m volume: 4.2x avg
Position size: $15.23
```

## Tune Signals

**Too many false alerts?** Increase thresholds in `config.json`:
```json
"price_thresholds": {
  "60": 4.0,      // ← was 3.0, now 4%
  "300": 6.0      // ← was 5.0, now 6%
},
"volume_spike_multiplier": 4.0  // ← was 3.0, now 4x
```

**No signals at all?** Decrease thresholds:
```json
"price_thresholds": {
  "60": 2.0,      // ← was 3.0, now 2%
  "300": 4.0      // ← was 5.0, now 4%
},
"volume_spike_multiplier": 2.5  // ← was 3.0, now 2.5x
```

## Monitor

**View live logs:**
```bash
tail -f growhf_bot.log
```

**Check if bot is running:**
```bash
# PowerShell
Get-Process python | Where-Object {$_.CommandLine -like "*growhf*"}
```

## What's Next

**Execution (roadmap):**
- Add OKX API integration for actual grid orders
- Hyperliquid order placement
- Stop-loss + take-profit logic
- P&L tracking

**Currently:** Signals + position sizing only (no execution yet).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Bot stops after 1 hour | Check `growhf_bot.log` for errors |
| No Telegram alerts | Verify bot token & chat ID are correct |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in venv |
| Task Scheduler won't run | See `deploy/windows_scheduler.md` Step 4 |

## Support

- Check `growhf_bot.log` for detailed errors
- Read `README.md` for full documentation
- GitHub Issues for bug reports

---

**Status**: Ready to run. Execution pending OKX API credentials.
