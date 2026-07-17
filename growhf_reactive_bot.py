"""
GrowiHF Reactive Signal Bot
Monitors OKX & Hyperliquid for volume+price spikes, executes small grid positions.
Small-account optimized ($300 sizing). Runs as non-stop daemon.

Run:
    python growhf_reactive_bot.py

Config lives in config.json next to this file.
"""

import json
import time
import urllib.error
import urllib.request
import logging
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime, timedelta
import sys

CONFIG_PATH = Path(__file__).with_name("config.json")
OKX_BASE = "https://www.okx.com"
HYPERLIQUID_API = "https://api.hyperliquid.exchange"
USER_AGENT = "growhf-bot/1.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("growhf_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("telegram_bot_token", "").startswith("PUT_YOUR"):
        raise SystemExit(f"Edit {CONFIG_PATH} first")
    return cfg


def http_get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"HTTP GET failed {url}: {e}")
        return None


def http_post_json(url, payload, timeout=10):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"HTTP POST failed {url}: {e}")
        return None


def fetch_okx_tickers(quote_filter="USDT"):
    data = http_get_json(f"{OKX_BASE}/api/v5/market/tickers?instType=SWAP")
    if not data:
        return []
    rows = data.get("data", [])
    if quote_filter:
        rows = [r for r in rows if r["instId"].split("-")[1:2] == [quote_filter]]
    return rows


def fetch_okx_candles(inst_id, bar="1m", limit=20):
    url = f"{OKX_BASE}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    data = http_get_json(url)
    return data.get("data", []) if data else []


def fetch_hyperliquid_tickers():
    """Fetch all Hyperliquid perpetual tickers."""
    payload = {"type": "welltimeMetrics"}
    data = http_post_json(f"{HYPERLIQUID_API}/info", payload)
    if not data:
        return []
    return data.get("data", []) if isinstance(data, dict) else []


def fetch_hyperliquid_candles(coin, interval="1m", limit=20):
    """Fetch Hyperliquid candles."""
    payload = {
        "type": "candles",
        "coin": coin,
        "interval": interval,
        "dir": 1,
        "startTime": int((time.time() - limit * 60) * 1000)
    }
    data = http_post_json(f"{HYPERLIQUID_API}/info", payload)
    return data if isinstance(data, list) else []


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


def calculate_position_size(equity_usd, vol_ratio, pct_change, cfg_sizing):
    """Kelly-weighted position sizing for small accounts."""
    kelly_frac = cfg_sizing.get("kelly_fraction", 0.25)
    max_risk = cfg_sizing.get("max_risk_per_trade", 0.02)
    max_leverage = cfg_sizing.get("max_leverage", 3.0)
    min_notional = cfg_sizing.get("min_notional_usd", 5.0)

    base_risk = min(max_risk * equity_usd, equity_usd * 0.01)

    signal_strength = min(vol_ratio / 3.0, 1.0) * min(abs(pct_change) / 5.0, 1.0)
    position_notional = base_risk * kelly_frac * (1.0 + signal_strength) * equity_usd
    position_notional = max(position_notional, min_notional)
    position_notional = min(position_notional, equity_usd * max_leverage)

    return position_notional


def process_okx_spike(row, cfg, signal_history):
    """Evaluate OKX spike for execution."""
    inst_id = row["instId"]
    try:
        last_price = float(row["last"])
    except (KeyError, ValueError):
        return None

    price_thresholds = {int(k): float(v) for k, v in cfg.get("price_thresholds", {}).items()}
    vol_multiplier = cfg.get("volume_spike_multiplier", 3.0)
    cooldown = cfg.get("cooldown_seconds", 900)

    now = time.time()

    if inst_id in signal_history and now - signal_history[inst_id]["ts"] < cooldown:
        return None

    dq = signal_history[inst_id]
    dq.append((now, last_price))
    max_window = max(price_thresholds) if price_thresholds else 300
    while dq and now - dq[0][0] > max_window + 60:
        dq.popleft()

    for window, threshold in price_thresholds.items():
        past = None
        for ts, price in dq:
            if now - ts >= window:
                past = price
            else:
                break

        if past is None or past == 0:
            continue

        pct_change = (last_price - past) / past * 100
        if abs(pct_change) < threshold:
            continue

        candles = fetch_okx_candles(inst_id, bar="1m", limit=20)
        if len(candles) < 6:
            continue

        current_vol = float(candles[0][5])
        prior_vols = [float(c[5]) for c in candles[1:11]]
        avg_prior_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 0
        if avg_prior_vol <= 0:
            continue

        vol_ratio = current_vol / avg_prior_vol
        if vol_ratio < vol_multiplier:
            continue

        signal_history[inst_id]["ts"] = now

        sizing_cfg = cfg.get("sizing", {})
        equity = sizing_cfg.get("account_equity_usd", 300)
        notional = calculate_position_size(equity, vol_ratio, pct_change, sizing_cfg)

        direction = "🚀 UP" if pct_change > 0 else "🔴 DOWN"
        return {
            "exchange": "OKX",
            "pair": inst_id,
            "direction": pct_change > 0,
            "pct_change": pct_change,
            "vol_ratio": vol_ratio,
            "window": window,
            "price": last_price,
            "notional": notional,
            "signal_text": f"{direction} spike: <b>{inst_id}</b>\n{pct_change:+.2f}% over {window}s\nPrice: {last_price}\n1m vol: {vol_ratio:.1f}x\nPosition size: ${notional:.2f}"
        }

    return None


def main_loop():
    cfg = load_config()
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    poll_interval = cfg.get("poll_interval_seconds", 15)

    signal_history = defaultdict(lambda: {"ts": 0, "prices": deque(maxlen=100)})

    logger.info(f"GrowiHF Reactive Bot started. Poll={poll_interval}s, Account=${cfg['sizing'].get('account_equity_usd', 300)}")
    send_telegram(token, chat_id, "🤖 GrowiHF reactive bot started (OKX + Hyperliquid)")

    consecutive_errors = 0

    while True:
        try:
            loop_start = time.time()

            tickers = fetch_okx_tickers(cfg.get("quote_filter"))
            if not tickers:
                logger.warning("No OKX tickers fetched")
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            for row in tickers:
                try:
                    signal = process_okx_spike(row, cfg, signal_history)
                    if signal:
                        logger.info(f"SIGNAL: {signal['pair']} {signal['direction']} {signal['pct_change']:.2f}%")
                        send_telegram(token, chat_id, signal["signal_text"])
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error processing {row.get('instId')}: {e}")

            if consecutive_errors > 5:
                logger.error("Too many consecutive fetch errors, restarting loop")
                time.sleep(30)
                consecutive_errors = 0
                continue

            elapsed = time.time() - loop_start
            sleep_time = max(0.5, poll_interval - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Stopping bot")
            send_telegram(token, chat_id, "🔴 GrowiHF bot stopped")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main_loop()
