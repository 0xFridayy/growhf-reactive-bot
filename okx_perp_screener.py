"""
OKX perpetual-futures screener: alerts to Telegram on unusual price spikes
confirmed by a volume surge. Uses only OKX public market-data endpoints
(no API key/secret needed) and Python's standard library (no pip installs).

Run:
    python okx_perp_screener.py

Config lives in config.json next to this file.
"""

import json
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

from kelly_sizing import SizingConfig, size_position

CONFIG_PATH = Path(__file__).with_name("config.json")
OKX_BASE = "https://www.okx.com"
USER_AGENT = "okx-perp-screener/1.0"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("telegram_bot_token", "").startswith("PUT_YOUR"):
        raise SystemExit(
            f"Edit {CONFIG_PATH} first: fill in telegram_bot_token and telegram_chat_id."
        )
    return cfg


def http_get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_tickers(quote_filter):
    data = http_get_json(f"{OKX_BASE}/api/v5/market/tickers?instType=SWAP")
    rows = data.get("data", [])
    if quote_filter:
        rows = [r for r in rows if r["instId"].split("-")[1:2] == [quote_filter]]
    return rows


def fetch_recent_candles(inst_id, bar="1m", limit=20):
    url = f"{OKX_BASE}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    data = http_get_json(url)
    return data.get("data", [])  # newest first: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.URLError as e:
        print(f"[telegram] send failed: {e}")


def build_dip_trade_plan(candles, entry_price, scfg, sizer_cfg, equity, open_positions):
    """
    Turn a DOWN spike (mean-reversion long signal) into a sized trade plan.
    Stop is derived from the dip's own depth: just below the recent swing low,
    clamped to [min_stop_frac, max_stop_frac]. Returns (Position, stop_price) or None.
    """
    lookback = int(scfg.get("swing_lookback_candles", 10))
    lows = [float(c[3]) for c in candles[:lookback]]  # candles newest-first; low at idx 3
    if not lows:
        return None
    swing_low = min(lows)
    buffer = float(scfg.get("stop_buffer_frac", 0.001))
    min_sf = float(scfg.get("min_stop_frac", 0.004))
    max_sf = float(scfg.get("max_stop_frac", 0.05))

    stop_price = swing_low * (1.0 - buffer)
    stop_dist = (entry_price - stop_price) / entry_price
    if stop_dist <= 0:              # swing low above entry -> fall back to floor
        stop_dist = min_sf
    stop_dist = max(min_sf, min(max_sf, stop_dist))
    stop_price = entry_price * (1.0 - stop_dist)

    pos = size_position(
        equity_usd=equity,
        entry_price=entry_price,
        stop_price=stop_price,
        open_positions=open_positions,
        cfg=sizer_cfg,
    )
    if pos is None:
        return None
    return pos, stop_price


def format_trade_plan(pos, entry_price, stop_price):
    take_profit = entry_price + (entry_price - stop_price)  # 1R target (mean-revert bounce)
    stop_pct = (entry_price - stop_price) / entry_price * 100
    return (
        f"\n\U0001F4C8 <b>Dip-buy plan (quarter-Kelly)</b>\n"
        f"Entry ~{entry_price:.6g} | Stop {stop_price:.6g} (-{stop_pct:.2f}%) | TP {take_profit:.6g} (+1R)\n"
        f"Leverage {pos.leverage}x | Notional ${pos.notional_usd} | Margin ${pos.margin_usd}\n"
        f"Size {pos.contracts} | Risk ${pos.risk_usd} [{pos.reason}]"
    )


def price_n_seconds_ago(history, now, window):
    past = None
    for ts, price in history:
        if now - ts >= window:
            past = price
        else:
            break
    return past


def main():
    cfg = load_config()
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    quote_filter = cfg.get("quote_filter") or None
    poll_interval = cfg.get("poll_interval_seconds", 15)
    price_thresholds = {int(k): float(v) for k, v in cfg.get("price_thresholds", {}).items()}
    max_window = max(price_thresholds) if price_thresholds else 300
    vol_multiplier = cfg.get("volume_spike_multiplier", 3.0)
    cooldown = cfg.get("cooldown_seconds", 900)

    scfg = cfg.get("sizing", {})
    sizing_enabled = scfg.get("enabled", False)
    equity = float(scfg.get("account_equity_usd", 0) or 0)
    sizer_cfg = SizingConfig(
        kelly_fraction=float(scfg.get("kelly_fraction", 0.25)),
        max_risk_per_trade=float(scfg.get("max_risk_per_trade", 0.02)),
        max_leverage=float(scfg.get("max_leverage", 3.0)),
        exchange_max_leverage=float(scfg.get("exchange_max_leverage", 50.0)),
        min_notional_usd=float(scfg.get("min_notional_usd", 5.0)),
        max_concurrent=int(scfg.get("max_concurrent", 3)),
    )

    history = defaultdict(deque)  # instId -> deque[(ts, last_price)]
    last_alert = {}  # instId -> ts of last alert
    active_plans = {}  # instId -> expiry ts (concurrency proxy for open positions)

    print(
        f"Starting OKX perp screener. poll={poll_interval}s "
        f"quote_filter={quote_filter or 'ALL'} thresholds={price_thresholds} "
        f"vol_multiplier={vol_multiplier}x cooldown={cooldown}s"
    )
    send_telegram(token, chat_id, "\U0001F7E2 OKX perp screener started.")

    try:
        while True:
            loop_start = time.time()
            try:
                tickers = fetch_tickers(quote_filter)
            except Exception as e:
                print(f"[poll] fetch_tickers failed: {e}")
                time.sleep(poll_interval)
                continue

            now = time.time()
            for row in tickers:
                inst_id = row["instId"]
                try:
                    last_price = float(row["last"])
                except (KeyError, ValueError, TypeError):
                    continue

                dq = history[inst_id]
                dq.append((now, last_price))
                while dq and now - dq[0][0] > max_window + poll_interval:
                    dq.popleft()

                if inst_id in last_alert and now - last_alert[inst_id] < cooldown:
                    continue

                for window, threshold in price_thresholds.items():
                    past = price_n_seconds_ago(dq, now, window)
                    if past is None or past == 0:
                        continue
                    pct_change = (last_price - past) / past * 100
                    if abs(pct_change) < threshold:
                        continue

                    try:
                        candles = fetch_recent_candles(inst_id, bar="1m", limit=20)
                    except Exception as e:
                        print(f"[confirm] candle fetch failed for {inst_id}: {e}")
                        continue
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

                    direction = "\U0001F680 UP" if pct_change > 0 else "\U0001F53B DOWN"
                    msg = (
                        f"{direction} spike: <b>{inst_id}</b>\n"
                        f"{pct_change:+.2f}% over {window}s\n"
                        f"Last: {last_price}\n"
                        f"1m volume: {vol_ratio:.1f}x avg (last {len(prior_vols)} candles)"
                    )

                    # Mean-reversion dip-buy: only DOWN spikes get a sized long plan.
                    if sizing_enabled and equity > 0 and pct_change < 0:
                        active_plans = {k: v for k, v in active_plans.items() if v > now}
                        plan = build_dip_trade_plan(
                            candles, last_price, scfg, sizer_cfg,
                            equity, open_positions=len(active_plans),
                        )
                        if plan is not None:
                            pos, stop_price = plan
                            msg += format_trade_plan(pos, last_price, stop_price)
                            active_plans[inst_id] = now + cooldown
                        else:
                            msg += "\n(no sized plan: max concurrent reached or below min size)"

                    send_telegram(token, chat_id, msg)
                    last_alert[inst_id] = now
                    print(f"[alert] {inst_id} {pct_change:+.2f}% / {window}s, vol {vol_ratio:.1f}x")
                    time.sleep(0.1)  # small throttle before next candle-confirm fetch
                    break

            elapsed = time.time() - loop_start
            time.sleep(max(0.5, poll_interval - elapsed))
    except KeyboardInterrupt:
        print("Stopping screener.")
        send_telegram(token, chat_id, "\U0001F534 OKX perp screener stopped.")


if __name__ == "__main__":
    main()
