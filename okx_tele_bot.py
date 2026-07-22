"""
okx_tele_bot.py — OKX perpetual-futures Telegram bot.

Three jobs, one process:
  1. OI-flip alerts      — watches open-interest regime vs price and alerts when
                           the positioning regime flips (e.g. "shorts building"
                           -> "longs building").
  2. Funding-flip alerts — alerts when the funding rate crosses zero (sentiment
                           flip) with enough magnitude to matter.
  3. On-demand analysis  — reply to a Telegram message ("/analyze SOL", or just
                           "sol") with price/OI/funding context and a verdict
                           (LONG / SHORT / NEUTRAL) plus an optional sized plan.

Uses only OKX public market-data endpoints (no API key/secret needed).
Config lives in config.json next to this file.

Run:
    python okx_tele_bot.py
"""

import json
import time
import traceback
from pathlib import Path

import requests

from kelly_sizing import SizingConfig, size_position

CONFIG_PATH = Path(__file__).with_name("config.json")
OKX_BASE = "https://www.okx.com"
TG_BASE = "https://api.telegram.org"
USER_AGENT = "okx-tele-bot/1.0"

# Positioning regimes derived from price direction x OI direction.
REGIME_LABELS = {
    ("up", "up"): ("\U0001F7E2 longs building", "bullish"),      # new money long
    ("up", "down"): ("\U0001F7E1 short covering", "weak-bull"),   # squeeze, not fresh demand
    ("down", "up"): ("\U0001F534 shorts building", "bearish"),    # new money short
    ("down", "down"): ("\U0001F7E0 long unwind", "weak-bear"),    # deleveraging
    ("flat", "up"): ("⚖️ OI up, price flat", "neutral"),
    ("flat", "down"): ("⚖️ OI down, price flat", "neutral"),
    ("up", "flat"): ("⚖️ price up, OI flat", "neutral"),
    ("down", "flat"): ("⚖️ price down, OI flat", "neutral"),
    ("flat", "flat"): ("⚖️ quiet", "neutral"),
}


# --------------------------------------------------------------------------- #
# Config / HTTP helpers
# --------------------------------------------------------------------------- #
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if str(cfg.get("telegram_bot_token", "")).startswith("PUT_YOUR"):
        raise SystemExit(
            f"Edit {CONFIG_PATH} first: fill in telegram_bot_token and telegram_chat_id."
        )
    return cfg


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


SESSION = _session()


def okx_get(path, params=None, timeout=10):
    r = SESSION.get(f"{OKX_BASE}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# OKX data
# --------------------------------------------------------------------------- #
def fetch_tickers(quote_filter):
    rows = okx_get("/api/v5/market/tickers", {"instType": "SWAP"}).get("data", [])
    if quote_filter:
        rows = [r for r in rows if r["instId"].split("-")[1:2] == [quote_filter]]
    return rows


def fetch_open_interest(quote_filter):
    """Return {instId: open_interest_ccy} for SWAP instruments."""
    rows = okx_get("/api/v5/public/open-interest", {"instType": "SWAP"}).get("data", [])
    out = {}
    for r in rows:
        inst = r.get("instId", "")
        if quote_filter and inst.split("-")[1:2] != [quote_filter]:
            continue
        try:
            out[inst] = float(r.get("oiCcy") or r.get("oi") or 0.0)
        except (TypeError, ValueError):
            continue
    return out


def fetch_funding(inst_id):
    """Current funding rate (fraction, e.g. 0.0001 = 0.01%) for one instrument."""
    data = okx_get("/api/v5/public/funding-rate", {"instId": inst_id}).get("data", [])
    if not data:
        return None
    try:
        return float(data[0]["fundingRate"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_candles(inst_id, bar="15m", limit=96):
    data = okx_get(
        "/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit}
    ).get("data", [])
    return data  # newest first: [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]


def resolve_inst_id(query, quote_filter):
    """Turn user text ('sol', 'sol-usdt', 'SOL-USDT-SWAP') into a valid SWAP instId."""
    q = query.strip().upper().replace("/", "-")
    quote = (quote_filter or "USDT").upper()
    if q.endswith("-SWAP"):
        return q
    if "-" in q:
        return f"{q}-SWAP" if not q.endswith("SWAP") else q
    return f"{q}-{quote}-SWAP"


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def tg_send(token, chat_id, text):
    try:
        SESSION.post(
            f"{TG_BASE}/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
    except requests.RequestException as e:
        print(f"[telegram] send failed: {e}")


def tg_get_updates(token, offset, timeout=20):
    try:
        r = SESSION.get(
            f"{TG_BASE}/bot{token}/getUpdates",
            params={"offset": offset, "timeout": timeout},
            timeout=timeout + 10,
        )
        r.raise_for_status()
        return r.json().get("result", [])
    except requests.RequestException as e:
        print(f"[telegram] getUpdates failed: {e}")
        return []


# --------------------------------------------------------------------------- #
# Signal logic
# --------------------------------------------------------------------------- #
def direction(pct, dead_zone):
    if pct > dead_zone:
        return "up"
    if pct < -dead_zone:
        return "down"
    return "flat"


def pct_change(new, old):
    if not old:
        return 0.0
    return (new - old) / old * 100.0


def analyze(inst_id, quote_filter, sizing_cfg=None):
    """
    Build a verdict for one instrument from momentum, OI, funding and volume.
    Returns (message_text, verdict_label). Raises on hard fetch failure.
    """
    tickers = {r["instId"]: r for r in fetch_tickers(quote_filter)}
    row = tickers.get(inst_id)
    if row is None:
        # Fall back to a direct ticker lookup (covers filtered-out quotes).
        data = okx_get("/api/v5/market/ticker", {"instId": inst_id}).get("data", [])
        if not data:
            raise ValueError(f"unknown instrument {inst_id}")
        row = data[0]

    last = float(row["last"])
    open24 = float(row.get("open24h") or last)
    vol24 = float(row.get("volCcy24h") or 0.0)
    chg24 = pct_change(last, open24)

    candles = fetch_candles(inst_id, bar="15m", limit=96)
    chg1h = chg4h = 0.0
    vol_ratio = 0.0
    if len(candles) >= 20:
        close_now = float(candles[0][4])
        chg1h = pct_change(close_now, float(candles[4][4]))     # 4 x 15m
        chg4h = pct_change(close_now, float(candles[16][4]))    # 16 x 15m
        cur_vol = float(candles[0][5])
        prior = [float(c[5]) for c in candles[1:17]]
        avg_vol = sum(prior) / len(prior) if prior else 0.0
        vol_ratio = cur_vol / avg_vol if avg_vol else 0.0

    funding = fetch_funding(inst_id)

    # Scoring: momentum + funding (contrarian) + volume conviction.
    score = 0.0
    reasons = []
    if chg1h > 1.0:
        score += 1
        reasons.append(f"1h momentum {chg1h:+.2f}%")
    elif chg1h < -1.0:
        score -= 1
        reasons.append(f"1h momentum {chg1h:+.2f}%")
    if chg4h > 2.0:
        score += 1
        reasons.append(f"4h trend {chg4h:+.2f}%")
    elif chg4h < -2.0:
        score -= 1
        reasons.append(f"4h trend {chg4h:+.2f}%")

    if funding is not None:
        fpct = funding * 100.0
        # Extreme funding is a contrarian signal (crowded side pays).
        if funding > 0.0005:
            score -= 1
            reasons.append(f"crowded longs (funding {fpct:+.4f}%)")
        elif funding < -0.0005:
            score += 1
            reasons.append(f"crowded shorts (funding {fpct:+.4f}%)")

    if vol_ratio >= 3.0:
        reasons.append(f"volume {vol_ratio:.1f}x avg")

    if score >= 2:
        verdict = "\U0001F7E2 LONG bias"
    elif score <= -2:
        verdict = "\U0001F534 SHORT bias"
    else:
        verdict = "⚖️ NEUTRAL / no edge"

    lines = [
        f"\U0001F50D <b>{inst_id}</b>",
        f"Last {last:.6g} | 24h {chg24:+.2f}% | vol ${vol24:,.0f}",
        f"1h {chg1h:+.2f}% | 4h {chg4h:+.2f}%"
        + (f" | vol {vol_ratio:.1f}x" if vol_ratio else ""),
    ]
    if funding is not None:
        lines.append(f"Funding {funding * 100:+.4f}% / 8h")
    lines.append(f"\n<b>Verdict:</b> {verdict}")
    if reasons:
        lines.append("• " + "\n• ".join(reasons))

    # Optional sized plan when the verdict has a directional bias.
    if sizing_cfg and sizing_cfg.get("enabled") and "NEUTRAL" not in verdict:
        plan = _sized_plan(last, chg4h, score, sizing_cfg)
        if plan:
            lines.append(plan)

    return "\n".join(lines), verdict


def _sized_plan(entry, chg4h, score, scfg):
    """Rough quarter-Kelly plan: stop from recent volatility, 1R target."""
    equity = float(scfg.get("account_equity_usd", 0) or 0)
    if equity <= 0:
        return None
    sizer = SizingConfig(
        kelly_fraction=float(scfg.get("kelly_fraction", 0.25)),
        max_risk_per_trade=float(scfg.get("max_risk_per_trade", 0.02)),
        max_leverage=float(scfg.get("max_leverage", 3.0)),
        exchange_max_leverage=float(scfg.get("exchange_max_leverage", 50.0)),
        min_notional_usd=float(scfg.get("min_notional_usd", 5.0)),
        max_concurrent=int(scfg.get("max_concurrent", 3)),
    )
    stop_frac = min(
        float(scfg.get("max_stop_frac", 0.05)),
        max(float(scfg.get("min_stop_frac", 0.004)), abs(chg4h) / 100.0),
    )
    long_side = score > 0
    stop_price = entry * (1 - stop_frac) if long_side else entry * (1 + stop_frac)
    pos = size_position(
        equity_usd=equity, entry_price=entry, stop_price=stop_price, cfg=sizer
    )
    if pos is None:
        return None
    tp = entry + (entry - stop_price)
    side = "LONG" if long_side else "SHORT"
    return (
        f"\n\U0001F4C8 <b>{side} plan (quarter-Kelly)</b>\n"
        f"Entry ~{entry:.6g} | Stop {stop_price:.6g} ({stop_frac * 100:.2f}%) | TP {tp:.6g} (+1R)\n"
        f"Lev {pos.leverage}x | Notional ${pos.notional_usd} | Risk ${pos.risk_usd} [{pos.reason}]"
    )


# --------------------------------------------------------------------------- #
# Command handling
# --------------------------------------------------------------------------- #
def handle_command(text, cfg):
    quote_filter = cfg.get("quote_filter") or "USDT"
    sizing_cfg = cfg.get("sizing", {})
    parts = text.strip().split()
    if not parts:
        return None
    cmd = parts[0].lower().lstrip("/").split("@")[0]

    if cmd in ("start", "help"):
        return (
            "\U0001F916 <b>OKX Telegram bot</b>\n"
            "Commands:\n"
            "• <code>/analyze BTC</code> — full read + verdict (or just send <code>btc</code>)\n"
            "• <code>/status</code> — what I'm watching\n"
            "• <code>/help</code> — this message\n\n"
            "I also push OI-flip and funding-flip alerts automatically."
        )
    if cmd == "status":
        of = cfg.get("oi_funding", {})
        return (
            "✅ <b>Watching</b>\n"
            f"OI/funding scan: {'on' if of.get('enabled', True) else 'off'} "
            f"(every {of.get('scan_interval_seconds', 300)}s, "
            f"top {of.get('universe_top_n', 40)} by volume)\n"
            f"Quote filter: {quote_filter}"
        )

    target = parts[1] if cmd == "analyze" and len(parts) > 1 else (
        parts[0] if cmd not in ("analyze",) else None
    )
    if cmd == "analyze" and len(parts) < 2:
        return "Usage: <code>/analyze BTC</code>"
    if target is None:
        return None

    inst_id = resolve_inst_id(target, quote_filter)
    try:
        msg, _ = analyze(inst_id, quote_filter, sizing_cfg)
        return msg
    except Exception as e:  # noqa: BLE001 - report cleanly to the user
        return f"⚠️ Couldn't analyze <b>{inst_id}</b>: {e}"


# --------------------------------------------------------------------------- #
# Background OI / funding flip scan
# --------------------------------------------------------------------------- #
class FlipScanner:
    def __init__(self, cfg):
        self.cfg = cfg
        of = cfg.get("oi_funding", {})
        self.enabled = of.get("enabled", True)
        self.quote_filter = cfg.get("quote_filter") or "USDT"
        self.top_n = int(of.get("universe_top_n", 40))
        self.oi_dead_zone = float(of.get("oi_change_threshold_pct", 5.0))
        self.price_dead_zone = float(of.get("price_change_threshold_pct", 0.5))
        self.funding_min_abs = float(of.get("funding_flip_min_abs", 0.0001))
        self.cooldown = float(of.get("cooldown_seconds", 3600))
        self.interval = float(of.get("scan_interval_seconds", 300))
        self.last_scan = 0.0
        self.prev_oi = {}          # instId -> oi
        self.prev_price = {}       # instId -> price
        self.prev_regime = {}      # instId -> regime label
        self.prev_funding = {}     # instId -> funding rate
        self.last_alert = {}       # instId -> ts

    def due(self, now):
        return self.enabled and (now - self.last_scan) >= self.interval

    def _cooling(self, inst, now):
        return inst in self.last_alert and (now - self.last_alert[inst]) < self.cooldown

    def scan(self, token, chat_id):
        now = time.time()
        self.last_scan = now
        try:
            tickers = fetch_tickers(self.quote_filter)
            oi = fetch_open_interest(self.quote_filter)
        except Exception as e:  # noqa: BLE001
            print(f"[scan] fetch failed: {e}")
            return

        # Restrict to the most liquid names to keep funding calls bounded.
        tickers.sort(key=lambda r: float(r.get("volCcy24h") or 0.0), reverse=True)
        universe = [r["instId"] for r in tickers[: self.top_n]]
        prices = {r["instId"]: float(r["last"]) for r in tickers}

        for inst in universe:
            price = prices.get(inst)
            cur_oi = oi.get(inst)
            if price is None or cur_oi is None:
                continue

            self._check_oi_flip(inst, price, cur_oi, now, token, chat_id)
            self.prev_oi[inst] = cur_oi
            self.prev_price[inst] = price

        self._check_funding_flips(universe, now, token, chat_id)

    def _check_oi_flip(self, inst, price, cur_oi, now, token, chat_id):
        old_oi = self.prev_oi.get(inst)
        old_price = self.prev_price.get(inst)
        if old_oi is None or old_price is None:
            return
        oi_dir = direction(pct_change(cur_oi, old_oi), self.oi_dead_zone)
        px_dir = direction(pct_change(price, old_price), self.price_dead_zone)
        label, tone = REGIME_LABELS.get((px_dir, oi_dir), ("", "neutral"))
        prev = self.prev_regime.get(inst)
        self.prev_regime[inst] = label
        if not label or prev is None or label == prev or tone == "neutral":
            return
        if self._cooling(inst, now):
            return
        oi_chg = pct_change(cur_oi, old_oi)
        px_chg = pct_change(price, old_price)
        tg_send(
            token,
            chat_id,
            f"\U0001F504 <b>OI flip: {inst}</b>\n"
            f"{prev} → {label}\n"
            f"Price {px_chg:+.2f}% | OI {oi_chg:+.2f}% (last scan)\n"
            f"Last {price:.6g}",
        )
        self.last_alert[inst] = now
        print(f"[oi-flip] {inst}: {prev} -> {label}")

    def _check_funding_flips(self, universe, now, token, chat_id):
        for inst in universe:
            try:
                funding = fetch_funding(inst)
            except Exception:  # noqa: BLE001
                continue
            if funding is None:
                continue
            prev = self.prev_funding.get(inst)
            self.prev_funding[inst] = funding
            if prev is None:
                continue
            crossed = (prev <= 0 < funding) or (prev >= 0 > funding)
            if not crossed or abs(funding) < self.funding_min_abs:
                continue
            if self._cooling(inst, now):
                continue
            side = "positive (longs pay)" if funding > 0 else "negative (shorts pay)"
            tg_send(
                token,
                chat_id,
                f"\U0001F4B8 <b>Funding flip: {inst}</b>\n"
                f"{prev * 100:+.4f}% → {funding * 100:+.4f}% / 8h\n"
                f"Now {side}",
            )
            self.last_alert[inst] = now
            print(f"[funding-flip] {inst}: {prev:+.5f} -> {funding:+.5f}")
            time.sleep(0.05)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]

    scanner = FlipScanner(cfg)
    offset = 0

    print(
        f"Starting OKX Telegram bot. quote_filter={cfg.get('quote_filter')} "
        f"oi/funding_scan={'on' if scanner.enabled else 'off'} "
        f"interval={scanner.interval}s top_n={scanner.top_n}"
    )
    tg_send(token, chat_id, "\U0001F7E2 OKX Telegram bot online. Send /help.")

    try:
        while True:
            # 1) Handle inbound commands (short long-poll keeps us responsive).
            for upd in tg_get_updates(token, offset, timeout=10):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("channel_post")
                if not msg:
                    continue
                text = msg.get("text", "")
                reply = handle_command(text, cfg)
                if reply:
                    tg_send(token, str(msg["chat"]["id"]), reply)

            # 2) Run the OI/funding flip scan when due.
            if scanner.due(time.time()):
                scanner.scan(token, chat_id)
    except KeyboardInterrupt:
        print("Stopping bot.")
        tg_send(token, chat_id, "\U0001F534 OKX Telegram bot stopped.")
    except Exception:  # noqa: BLE001 - never die silently under systemd
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
