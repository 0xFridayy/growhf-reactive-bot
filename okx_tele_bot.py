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
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from kelly_sizing import SizingConfig, size_position
from macro_feeds import META as MACRO_META, macro_candles, resolve_macro
from market_profile import build_tpo, build_tpo_profile, mean_reversion, regime

CONFIG_PATH = Path(__file__).with_name("config.json")
# Scan state for --once runs. Small (a few KB), unlike the news SQLite blob, so
# committing it back from CI each run stays cheap.
STATE_PATH = Path(__file__).with_name("okx_scan_state.json")
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

def regime_bias(label):
    """Return a simple bias for downstream scoring from the OI regime label."""
    if label == "🟢 longs building":
        return "long"
    if label == "🔴 shorts building":
        return "short"
    return "neutral"

# --------------------------------------------------------------------------- #
# Config / HTTP helpers
# --------------------------------------------------------------------------- #
def load_config():
    """Config from config.json, or from the environment when it is absent.

    CI has no config.json (it is gitignored, and secrets must not be committed),
    so GitHub Actions supplies TG_BOT_TOKEN/TG_CHAT_ID instead and the rest
    falls back to config.example.json's defaults.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        example = Path(__file__).with_name("config.example.json")
        cfg = json.loads(example.read_text(encoding="utf-8")) if example.exists() else {}

    # Environment always wins, so CI never depends on a committed secret.
    if os.environ.get("TG_BOT_TOKEN"):
        cfg["telegram_bot_token"] = os.environ["TG_BOT_TOKEN"]
    if os.environ.get("TG_CHAT_ID"):
        cfg["telegram_chat_id"] = os.environ["TG_CHAT_ID"]

    token = str(cfg.get("telegram_bot_token", ""))
    if not token or token.startswith(("PUT_YOUR", "REPLACE_WITH")):
        raise SystemExit(
            f"No Telegram token: set TG_BOT_TOKEN, or fill in {CONFIG_PATH}."
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


def tg_send_photo(token, chat_id, png_bytes, caption):
    try:
        SESSION.post(
            f"{TG_BASE}/bot{token}/sendPhoto",
            data={
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "Markdown",
            },
            files={"photo": ("chart.png", png_bytes, "image/png")},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"[telegram] sendPhoto failed: {e}")


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
    if abs(pct) <= dead_zone:
        return "flat"
    if pct > 0:
        return "up"
    return "down"


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
        try:
            data = okx_get("/api/v5/market/ticker", {"instId": inst_id}).get("data", [])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"unknown instrument {inst_id}") from exc
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

    # Market-profile analytics (TPO POC/VAH/VAL, mean reversion, regime).
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    prev_session = today_start - timedelta(days=1)
    prof = build_tpo_profile(candles, session_start=today_start) if candles else None
    prof_today = prof
    prof_prev = build_tpo(candles, session=prev_session) if candles else None
    mr = mean_reversion(candles) if candles else None
    reg = regime(candles) if candles else None

    # Scoring: momentum + funding (contrarian) + volume + regime-aware fade.
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

    # Regime decides which playbook drives the verdict. In a range, fade the
    # extremes back toward POC; in a trend, let momentum carry the score.
    if reg is not None and reg.favored == "mean-reversion" and mr is not None and prof is not None:
        if prof_prev is not None and prof_today is not None:
            poc_drift = prof_today.poc - prof_prev.poc
            price_move = last - prof_today.poc
            if (poc_drift > 0 and price_move > 0) or (poc_drift < 0 and price_move < 0):
                reasons.append("POC drift agrees with price move → trend, no fade")
            elif mr.signal == "fade-short" and last > prof.vah:
                score -= 2
                reasons.append("range regime: price stretched high → fade toward POC")
            elif mr.signal == "fade-long" and last < prof.val:
                score += 2
                reasons.append("range regime: price stretched low → fade toward POC")
            else:
                reasons.append("profile is inside value / near POC → no fade edge")
        elif mr.signal == "fade-short" and last > prof.vah:
            score -= 2
            reasons.append("range regime: price stretched high → fade toward POC")
        elif mr.signal == "fade-long" and last < prof.val:
            score += 2
            reasons.append("range regime: price stretched low → fade toward POC")

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

    if reg is not None:
        lines.append(f"\n\U0001F9ED <b>Regime:</b> {reg.label} — favours {reg.favored}")

    if prof is not None:
        loc = "inside value" if prof.in_value_area else (
            "above value (premium)" if last > prof.vah else "below value (discount)"
        )
        lines.append(
            f"\n\U0001F4CA <b>TPO profile (24h)</b>\n"
            f"POC {prof.poc:.6g} ({prof.poc_dist_pct:+.2f}%) | "
            f"VAH {prof.vah:.6g} | VAL {prof.val:.6g}\n"
            f"Price is {loc} [{prof.value_area_pct:.0f}% VA]"
        )

    if mr is not None:
        lines.append(
            f"\n\U0001F501 <b>Mean reversion</b>\n"
            f"Mean {mr.mean:.6g} | ±2σ [{mr.lower:.6g}, {mr.upper:.6g}]\n"
            f"{mr.note}"
        )

    lines.append(f"\n<b>Verdict:</b> {verdict}")
    if reasons:
        lines.append("• " + "\n• ".join(reasons))

    # Optional sized plan when the verdict has a directional bias.
    if sizing_cfg and sizing_cfg.get("enabled") and "NEUTRAL" not in verdict:
        plan = _sized_plan(last, chg4h, score, sizing_cfg)
        if plan:
            lines.append(plan)

    # Edge check: report whether CVD/structure support an execution edge and show details
    try:
        from okx_perp_screener import spike_has_edge
        import chart_command as cc

        # Build chart-style candles (oldest-first dicts) from the API newest-first rows
        chart_candles = [
            {
                "ts": int(c[0]),
                "o": float(c[1]),
                "h": float(c[2]),
                "l": float(c[3]),
                "c": float(c[4]),
                "v": float(c[5]),
            }
            for c in reversed(candles or [])
            if len(c) >= 6
        ]

        ccy = inst_id.split("-")[0]
        cvd_series = cc.fetch_cvd(ccy)
        cvd_vals = cc.resample_to_candles(cvd_series, chart_candles, "cvd") if cvd_series else []
        cvd_tag, cvd_msg = (None, None)
        if cvd_vals:
            cvd_tag, cvd_msg = cc.cvd_read(chart_candles, cvd_vals)

        highs, lows = cc.causal_pivots(chart_candles, n=3) if chart_candles else ([], [])
        st = cc.structure_state(chart_candles, highs, lows) if chart_candles else None

        spike_side = None
        if "LONG" in verdict or "LONG bias" in verdict:
            spike_side = "long"
        elif "SHORT" in verdict or "SHORT bias" in verdict:
            spike_side = "short"

        if spike_side:
            has_edge = spike_has_edge(inst_id, candles, spike_side)
            if has_edge:
                lines.append("\n🧭 Edge: ✅ CVD/structure support this bias")
            else:
                lines.append("\n🧭 Edge: ❌ no CVD/structure support — treat cautiously")
        else:
            long_edge = spike_has_edge(inst_id, candles, "long")
            short_edge = spike_has_edge(inst_id, candles, "short")
            if long_edge or short_edge:
                edges = []
                if long_edge:
                    edges.append("long")
                if short_edge:
                    edges.append("short")
                lines.append(f"\n🧭 Edge: ✅ available for {', '.join(edges)}")
            else:
                lines.append("\n🧭 Edge: ❌ none detected (CVD/structure divergent)")

        # Add detailed context when available
        details = []
        if cvd_tag:
            details.append(f"CVD: {cvd_tag} — {cvd_msg}")
        if st is not None:
            details.append(f"Structure: {st.state} / {st.last_event} (idx {st.at_index})")
        if details:
            lines.append("\n" + " · ".join(details))
    except Exception:
        # Non-fatal: leave analyze() working if chart helpers are unavailable
        pass

    return "\n".join(lines), verdict


def analyze_macro(sym, bar="15m", limit=96):
    """
    Same read as analyze(), for a macro gauge (BTC.D / USDT.D / DXY).

    Deliberately NOT analyze() with a different symbol: there is no ticker, no funding,
    no open interest and no volume behind these, and none of the three is executable on
    OKX — so this returns positioning context and never a sized trade plan.
    """
    candles, meta = macro_candles(sym, bar=bar, limit=limit)
    name, unit, up_msg, down_msg = MACRO_META[sym]

    closes = [float(c[4]) for c in candles]          # newest first
    last = closes[0]

    def chg(bars):
        return pct_change(last, closes[bars]) if len(closes) > bars else 0.0

    chg1h, chg4h, chg_full = chg(4), chg(16), chg(len(closes) - 1)

    prof = build_tpo_profile(candles)
    mr = mean_reversion(candles)
    reg = regime(candles)

    # Momentum scoring only — no funding/volume inputs exist for an index. Thresholds
    # are an order of magnitude tighter than the perp side because dominance moves in
    # basis points, not percent (measured 5m sigma: BTC.D 0.018%, DXY 0.016%).
    thresh_1h, thresh_4h = (0.15, 0.30) if unit == "%" else (0.10, 0.20)
    score = 0.0
    reasons = []
    if abs(chg1h) > thresh_1h:
        score += 1 if chg1h > 0 else -1
        reasons.append(f"1h {chg1h:+.3f}%")
    if abs(chg4h) > thresh_4h:
        score += 1 if chg4h > 0 else -1
        reasons.append(f"4h {chg4h:+.3f}%")
    if reg is not None and reg.favored == "mean-reversion" and mr is not None:
        if mr.signal == "fade-short":
            score -= 1
            reasons.append("range regime: stretched high → expect mean reversion")
        elif mr.signal == "fade-long":
            score += 1
            reasons.append("range regime: stretched low → expect mean reversion")

    if score >= 1:
        verdict = f"\U0001F7E2 {sym} rising — {up_msg}"
    elif score <= -1:
        verdict = f"\U0001F534 {sym} falling — {down_msg}"
    else:
        verdict = f"⚖️ {sym} flat / no clear regime signal"

    span_h = len(closes) * (0.25 if bar == "15m" else 1)
    lines = [
        f"\U0001F50D <b>{sym}</b> — {name} <i>(derived, not an OKX perp)</i>",
        f"Level {last:.4f}{unit} | {span_h:.0f}h {chg_full:+.3f}% | "
        f"1h {chg1h:+.3f}% | 4h {chg4h:+.3f}%",
    ]
    if "lag_min" in meta:
        lines.append(f"Source: {meta['source']} · feed lag {meta['lag_min']:.0f}m")
    else:
        lines.append(f"Source: {meta['source']} · {meta['basket']} alts, "
                     f"{meta['coverage'] * 100:.0f}% of crypto mcap · "
                     f"anchored {meta['anchor_age_min']:.0f}m ago")

    if reg is not None:
        lines.append(f"\n\U0001F9ED <b>Regime:</b> {reg.label} — favours {reg.favored}")
    if prof is not None:
        loc = "inside value" if prof.in_value_area else (
            "above value (premium)" if last > prof.vah else "below value (discount)"
        )
        lines.append(
            f"\n\U0001F4CA <b>TPO profile</b>\n"
            f"POC {prof.poc:.4f} ({prof.poc_dist_pct:+.2f}%) | "
            f"VAH {prof.vah:.4f} | VAL {prof.val:.4f}\n"
            f"Level is {loc} [{prof.value_area_pct:.0f}% VA]"
        )
    if mr is not None:
        lines.append(
            f"\n\U0001F501 <b>Mean reversion</b>\n"
            f"Mean {mr.mean:.4f} | ±2σ [{mr.lower:.4f}, {mr.upper:.4f}]\n"
            f"{mr.note}"
        )

    lines.append(f"\n<b>Verdict:</b> {verdict}")
    if reasons:
        lines.append("• " + "\n• ".join(reasons))
    lines.append("\n<i>Regime context — not tradeable on OKX, so no sized plan.</i>")
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
def _alpha_report(which, *args):
    """Reads alpha_ml/status.py's persisted results. Lazy, path-scoped import
    so the live bot never needs xgboost/torch installed just to answer these —
    status.py itself is stdlib-only, and the heavy training/search runs
    elsewhere and leaves its output on disk."""
    try:
        alpha_ml_dir = str(Path(__file__).with_name("alpha_ml"))
        if alpha_ml_dir not in sys.path:
            sys.path.insert(0, alpha_ml_dir)
        import status as alpha_status
        return getattr(alpha_status, which)(*args)
    except Exception as e:  # noqa: BLE001 - report cleanly, never crash the bot over this
        return f"⚠️ Alpha ML report unavailable: {e}"


def handle_command(text, cfg, ack=None):
    """Return the reply for an inbound Telegram message.

    `ack` is an optional callable(str) used to send an immediate interim reply
    for commands that take long enough to look hung. It is optional so the
    existing two-arg callers (and tests) keep working.
    """
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
            "• <code>/bias</code> — market bias now: TOTAL3, BTC.D, USDT.D + macro\n"
            "  same read the 19:00–00:00 auto-post sends\n"
            "• <code>/analyze BTC</code> — full read + verdict (or just send <code>btc</code>)\n"
            "  includes TPO profile (POC/VAH/VAL), mean-reversion z-score, and regime\n"
            "• <code>/profile BTC</code> / <code>/regime BTC</code> — same full read\n"
            "• <code>/analyze BTC.D</code> — macro gauges: <code>BTC.D</code> "
            "(a.k.a. BTCDOM), <code>USDT.D</code>, <code>DXY</code>\n"
            "  regime context only — derived indices, so no sized plan\n"
            "• <code>/status</code> — what I'm watching\n"
            "• <code>/alpha_status</code> — phase-1 XGBoost + phase-2 DDQN training/backtest progress\n"
            "• <code>/alpha_search</code> — today's entry/exit strategy search: "
            "what won, and whether it clears your bar\n"
            "• <code>/alpha_trades [n]</code> — the last n trades with the reason "
            "each was entered and exited\n"
            "• <code>/help</code> — this message\n\n"
            "I also push OI-flip and funding-flip alerts automatically."
        )

    if cmd == "bias":
        # Imported lazily: a broken/missing bias_telegram must not stop this
        # bot from starting or from serving every other command.
        if ack:
            ack("⏳ Building bias — a few seconds…")
        try:
            from bias_telegram import render_bias

            return render_bias()
        except Exception as e:  # noqa: BLE001 - report cleanly to the user
            return f"⚠️ Couldn't build bias: {e}"
    if cmd == "chart":
        symbol = parts[1] if len(parts) > 1 else "BTC"
        if ack:
            ack(f"⏳ Building chart for {symbol}…")
        try:
            from chart_command import render_chart_report

            png, caption = render_chart_report(symbol)
            return {"type": "photo", "png": png, "caption": caption}
        except Exception as e:  # noqa: BLE001 - report cleanly to the user
            return f"⚠️ Couldn't build chart for <b>{symbol}</b>: {e}"
    if cmd == "status":
        of = cfg.get("oi_funding", {})
        return (
            "✅ <b>Watching</b>\n"
            f"OI/funding scan: {'on' if of.get('enabled', True) else 'off'} "
            f"(every {of.get('scan_interval_seconds', 300)}s, "
            f"top {of.get('universe_top_n', 40)} by volume)\n"
            f"Quote filter: {quote_filter}"
        )
    if cmd in ("alpha_status", "alpha"):
        return _alpha_report("format_telegram")
    if cmd in ("alpha_search", "alpha_strategy"):
        return _alpha_report("format_search_telegram")
    if cmd in ("alpha_trades", "trades"):
        limit = 8
        if len(parts) > 1:
            try:
                limit = max(1, min(25, int(parts[1])))   # keep it inside one Telegram message
            except ValueError:
                return "Usage: <code>/alpha_trades 10</code>"
        return _alpha_report("format_trades_telegram", limit)

    named_cmds = ("analyze", "profile", "regime")
    if cmd in named_cmds:
        if len(parts) < 2:
            return f"Usage: <code>/{cmd} BTC</code>"
        target = parts[1]
    else:
        target = parts[0]
    if target is None:
        return None

    # Macro gauges first — they are not OKX instruments, so appending -USDT-SWAP
    # would only ever produce "unknown instrument BTC.D-USDT-SWAP".
    macro_sym = resolve_macro(target)
    if macro_sym:
        try:
            msg, _ = analyze_macro(macro_sym)
            return msg
        except Exception as e:  # noqa: BLE001 - report cleanly to the user
            return f"⚠️ Couldn't analyze <b>{macro_sym}</b>: {e}"

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
        self.oi_flip_enabled = bool(of.get("oi_flip_enabled", True))
        self.funding_min_abs = float(of.get("funding_flip_min_abs", 0.0001))
        self.cooldown = float(of.get("cooldown_seconds", 3600))
        self.interval = float(of.get("scan_interval_seconds", 300))
        self.last_scan = 0.0
        self.prev_oi = {}          # instId -> oi
        self.prev_price = {}       # instId -> price
        self.prev_regime = {}      # instId -> regime label
        self.prev_funding = {}     # instId -> funding rate
        self.last_alert = {}       # instId -> ts

    # State lives in memory for the long-running process, but a --once run is a
    # fresh process every time: with empty prev_oi there is nothing to compare
    # against, so no flip would ever be detected. Persist it between runs.
    STATE_KEYS = ("prev_oi", "prev_price", "prev_regime", "prev_funding", "last_alert")

    def load_state(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            return False
        for key in self.STATE_KEYS:
            saved = blob.get(key)
            if isinstance(saved, dict):
                setattr(self, key, saved)
        return True

    def save_state(self, path):
        blob = {k: getattr(self, k) for k in self.STATE_KEYS}
        blob["saved_at"] = datetime.now(timezone.utc).isoformat()
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=1, default=str)
        os.replace(tmp, path)   # atomic, so a killed run cannot leave a half file

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

            # OI-flip alerts removed per user request — do not call the checker.
            # Keep history for other features (funding flips, diagnostics).
            self.prev_oi[inst] = cur_oi
            self.prev_price[inst] = price

        self._check_funding_flips(universe, now, token, chat_id)

    def _check_oi_flip(self, inst, price, cur_oi, now, token, chat_id):
        # OI-flip logic has been removed per user request; do nothing.
        return

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
def _serve_pending(token, cfg, offset=0):
    """Handle every queued Telegram update once, and return the next offset."""
    for upd in tg_get_updates(token, offset, timeout=0):
        offset = upd["update_id"] + 1
        msg = upd.get("message") or upd.get("channel_post")
        if not msg:
            continue
        origin = str(msg["chat"]["id"])
        reply = handle_command(
            msg.get("text", ""), cfg, ack=lambda t, c=origin: tg_send(token, c, t)
        )
        if reply:
            if isinstance(reply, dict) and reply.get("type") == "photo":
                tg_send_photo(token, origin, reply["png"], reply["caption"])
            else:
                tg_send(token, origin, reply)
    return offset


def run_once():
    """One scheduled pass: answer queued commands, run one scan, exit.

    For GitHub Actions/cron. Telegram drops updates once they are confirmed by
    a getUpdates call with a higher offset, so the final confirm below is what
    stops the next run from replaying the same commands — no offset file needed.
    """
    cfg = load_config()
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]

    scanner = FlipScanner(cfg)
    had_state = scanner.load_state(STATE_PATH)
    print(f"run-once: previous scan state {'loaded' if had_state else 'not found (first run: baseline only)'}")

    offset = _serve_pending(token, cfg)
    if offset:
        tg_get_updates(token, offset, timeout=0)   # confirm -> Telegram discards

    if scanner.enabled:
        scanner.scan(token, chat_id)
    scanner.save_state(STATE_PATH)
    print(f"run-once: done, tracking {len(scanner.prev_oi)} instruments")


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
                origin = str(msg["chat"]["id"])
                reply = handle_command(
                    text, cfg, ack=lambda t, c=origin: tg_send(token, c, t)
                )
                if reply:
                    if isinstance(reply, dict) and reply.get("type") == "photo":
                        tg_send_photo(token, origin, reply["png"], reply["caption"])
                    else:
                        tg_send(token, origin, reply)

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
    if "--once" in sys.argv:
        run_once()
    else:
        main()
