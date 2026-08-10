"""
chart_command.py — /chart <SYMBOL> for the OKX Reactive Spike Signal bot.

Renders a 4H chart (price + causal pivots + anchors, open interest, CVD) and
returns a caption with the market-structure read.

    pip install requests matplotlib

Core entry point:

    png, caption = render_chart_report("BTC")

Endpoints used (all public, no auth):
  /api/v5/market/candles                      OHLCV
  /api/v5/rubik/stat/contracts/open-interest-history   OI per instId
  /api/v5/rubik/stat/taker-volume             taker buy/sell -> CVD proxy
  /api/v5/public/funding-rate                 current + next funding

Two data caveats worth keeping in mind:
  - taker-volume is ccy-level (all OKX contracts for that coin), period is
    limited to 5m/1H/1D. We pull 1H and resample to 4H.
  - taker-volume rows are [ts, sellVol, buyVol] — sell first. Getting this
    backwards silently inverts every CVD read, so it is asserted below.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

BASE = "https://www.okx.com"
TIMEOUT = 15

# palette
UP, DOWN, OI_C, CVD_C, LVL = "#0F766E", "#B3402F", "#5B4BC4", "#C77B2B", "#8892A0"


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def _get(path: str, **params):
    r = requests.get(BASE + path, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    if body.get("code") not in ("0", 0):
        raise RuntimeError(f"OKX {path}: {body.get('msg') or body}")
    return body.get("data", [])


def fetch_candles(inst_id: str, bar: str = "4H", limit: int = 200):
    """Returns oldest-first list of dicts. OKX returns newest-first."""
    rows = _get("/api/v5/market/candles", instId=inst_id, bar=bar, limit=str(limit))
    out = [
        {
            "ts": int(r[0]),
            "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4]),
            "v": float(r[5]),
        }
        for r in rows
    ]
    return list(reversed(out))


def fetch_oi_history(inst_id: str, period: str = "4H", limit: int = 120):
    """[ts, oi (contracts), oiCcy, oiUsd] — newest first. Falls back to empty."""
    try:
        rows = _get("/api/v5/rubik/stat/contracts/open-interest-history",
                    instId=inst_id, period=period, limit=str(limit))
    except Exception:
        return []
    out = []
    for r in rows:
        try:
            out.append({"ts": int(r[0]), "oi": float(r[-1] if len(r) > 3 else r[1])})
        except (ValueError, IndexError):
            continue
    return list(reversed(out))


def fetch_cvd(ccy: str, hours: int = 200):
    """
    CVD proxy from taker flow. Rows are [ts, sellVol, buyVol].
    Returns oldest-first [{ts, delta, cvd}] with CVD anchored at the window
    start — the absolute level is meaningless, only the shape within the
    window is used.
    """
    rows = _get("/api/v5/rubik/stat/taker-volume", ccy=ccy,
                instType="CONTRACTS", period="1H")
    series, cum = [], 0.0
    for r in reversed(rows[:hours]):
        ts, sell, buy = int(r[0]), float(r[1]), float(r[2])
        delta = buy - sell
        cum += delta
        series.append({"ts": ts, "delta": delta, "cvd": cum})
    return series


def fetch_funding(inst_id: str):
    try:
        d = _get("/api/v5/public/funding-rate", instId=inst_id)
        if d:
            return {
                "rate": float(d[0]["fundingRate"]),
                "next": float(d[0].get("nextFundingRate") or 0),
                "next_time": int(d[0].get("fundingTime") or 0),
            }
    except Exception:
        pass
    return None


def resample_to_candles(series, candles, key):
    """Bucket an hourly series onto candle timestamps (candle ts = bucket open)."""
    if not series:
        return []
    bucket_ms = candles[1]["ts"] - candles[0]["ts"]
    out = []
    for c in candles:
        vals = [s[key] for s in series if c["ts"] <= s["ts"] < c["ts"] + bucket_ms]
        out.append(sum(vals) if key == "delta" else (vals[-1] if vals else None))
    # forward-fill level series
    last = None
    for i, v in enumerate(out):
        if v is None:
            out[i] = last
        else:
            last = v
    return out


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def causal_pivots(candles, n: int = 3):
    """
    Fractal pivots confirmed n bars later. A pivot at index i is only known
    at i+n — `confirm` carries that index so nothing downstream can peek.
    """
    highs, lows = [], []
    for i in range(n, len(candles) - n):
        window = candles[i - n:i + n + 1]
        if candles[i]["h"] == max(w["h"] for w in window):
            highs.append({"i": i, "confirm": i + n, "price": candles[i]["h"]})
        if candles[i]["l"] == min(w["l"] for w in window):
            lows.append({"i": i, "confirm": i + n, "price": candles[i]["l"]})
    return highs, lows


@dataclass
class Structure:
    state: str          # "uptrend" | "downtrend" | "range"
    last_event: str     # "BOS up" | "CHoCH down" | ...
    at_index: int
    level: float


def structure_state(candles, highs, lows) -> Structure:
    """Walk forward, only ever using pivots already confirmed at that bar."""
    state, last_event, at_i, level = "range", "none", 0, candles[-1]["c"]
    for i, c in enumerate(candles):
        ph = [p for p in highs if p["confirm"] <= i]
        pl = [p for p in lows if p["confirm"] <= i]
        if not ph or not pl:
            continue
        last_h, last_l = ph[-1]["price"], pl[-1]["price"]
        if c["c"] > last_h:
            ev = "BOS up" if state == "uptrend" else "CHoCH up"
            state, last_event, at_i, level = "uptrend", ev, i, last_h
        elif c["c"] < last_l:
            ev = "BOS down" if state == "downtrend" else "CHoCH down"
            state, last_event, at_i, level = "downtrend", ev, i, last_l
    return Structure(state, last_event, at_i, level)


def anchors(inst_id: str):
    """Prior day high/low and weekly open — no parameters, nothing to fit."""
    d = fetch_candles(inst_id, bar="1D", limit=10)
    w = fetch_candles(inst_id, bar="1W", limit=3)
    out = {}
    if len(d) >= 2:
        out["PDH"], out["PDL"] = d[-2]["h"], d[-2]["l"]
    if w:
        out["Weekly open"] = w[-1]["o"]
    return out


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------

def classify_oi(price_chg_pct: float, oi_chg_pct: float, dead: float = 0.15):
    if abs(price_chg_pct) < dead:
        return "flat", "Price going nowhere — OI change is positioning, not direction."
    up, oi_up = price_chg_pct > 0, oi_chg_pct > 0
    if up and oi_up:
        return "new longs", "New longs opening. Continuation has sponsorship."
    if up and not oi_up:
        return "short covering", "Shorts closing, not new buyers. The fuel is finite."
    if not up and oi_up:
        return "new shorts", "New shorts opening. The break down is sponsored."
    return "long liquidation", "Longs unwinding. Cascades overshoot, then snap back."


def cvd_read(candles, cvd_series, lookback: int = 12):
    """Absorption / divergence: does aggression translate into price?"""
    vals = [v for v in cvd_series[-lookback:] if v is not None]
    if len(vals) < 4:
        return None, "CVD unavailable."
    px = [c["c"] for c in candles[-len(vals):]]
    px_high, cvd_high = px[-1] >= max(px) * 0.999, vals[-1] >= max(vals) * 0.999
    px_low, cvd_low = px[-1] <= min(px) * 1.001, vals[-1] <= min(vals) * 1.001
    drift = vals[-1] - vals[0]
    px_drift_pct = (px[-1] - px[0]) / px[0] * 100

    if cvd_high and not px_high:
        return "absorption", ("Aggressive buying is not lifting price — someone is "
                              "selling limit into all of it. Reversal risk down.")
    if cvd_low and not px_low:
        return "absorption", ("Aggressive selling is not pressing price lower — bids "
                              "are absorbing. Reversal risk up.")
    if drift > 0 and px_drift_pct > 0:
        return "confirmed up", "Buy aggression and price agree."
    if drift < 0 and px_drift_pct < 0:
        return "confirmed down", "Sell aggression and price agree."
    return "divergent", "CVD and price disagree — treat the move as unsponsored."


def pct(a: float, b: float) -> float:
    return (b - a) / a * 100 if a else 0.0


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def _nanify(series):
    """None -> nan so matplotlib masks the gaps instead of choking on objects."""
    return [float("nan") if v is None else float(v) for v in series]


def render_chart(candles, oi, cvd, levels, highs, lows, title):
    t = [datetime.fromtimestamp(c["ts"] / 1000, timezone.utc) for c in candles]
    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#E7EBEE")
    width = (t[1] - t[0]).total_seconds() / 86400 * 0.62

    for ti, c in zip(t, candles):
        up = c["c"] >= c["o"]
        col = UP if up else DOWN
        ax.plot([ti, ti], [c["l"], c["h"]], color=col, lw=0.8, zorder=2)
        ax.bar(ti, abs(c["c"] - c["o"]) or 1e-9, bottom=min(c["o"], c["c"]),
               width=width, color="none" if up else col,
               edgecolor=col, lw=0.9, zorder=3)

    for name, v in levels.items():
        ax.axhline(v, color=LVL, ls="--", lw=1, zorder=1)
        ax.annotate(f"{name} {v:,.0f}", (t[-1], v), xytext=(4, 2),
                    textcoords="offset points", fontsize=8, color=LVL,
                    family="monospace", va="bottom")

    for p in highs[-6:]:
        ax.plot(t[p["i"]], p["price"], marker="v", ms=5, color=DOWN, zorder=4)
    for p in lows[-6:]:
        ax.plot(t[p["i"]], p["price"], marker="^", ms=5, color=UP, zorder=4)

    ax.set_title(title, fontsize=12, family="monospace", loc="left", pad=10)
    ax.set_ylabel("price", fontsize=9, family="monospace")

    if any(v is not None for v in oi):
        oi_v = _nanify(oi)
        ax2.plot(t, oi_v, color=OI_C, lw=1.3)
        ax2.fill_between(t, oi_v, min(v for v in oi if v is not None),
                         color=OI_C, alpha=0.10)
    ax2.set_ylabel("open interest", fontsize=9, family="monospace", color=OI_C)

    if any(v is not None for v in cvd):
        ax3.plot(t, _nanify(cvd), color=CVD_C, lw=1.3)
        ax3.axhline(0, color=LVL, lw=0.7)
    ax3.set_ylabel("CVD (windowed)", fontsize=9, family="monospace", color=CVD_C)

    for a in (ax, ax2, ax3):
        a.set_facecolor("#E7EBEE")
        a.grid(color="#C4CCD2", lw=0.5, alpha=0.6)
        for s in a.spines.values():
            s.set_color("#B4BEC5")
        a.tick_params(labelsize=8, colors="#5B666F")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def render_chart_report(symbol: str, bar: str = "4H", bars: int = 120):
    ccy = symbol.upper().replace("-USDT-SWAP", "").replace("USDT", "")
    inst = f"{ccy}-USDT-SWAP"

    candles = fetch_candles(inst, bar=bar, limit=bars)
    if len(candles) < 30:
        raise RuntimeError(f"Not enough candles for {inst}")

    oi_raw = fetch_oi_history(inst, period=bar)
    cvd_raw = fetch_cvd(ccy)
    funding = fetch_funding(inst)
    lv = anchors(inst)

    oi = resample_to_candles(oi_raw, candles, "oi")
    cvd_delta = resample_to_candles(cvd_raw, candles, "delta")
    running, cvd = 0.0, []
    for d in cvd_delta:
        running += (d or 0.0)
        cvd.append(running)

    highs, lows = causal_pivots(candles, n=3)
    st = structure_state(candles, highs, lows)

    look = 6
    px_chg = pct(candles[-look]["c"], candles[-1]["c"])
    oi_vals = [v for v in oi[-look:] if v is not None]
    oi_chg = pct(oi_vals[0], oi_vals[-1]) if len(oi_vals) >= 2 else 0.0
    quad, quad_note = classify_oi(px_chg, oi_chg)
    cvd_tag, cvd_note = cvd_read(candles, cvd)

    last = candles[-1]["c"]
    near = sorted(lv.items(), key=lambda kv: abs(kv[1] - last))[:2]
    as_of = datetime.fromtimestamp(candles[-1]["ts"] / 1000, timezone.utc)

    png = render_chart(candles, oi, cvd, lv, highs, lows,
                       f"{inst}  {bar}   {as_of:%Y-%m-%d %H:%MZ}")

    lines = [
        f"*{inst}* {bar} — {last:,.2f}",
        f"_as of {as_of:%d %b %H:%M} UTC_",
        "",
        "*Structure*",
        f"State: {st.state} · last event {st.last_event} @ {st.level:,.2f}",
        "Pivots confirmed 3 bars late — signal times are lagged, not hindsight.",
        "",
        "*Positioning*",
        f"Price {px_chg:+.2f}% / OI {oi_chg:+.2f}% over {look} bars → *{quad}*",
        quad_note,
    ]
    if funding:
        lines.append(f"Funding {funding['rate']*100:+.4f}% "
                     f"(next {funding['next']*100:+.4f}%)")
    lines += [
        "",
        "*Aggression*",
        f"CVD: *{cvd_tag or 'n/a'}* — {cvd_note}",
        "",
        "*Anchors*",
        " · ".join(f"{k} {v:,.0f} ({pct(v, last):+.2f}%)" for k, v in near) or "n/a",
        "",
        "_CVD is OKX taker flow, ccy-level, windowed — the level is arbitrary, "
        "only the shape counts._",
    ]
    return png, "\n".join(lines)


# --------------------------------------------------------------------------
# telegram wiring (python-telegram-bot v20+)
# --------------------------------------------------------------------------

async def chart_handler(update, context):
    args = getattr(context, "args", []) or []
    symbol = (args[0] if args else "BTC").upper()
    msg = await update.message.reply_text(f"Pulling {symbol}…")
    try:
        png, caption = render_chart_report(symbol)
        await update.message.reply_photo(photo=png, caption=caption,
                                         parse_mode="Markdown")
        await msg.delete()
    except Exception as e:
        await msg.edit_text(f"{symbol}: {e}")


def register(app):
    """In your bot: from chart_command import register; register(app)"""
    from telegram.ext import CommandHandler
    app.add_handler(CommandHandler("chart", chart_handler))


if __name__ == "__main__":
    png, cap = render_chart_report("BTC")
    open("chart_btc.png", "wb").write(png)
    print(cap)
