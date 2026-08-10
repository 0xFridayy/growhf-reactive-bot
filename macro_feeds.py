"""
macro_feeds.py — BTC.D / USDT.D / DXY as analysable candle series.

`/analyze btc.d` used to fail with "unknown instrument BTC.D-USDT-SWAP" because
resolve_inst_id() just appends -USDT-SWAP to whatever you type. None of these three
IS an OKX perp (checked: 0 matches across 431 swaps), and Binance's real BTCDOMUSDT
futures is unreachable from here (403 / Cloudflare), so each needs its own feed.

The trick that makes them work with the rest of the bot: each is returned as OKX-shaped
candle rows, so build_tpo_profile / mean_reversion / regime consume them unchanged.

  DXY     Yahoo DX-Y.NYB — real OHLC bars, but the feed runs ~10 min behind.
  BTC.D   derived. CoinGecko refreshes its own dominance number only every 600s, far
  USDT.D  too slow, so instead CoinGecko supplies an hourly ANCHOR (circulating
          supplies + true total mcap) and OKX candles supply the prices:

            btc_t = supply_BTC * close_BTC(t)
            alt_t = SUM supply_i * close_i(t)      mcap-weighted crypto basket
            den_t = btc_t + alt_t + uncovered      uncovered = stables + long tail
            BTC.D(t) = 100*btc_t/den_t     USDT.D(t) = 100*usdt_mcap/den_t

          Holding `uncovered` constant is what keeps the moves correctly damped — a
          naive basket-only ratio overstates dominance swings by ~1/coverage (~1.23x).
          At anchor time den equals CoinGecko's reported total, so the level lands on
          the official number (verified: 56.474% derived vs 56.543% official).

Basket hygiene matters more than it looks: OKX's highest-turnover "alts" include
TOKENIZED EQUITIES AND COMMODITIES (SNDK, SKHYNIX, MU, SOXL, SPCX, CL, XAU), and OKX
metadata does not flag them — every USDT swap is category "1", ctType "linear". They
are excluded by requiring each symbol to match a CoinGecko top-250 crypto whose price
agrees with OKX within PX_TOL.

Derived candles carry no intrabar extremes (dominance highs and lows across coins are
not simultaneous, so inventing them would be fiction). Each bar is built line-style:
open = previous level, close = this level, high/low = max/min of the two.

NOTE: the same derivation lives in the Perps repo's okx_spike_screener.py, which
screens these three for spikes continuously. Duplicated deliberately — this repo ships
as its own container/service and must stay self-contained.
"""

import os
import time

import requests

OKX_BASE = "https://www.okx.com"
CG_BASE = "https://api.coingecko.com/api/v3"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
USER_AGENT = "Mozilla/5.0 (compatible; okx-tele-bot/1.0)"

BASKET_N = int(os.environ.get("MACRO_BASKET_N", "30"))   # alts priced per request
PX_TOL = float(os.environ.get("MACRO_PX_TOL", "0.05"))   # OKX-vs-CoinGecko price gap
ANCHOR_TTL = int(os.environ.get("MACRO_ANCHOR_TTL", "3600"))
SERIES_TTL = int(os.environ.get("MACRO_SERIES_TTL", "300"))

# Pinned-price assets: excluded from the alt basket (their supply, not their price, is
# the signal) and folded into the constant uncovered mass instead.
STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USDD", "USD1",
           "BUSD", "USDS", "EURC", "EURT", "USDG", "RLUSD", "USDF", "USD0"}

# What the user might type -> canonical symbol.
ALIASES = {
    "BTC.D": "BTC.D", "BTCD": "BTC.D", "BTCDOM": "BTC.D", "BTC-D": "BTC.D",
    "BTCDOMINANCE": "BTC.D", "DOMINANCE": "BTC.D",
    "USDT.D": "USDT.D", "USDTD": "USDT.D", "USDTM": "USDT.D", "USDT-D": "USDT.D",
    "USDTDOM": "USDT.D", "STABLES": "USDT.D",
    "DXY": "DXY", "DX-Y": "DXY", "DXY.D": "DXY", "USDX": "DXY", "DOLLAR": "DXY",
}

# symbol -> (display name, unit, meaning when rising, meaning when falling)
META = {
    "BTC.D": ("BTC dominance", "%",
              "favour BTC over alts", "alt rotation — alts outperforming"),
    "USDT.D": ("USDT dominance", "%",
               "risk-OFF — money fleeing into stables", "risk-ON — stables deploying"),
    "DXY": ("US Dollar Index", "",
            "dollar bid — crypto headwind", "dollar soft — crypto tailwind"),
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
if os.environ.get("INSECURE_TLS") == "1":       # sandbox parity with the Perps repo
    SESSION.verify = False

_anchor_cache = {"ts": 0.0, "data": None}
_series_cache = {}


def resolve_macro(query):
    """'btc.d', 'BTCDOM', 'dxy' -> canonical symbol, or None if not a macro symbol."""
    if not query:
        return None
    q = query.strip().upper().replace("/", "-").replace("_", "-")
    for suffix in ("-USDT-SWAP", "-USDT", "-SWAP"):     # tolerate a resolved instId
        if q.endswith(suffix):
            q = q[: -len(suffix)]
    return ALIASES.get(q)


def is_macro(query):
    return resolve_macro(query) is not None


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _get(url, params=None, timeout=20):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _okx_candles(inst_id, bar, limit):
    data = _get(f"{OKX_BASE}/api/v5/market/candles",
                {"instId": inst_id, "bar": bar, "limit": str(limit)}).get("data", [])
    return data          # newest first


# --------------------------------------------------------------------------- #
# CoinGecko anchor
# --------------------------------------------------------------------------- #
def _anchor(force=False):
    """{supply, inst, uncovered, usdt_mcap, total, coverage, ts} — cached ANCHOR_TTL."""
    now = time.time()
    if not force and _anchor_cache["data"] and now - _anchor_cache["ts"] < ANCHOR_TTL:
        return _anchor_cache["data"]

    mkts = _get(f"{CG_BASE}/coins/markets",
                {"vs_currency": "usd", "order": "market_cap_desc",
                 "per_page": 250, "page": 1}, timeout=30)
    glob = _get(f"{CG_BASE}/global", timeout=20)["data"]

    by_sym = {}
    for c in mkts:                                  # first hit wins = highest mcap
        sym = (c.get("symbol") or "").upper()
        if sym and c.get("current_price") and c.get("market_cap"):
            by_sym.setdefault(sym, c)

    rows = _get(f"{OKX_BASE}/api/v5/market/tickers", {"instType": "SWAP"}).get("data", [])
    prices = {}
    for r in rows:
        inst = r.get("instId", "")
        if inst.endswith("-USDT-SWAP"):
            try:
                px = float(r.get("last") or 0)
            except (TypeError, ValueError):
                continue
            if px > 0:
                prices[inst] = px

    matched = {}
    for inst, px in prices.items():
        sym = inst.split("-")[0]
        c = by_sym.get(sym)
        if not c or sym in matched:
            continue
        cg_px = float(c["current_price"])
        if cg_px <= 0 or abs(px / cg_px - 1.0) > PX_TOL:
            continue                                # tokenized equity / ticker clash
        supply = float(c.get("circulating_supply") or 0) or float(c["market_cap"]) / cg_px
        matched[sym] = {"inst": inst, "mcap": float(c["market_cap"]), "supply": supply}

    total = float(glob["total_market_cap"]["usd"])
    if "BTC" not in matched or total <= 0:
        raise RuntimeError("could not anchor dominance (BTC unmatched or no total mcap)")

    basket = sorted(((s, v) for s, v in matched.items() if s not in STABLES and s != "BTC"),
                    key=lambda kv: -kv[1]["mcap"])[:BASKET_N]
    covered = matched["BTC"]["mcap"] + sum(v["mcap"] for _, v in basket)
    data = {
        "supply": {"BTC": matched["BTC"]["supply"], **{s: v["supply"] for s, v in basket}},
        "inst": {"BTC": matched["BTC"]["inst"], **{s: v["inst"] for s, v in basket}},
        "uncovered": max(0.0, total - covered),
        "usdt_mcap": total * float(glob["market_cap_percentage"].get("usdt", 0.0)) / 100.0,
        "total": total,
        "coverage": covered / total,
        "ts": now,
    }
    _anchor_cache.update({"ts": now, "data": data})
    return data


# --------------------------------------------------------------------------- #
# Series construction
# --------------------------------------------------------------------------- #
def _line_candles(points):
    """[(ts_ms, level)] oldest-first -> OKX-shaped rows, newest-first.

    No intrabar extremes exist for a derived index, so each bar spans open->close
    only: high/low are max/min of the two, never invented wicks."""
    rows = []
    prev = points[0][1]
    for ts, lvl in points:
        o, c = prev, lvl
        rows.append([str(int(ts)), f"{o:.10g}", f"{max(o, c):.10g}", f"{min(o, c):.10g}",
                     f"{c:.10g}", "0", "0", "0", "1"])
        prev = lvl
    rows.reverse()
    return rows


def _dominance_series(sym, bar, limit):
    """Rebuild BTC.D / USDT.D history from anchored supplies x OKX closes."""
    a = _anchor()
    supply, inst = a["supply"], a["inst"]
    closes = {}
    for s, iid in inst.items():
        try:
            rows = _okx_candles(iid, bar, limit)
        except requests.RequestException:
            continue
        if rows:
            closes[s] = {int(r[0]): float(r[4]) for r in rows}
        time.sleep(0.05)                     # stay under the OKX candles rate limit
    if "BTC" not in closes:
        raise RuntimeError("no BTC candles for dominance")

    alt_supply_total = sum(v for s, v in supply.items() if s != "BTC" and s in closes)
    points = []
    for ts in sorted(closes["BTC"]):
        btc = supply["BTC"] * closes["BTC"][ts]
        alt = 0.0
        present = 0.0
        for s, series in closes.items():
            if s == "BTC" or ts not in series:
                continue
            alt += supply[s] * series[ts]
            present += supply[s]
        # Skip bars where much of the basket has no candle (new listings, gaps) —
        # a partial basket would read as a dominance spike that never happened.
        if alt_supply_total > 0 and present / alt_supply_total < 0.8:
            continue
        den = btc + alt + a["uncovered"]
        if den <= 0:
            continue
        num = btc if sym == "BTC.D" else a["usdt_mcap"]
        points.append((ts, 100.0 * num / den))

    if len(points) < 5:
        raise RuntimeError("not enough overlapping candles to derive dominance")
    return _line_candles(points), {
        "basket": len(closes) - 1,
        "coverage": a["coverage"],
        "anchor_age_min": (time.time() - a["ts"]) / 60.0,
        "source": "OKX closes x CoinGecko supplies",
    }


_YAHOO_RANGE = {"1m": "1d", "5m": "5d", "15m": "5d", "30m": "1mo",
                "1H": "1mo", "60m": "1mo", "1D": "1y", "1d": "1y"}


def _dxy_series(bar, limit):
    interval = {"1H": "60m", "1D": "1d"}.get(bar, bar)
    res = _get(YAHOO, {"interval": interval,
                       "range": _YAHOO_RANGE.get(bar, "5d")})["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c):
            continue
        rows.append([str(int(t) * 1000), f"{o:.10g}", f"{h:.10g}", f"{l:.10g}",
                     f"{c:.10g}", "0", "0", "0", "1"])
    if len(rows) < 5:
        raise RuntimeError("DXY feed returned no usable bars")
    rows = rows[-limit:]
    rows.reverse()                                   # newest first, OKX convention
    lag_min = (time.time() - int(rows[0][0]) / 1000) / 60.0
    return rows, {"source": "Yahoo DX-Y.NYB", "lag_min": lag_min}


def macro_candles(sym, bar="15m", limit=96):
    """OKX-shaped candles (newest-first) + meta dict, cached SERIES_TTL seconds."""
    sym = resolve_macro(sym) or sym
    if sym not in META:
        raise ValueError(f"not a macro symbol: {sym}")
    key = (sym, bar, limit)
    hit = _series_cache.get(key)
    now = time.time()
    if hit and now - hit[0] < SERIES_TTL:
        return hit[1], hit[2]
    if sym == "DXY":
        rows, meta = _dxy_series(bar, limit)
    else:
        rows, meta = _dominance_series(sym, bar, limit)
    _series_cache[key] = (now, rows, meta)
    return rows, meta
