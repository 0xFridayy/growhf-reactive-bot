"""
Hourly crypto bias -> Telegram.

Posts a TOTAL3 / BTC.D / USDT.D + macro read to Telegram. Designed to be fired
by Windows Task Scheduler at 19:00-00:00 local; runs once and exits.

Reads telegram_bot_token / telegram_chat_id from config.json (shared with the
other bots in this repo). Stdlib only -- no extra dependencies.

Method
------
CoinGecko's free tier returns only a CURRENT dominance snapshot, no history.
Over a 3-5 day window dominance is driven by relative PRICE, not supply. So we
anchor on the live snapshot and propagate backwards along hourly OKX price paths:

    BTC_mcap(t) = BTC_mcap_now * P_btc(t)/P_btc(now)
    ETH_mcap(t) = ETH_mcap_now * P_eth(t)/P_eth(now)
    STABLE(t)   = STABLE_now                        (pegged, supply ~flat)
    ALT(t)      = ALT_now * BASKET(t)/BASKET(now)   (mcap-weighted alt index)
    TOTAL(t)    = BTC + ETH + STABLE + ALT

TOTAL3 follows the TradingView definition (TOTAL - BTC - ETH), which includes
stablecoins. TOTAL3X excludes them -- that is the real alt risk-appetite number
and what the scoring uses.

These deltas are RECONSTRUCTED, not measured tape. Every run also appends a
genuine snapshot to bias_history.csv so measured history accumulates.
"""

import csv
import html
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")
HIST = os.path.join(HERE, "bias_history.csv")
LOG = os.path.join(HERE, "bias_telegram.log")
TZ = timezone(timedelta(hours=7))  # SE Asia Standard Time

STABLES = {
    "usdt", "usdc", "dai", "usde", "fdusd", "pyusd", "usds", "busd", "tusd",
    "usd1", "rlusd", "usdd", "frax", "lusd", "gusd", "eurc", "usdf", "usdg",
    "susds", "susde", "buidl", "usdtb", "usdy", "bsc-usd", "xaut", "paxg",
}
DERIVS = {
    "wbtc", "weth", "steth", "wsteth", "weeth", "wbeth", "reth", "cbbtc",
    "lbtc", "solvbtc", "ezeth", "rseth", "meth", "cbeth", "sweth", "beth",
    "tbtc", "pumpbtc", "stbtc", "wbnb", "bnsol", "jitosol", "msol", "wsol",
    "jupsol", "oseth", "ethx", "sfrxeth", "frxeth", "clbtc", "enzobtc",
    "unibtc", "stone", "rsweth", "bbsol", "bsol",
}
EXCLUDE = STABLES | DERIVS
UA = {"User-Agent": "Mozilla/5.0 (compatible; bias-bot/1.0)"}


def log(msg):
    line = f"{datetime.now(TZ):%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def get(url, tries=3, pause=2.0, raw=False):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                return data if raw else json.loads(data.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(pause * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


# ------------------------------------------------------------------ telegram
with open(CONFIG, encoding="utf-8") as f:
    cfg = json.load(f)
TOKEN = cfg["telegram_bot_token"]
CHAT = str(cfg["telegram_chat_id"])


def send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    body = urllib.parse.urlencode({
        "chat_id": CHAT,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(url, data=body, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------- market inputs
def build():
    g = get("https://api.coingecko.com/api/v3/global")["data"]
    total_now = g["total_market_cap"]["usd"]
    vol_now = g["total_volume"]["usd"]
    mcp = g["market_cap_percentage"]
    btcd_now, ethd_now = mcp["btc"], mcp["eth"]
    usdtd_now = mcp.get("usdt", 0.0)

    mk = get(
        "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
        "&order=market_cap_desc&per_page=100&page=1"
        "&price_change_percentage=1h,24h,7d"
    )
    by_sym = {c["symbol"].lower(): c for c in mk if c.get("market_cap")}

    stable_mcap = sum(c["market_cap"] for s, c in by_sym.items() if s in STABLES)
    usdt_mcap = total_now * usdtd_now / 100.0
    btc_mcap = total_now * btcd_now / 100.0
    eth_mcap = total_now * ethd_now / 100.0
    alt_mcap = total_now - btc_mcap - eth_mcap - stable_mcap

    basket = [c for s, c in by_sym.items()
              if s not in EXCLUDE and s not in ("btc", "eth")]
    basket.sort(key=lambda c: -c["market_cap"])
    basket = basket[:24]

    okx = "https://www.okx.com/api/v5/market/candles?instId={}-USDT&bar=1H&limit=100"

    def candles(sym):
        try:
            d = get(okx.format(sym.upper()), tries=2, pause=1.0).get("data") or []
        except Exception:  # noqa: BLE001
            return None
        if len(d) < 80:
            return None
        return {int(r[0]) // 1000: float(r[4]) for r in d}

    series = {}
    for sym in ["btc", "eth"] + [c["symbol"].lower() for c in basket]:
        s = candles(sym)
        if s:
            series[sym] = s
    if "btc" not in series or "eth" not in series:
        raise RuntimeError("no BTC/ETH history from OKX")

    ts = sorted(set.intersection(*[set(v) for v in series.values()]), reverse=True)
    if len(ts) < 74:
        series = {k: v for k, v in series.items() if k in ("btc", "eth")}
        ts = sorted(set(series["btc"]) & set(series["eth"]), reverse=True)
    t_now = ts[0]

    alt_syms = [s for s in series if s not in ("btc", "eth")]
    weights = {s: by_sym[s]["market_cap"] for s in alt_syms}
    wsum = sum(weights.values()) or 1.0

    path = []
    for t in ts:
        rb = series["btc"][t] / series["btc"][t_now]
        re_ = series["eth"][t] / series["eth"][t_now]
        ra = (sum(weights[s] * series[s][t] / series[s][t_now]
                  for s in alt_syms) / wsum) if alt_syms else re_
        b, e, a = btc_mcap * rb, eth_mcap * re_, alt_mcap * ra
        tot = b + e + stable_mcap + a
        path.append({
            "t": t, "total": tot, "total3": stable_mcap + a, "total3x": a,
            "btcd": 100.0 * b / tot, "ethd": 100.0 * e / tot,
            "usdtd": 100.0 * usdt_mcap / tot,
            "btc": series["btc"][t], "eth": series["eth"][t],
        })
    return path, {"vol": vol_now, "nalts": len(alt_syms), "hours": len(ts)}


# ------------------------------------------------------------------ scoring
def score(path):
    P = {p["t"]: p for p in path}
    now = path[0]
    t_now = now["t"]

    def chg(field, hours):
        old = P.get(t_now - hours * 3600)
        if not old:
            return None
        if field in ("btcd", "ethd", "usdtd"):
            return now[field] - old[field]
        return 100.0 * (now[field] / old[field] - 1)

    vals = [p["total3x"] for p in path[:72]]
    lo, hi = min(vals), max(vals)
    rng = 50.0 if hi == lo else 100.0 * (now["total3x"] - lo) / (hi - lo)

    votes = []

    def vote(n, s, d):
        votes.append((n, max(-2.0, min(2.0, s)), d))

    t1, t4, t24, t72 = (chg("total3x", h) for h in (1, 4, 24, 72))
    m = sum(w * v for v, w in ((t1, .8), (t4, .5), (t24, .35), (t72, .15))
            if v is not None)
    vote("TOTAL3 mom", m, f"1h {t1:+.2f}% 4h {t4:+.2f}% 24h {t24:+.2f}%")
    vote("TOTAL3 range", (rng - 50) / 25.0, f"{rng:.0f}th pctile/72h")

    bd4, bd24 = chg("btcd", 4), chg("btcd", 24)
    vote("BTC.D trend", -(bd4 * 6.0 + bd24 * 2.5), f"4h {bd4:+.3f}pp")
    ud4, ud24 = chg("usdtd", 4), chg("usdtd", 24)
    vote("USDT.D trend", -(ud4 * 14.0 + ud24 * 6.0), f"4h {ud4:+.3f}pp")
    b4, b24 = chg("btc", 4), chg("btc", 24)
    vote("Alt vs BTC", (t24 - b24) * 0.9, f"{t24:+.2f}% vs {b24:+.2f}%")
    vote("BTC trend", b4 * 0.7 + b24 * 0.25, f"4h {b4:+.2f}%")

    raw = sum(v for _, v, _ in votes)
    norm = 100.0 * raw / (len(votes) * 2.0)

    if norm >= 35:
        lab, emo = "BULLISH", "\U0001F7E2"
    elif norm >= 12:
        lab, emo = "LEAN BULLISH", "\U0001F7E1"
    elif norm > -12:
        lab, emo = "NEUTRAL / CHOP", "⚪"
    elif norm > -35:
        lab, emo = "LEAN BEARISH", "\U0001F7E0"
    else:
        lab, emo = "BEARISH", "\U0001F534"

    return {"now": now, "chg": chg, "rng": rng, "votes": votes,
            "norm": norm, "label": lab, "emoji": emo,
            "bd4": bd4, "bd24": bd24, "ud4": ud4, "ud24": ud24,
            "b4": b4, "b24": b24, "t1": t1, "t4": t4, "t24": t24, "t72": t72}


# -------------------------------------------------------------------- macro
def macro():
    out = {}
    syms = {"DX-Y.NYB": "DXY", "^TNX": "US10Y", "ES=F": "ES", "^VIX": "VIX"}
    for s, n in syms.items():
        try:
            u = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
                 f"{urllib.parse.quote(s)}?range=5d&interval=1h")
            r = get(u, tries=2, pause=1.0)["chart"]["result"][0]
            c = [x for x in r["indicators"]["quote"][0]["close"] if x is not None]
            if len(c) > 2:
                out[n] = (c[-1], 100.0 * (c[-1] / c[0] - 1))
        except Exception:  # noqa: BLE001
            continue
    return out


def headlines(n=3):
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    out = []
    for f in feeds:
        try:
            root = ET.fromstring(get(f, tries=1, raw=True))
            for item in root.iter("item"):
                t = item.findtext("title")
                if t:
                    out.append(t.strip())
                if len(out) >= n:
                    return out[:n]
        except Exception:  # noqa: BLE001
            continue
    return out[:n]


# ------------------------------------------------------------------- compose
def fmt_b(x):
    return f"${x/1e9:.1f}B" if x < 1e12 else f"${x/1e12:.3f}T"


def compose(S, meta, mac, news):
    now, chg = S["now"], S["chg"]
    stamp = datetime.now(TZ)
    L = []
    L.append(f"{S['emoji']} <b>CRYPTO BIAS — {stamp:%H:%M} WIB</b>  "
             f"<i>{stamp:%a %d %b}</i>")
    L.append(f"<b>{S['label']}</b>  ·  score <b>{S['norm']:+.0f}/100</b>")
    L.append("")
    L.append(f"<b>TOTAL3</b>  {fmt_b(now['total3'])}  "
             f"<i>(ex-stables {fmt_b(now['total3x'])})</i>")
    L.append(f"   1h {S['t1']:+.2f}% · 4h {S['t4']:+.2f}% · 24h {S['t24']:+.2f}%"
             f" · 72h {S['t72']:+.2f}%")
    L.append(f"   range: <b>{S['rng']:.0f}th</b> pctile of 72h")
    L.append(f"<b>BTC.D</b>   {now['btcd']:.2f}%   "
             f"4h {S['bd4']:+.3f}pp · 24h {S['bd24']:+.3f}pp")
    L.append(f"<b>USDT.D</b>  {now['usdtd']:.3f}%  "
             f"4h {S['ud4']:+.3f}pp · 24h {S['ud24']:+.3f}pp")
    L.append(f"<b>ETH.D</b>   {now['ethd']:.2f}%")
    L.append("")
    L.append(f"BTC <b>${now['btc']:,.0f}</b>  4h {S['b4']:+.2f}% · "
             f"24h {S['b24']:+.2f}%")
    L.append(f"ETH <b>${now['eth']:,.2f}</b>  24h {chg('eth', 24):+.2f}%")
    L.append(f"TOTAL {fmt_b(now['total'])} · vol {fmt_b(meta['vol'])}")

    L.append("")
    L.append("<b>Drivers</b> <i>(+ = risk-on for alts)</i>")
    for n_, v, d in S["votes"]:
        bar = ("+" if v >= 0 else "−") * max(1, int(round(abs(v) * 3)))
        L.append(f"<code>{n_:<13}{v:+5.2f} {bar:<6}</code> {d}")

    # deterministic read of the BTC.D / alt divergence
    L.append("")
    alt_lag = (S["t24"] is not None and S["b24"] is not None
               and S["t24"] < S["b24"])
    if S["bd4"] > 0.05 and alt_lag:
        L.append("⚠️ <b>BTC-led.</b> BTC.D rising while alts lag — "
                 "risk-on is concentrating in BTC. Express long in BTC, "
                 "not alts. Need BTC.D to roll over before trusting TOTAL3.")
    elif S["bd4"] < -0.05 and not alt_lag:
        L.append("✅ <b>Rotation live.</b> BTC.D falling while alts "
                 "outperform — genuine alt bid.")
    elif S["ud4"] > 0.05:
        L.append("⚠️ <b>Risk-off.</b> USDT.D rising — capital "
                 "fleeing to stables.")
    else:
        L.append("ℹ️ <b>Mixed.</b> No clean rotation signal; "
                 "treat TOTAL3 as range-bound.")
    if S["rng"] > 85:
        L.append("• TOTAL3 pinned at top of its 3-day range — "
                 "stall risk into resistance.")
    elif S["rng"] < 15:
        L.append("• TOTAL3 at bottom of its 3-day range — "
                 "bounce risk for shorts.")

    if mac:
        L.append("")
        parts = []
        for k in ("DXY", "VIX", "ES", "US10Y"):
            if k in mac:
                lv, ch = mac[k]
                parts.append(f"{k} {lv:,.2f} ({ch:+.1f}% 5d)")
        L.append("<b>Macro</b>  " + " · ".join(parts))
        if "DXY" in mac and "VIX" in mac:
            d_ch, v_lv = mac["DXY"][1], mac["VIX"][0]
            if d_ch < -0.5 and v_lv < 20:
                L.append("   → dollar down + vol low = tailwind")
            elif d_ch > 0.5 or v_lv > 25:
                L.append("   → dollar up / vol elevated = headwind")

    if news:
        L.append("")
        L.append("<b>Headlines</b>")
        for h in news:
            L.append(f"• {html.escape(h[:110])}")

    L.append("")
    L.append(f"<i>Dominance deltas reconstructed from {meta['hours']}h OKX "
             f"paths + live CoinGecko anchor ({meta['nalts']} alts), not "
             f"measured tape.</i>")
    return "\n".join(L)


def _write_history(S):
    now = S["now"]
    stamp = datetime.now(TZ)
    new = not os.path.exists(HIST)
    try:
        with open(HIST, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["local_time", "total_usd", "total3_usd", "total3x_usd",
                            "btcd", "ethd", "usdtd", "btc", "eth", "score", "label"])
            w.writerow([stamp.strftime("%Y-%m-%d %H:%M"), f"{now['total']:.0f}",
                        f"{now['total3']:.0f}", f"{now['total3x']:.0f}",
                        f"{now['btcd']:.3f}", f"{now['ethd']:.3f}",
                        f"{now['usdtd']:.3f}", f"{now['btc']:.1f}",
                        f"{now['eth']:.2f}", f"{S['norm']:.1f}", S["label"]])
    except OSError as e:
        log(f"history write failed: {e}")


def current_bias(max_age_min=90):
    """Just the number: {'score': -100..100, 'label': str, 'age_min': float,
    'fresh': bool}. For consumers that need the REGIME, not the report.

    Reads the newest row of bias_history.csv first — the scheduled job appends
    one every run, so the common case costs zero network calls. Only when that
    row is stale (or missing) does it rebuild, which is ~26 OKX calls plus
    CoinGecko and takes several seconds. Callers on a fast polling loop should
    therefore cache the result, not call this every tick.

    Never raises: a screener must not stop screening because the bias feed is
    down. On total failure it returns score 0 / label UNKNOWN / fresh False,
    which callers should treat as "no regime opinion".
    """
    try:
        if os.path.exists(HIST):
            with open(HIST, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                t = datetime.strptime(last["local_time"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
                age = (datetime.now(TZ) - t).total_seconds() / 60.0
                if age <= max_age_min:
                    return {"score": float(last["score"]), "label": last["label"],
                            "age_min": age, "fresh": True}
    except Exception as e:  # noqa: BLE001
        log(f"current_bias: history unreadable ({e}) — rebuilding")

    try:
        S = score(build()[0])
        _write_history(S)
        return {"score": S["norm"], "label": S["label"], "age_min": 0.0, "fresh": True}
    except Exception as e:  # noqa: BLE001
        log(f"current_bias: rebuild failed ({e})")
        return {"score": 0.0, "label": "UNKNOWN", "age_min": float("inf"), "fresh": False}


def render_bias(write_history=True):
    """Build the full bias message and return it as a Telegram HTML string.

    Does NOT send. Raises on a fatal data failure; macro/headlines degrade
    silently since they are garnish, not the signal. Shared by the scheduled
    job below and by /bias in okx_tele_bot.py -- one source of truth.
    """
    path, meta = build()
    S = score(path)
    mac, news = {}, []
    try:
        mac = macro()
    except Exception as e:  # noqa: BLE001
        log(f"macro failed: {e}")
    try:
        news = headlines()
    except Exception as e:  # noqa: BLE001
        log(f"headlines failed: {e}")
    if write_history:
        _write_history(S)
    log(f"rendered score={S['norm']:+.0f} {S['label']}")
    return compose(S, meta, mac, news)


# ---------------------------------------------------------------------- main
def main():
    try:
        msg = render_bias()
    except Exception as e:  # noqa: BLE001
        log(f"ERROR building bias: {e}")
        try:
            send(f"⛔ <b>Bias job failed</b>\n<code>{html.escape(str(e))[:400]}</code>")
        except Exception:  # noqa: BLE001
            pass
        return 1
    try:
        r = send(msg)
        log(f"sent ok={r.get('ok')}")
    except Exception as e:  # noqa: BLE001
        log(f"ERROR sending: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
