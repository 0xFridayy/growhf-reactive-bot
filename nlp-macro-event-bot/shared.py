"""
SHARED — common config, Telegram, classifier, Haiku scorer, unified signal store
================================================================================
Single source of truth for the pieces that used to be copy-pasted between
macro_event_bot.py and news_nlp_bot.py, plus the new ingestion layers
(feeds_ws.py, telegram_listener.py) and the sentiment engine.

Design goal: import-safe with ZERO heavy deps at module load (no feedparser,
no apscheduler, no telethon). Only `requests` + stdlib. That way every new
module can `from shared import ...` and still run/test independently.

Env vars (all optional until the feature that needs them is switched on):
  TG_BOT_TOKEN, TG_CHAT_ID              Telegram bot creds — same bot/chat as
                                         okx_spike_screener.py, so all alerts
                                         (spike signals + macro/sentiment) land
                                         in one Telegram bot. Falls back to
                                         MACRO_BOT_TOKEN/MACRO_BOT_CHAT_ID.
  ANTHROPIC_API_KEY                    Claude Haiku scoring (existing)
  FINNHUB_API_KEY                      Finnhub economic calendar (free tier)
  FRED_API_KEY                         FRED release values (free)
  COINMARKETCAL_API_KEY                CoinMarketCal crypto events (free tier)
  TREE_API_KEY                         Tree of Alpha WS (optional; free feed w/o)
  PHOENIX_API_KEY                      Phoenix News WS (free base tier)
  TELEGRAM_API_ID, TELEGRAM_API_HASH   Telethon user-account listener
"""

import hashlib
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

# Windows consoles default to cp1252 and choke on the emoji/Unicode these bots
# emit. Force UTF-8 on the console for every module that imports shared.
# (Telegram messages are UTF-8 over HTTP and are unaffected either way.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BOT_TOKEN     = os.environ.get("TG_BOT_TOKEN") or os.environ.get("MACRO_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN")
CHAT_ID       = os.environ.get("TG_CHAT_ID") or os.environ.get("MACRO_BOT_CHAT_ID", "PASTE_YOUR_CHAT_ID")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "PASTE_YOUR_API_KEY")

FINNHUB_KEY       = os.environ.get("FINNHUB_API_KEY", "")
FRED_KEY          = os.environ.get("FRED_API_KEY", "")
COINMARKETCAL_KEY = os.environ.get("COINMARKETCAL_API_KEY", "")
TREE_KEY          = os.environ.get("TREE_API_KEY", "")           # optional
PHOENIX_KEY       = os.environ.get("PHOENIX_API_KEY", "")

HAIKU_MODEL = "claude-haiku-4-5"

WIB = ZoneInfo("Asia/Jakarta")
ET  = ZoneInfo("America/New_York")
UTC = timezone.utc

# Databases (SQLite, flat in the project dir — matches existing convention)
NEWS_DB     = os.environ.get("NEWS_DB", "news_nlp.db")   # written by news_nlp_bot.py
SIGNAL_DB   = os.environ.get("SIGNAL_DB", "signals.db")  # unified store for new layers

_TRUTHY_PLACEHOLDERS = {"PASTE_YOUR_BOT_TOKEN", "PASTE_YOUR_CHAT_ID",
                        "PASTE_YOUR_API_KEY", ""}


def have(value):
    """True if a credential is actually set (not a placeholder)."""
    return value not in _TRUTHY_PLACEHOLDERS


# ----------------------------------------------------------------------
# TELEGRAM (outbound bot messages — shared by all modules)
# ----------------------------------------------------------------------
def send(text, chat_id=None):
    if not have(BOT_TOKEN):
        print(f"[telegram disabled — no token] {text[:80]}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id or CHAT_ID, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=15)
    except requests.RequestException as e:
        print(f"[send error] {e}")


# ----------------------------------------------------------------------
# BINANCE PUBLIC SNAPSHOT (no key)
# ----------------------------------------------------------------------
def btc_snapshot(short=False):
    try:
        t = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                         params={"symbol": "BTCUSDT"}, timeout=10).json()
        price = float(t["lastPrice"])
        chg = float(t["priceChangePercent"])
        if short:
            return f"BTC ${price:,.0f} ({chg:+.2f}%)"
        p = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": "BTCUSDT"}, timeout=10).json()
        funding = float(p["lastFundingRate"]) * 100
        arrow = "🟢" if chg >= 0 else "🔴"
        return (f"{arrow} <b>BTC</b> ${price:,.0f}  ({chg:+.2f}% 24h)\n"
                f"⚙️ Funding: {funding:+.4f}%")
    except Exception as e:
        return f"BTC snapshot unavailable ({e})"


# ----------------------------------------------------------------------
# STAGE 1 — RULE CLASSIFIER (canonical taxonomy — the single source of truth)
# category: (keywords, base_score) — score 1..5, only >=2 escalates to the LLM
# ----------------------------------------------------------------------
TAXONOMY = {
    "FED":        (["fomc", "fed ", "federal reserve", "warsh", "rate cut",
                    "rate hike", "interest rate", "powell", "monetary policy",
                    "dot plot", "quantitative", "jackson hole", "testimony"], 4),
    "INFLATION":  (["cpi", "inflation", "pce", "ppi", "core price"], 4),
    "JOBS":       (["nonfarm", "payroll", "unemployment", "jobless",
                    "labor market"], 3),
    "REGULATION": (["sec ", "cftc", "clarity act", "genius act", "regulation",
                    "lawsuit", "settlement", "congress", "senate", "treasury",
                    "strategic bitcoin reserve", "executive order"], 4),
    "ETF_FLOWS":  (["etf inflow", "etf outflow", "blackrock", "ibit",
                    "fidelity", "spot etf", "grayscale"], 3),
    "LISTING":    (["will list", "lists ", "listing", "perpetual", "spot trading",
                    "adds support", "now available"], 4),
    "HACK":       (["hack", "exploit", "stolen", "breach", "drained",
                    "vulnerability"], 4),
    "STABLECOIN": (["depeg", "tether", "usdt", "usdc", "circle",
                    "stablecoin"], 3),
    "CORP_WHALE": (["strategy sells", "strategy buys", "microstrategy",
                    "saylor", "treasury company", "whale", "liquidation"], 3),
    "GEO":        (["tariff", "sanction", "war", "strike", "opec",
                    "china", "geopolit"], 2),
    "EXCHANGE":   (["binance", "coinbase", "okx", "bybit", "kraken",
                    "delist", "halt", "outage"], 2),
}

NOISE = ["price prediction", "price analysis", "here's why", "top 5",
         "top 10", "how to", "best crypto", "airdrop guide", "review",
         "sponsored", "opinion:", "interview", "podcast"]


def classify(title):
    """Return (category, base_score) or (None, 0) if noise/irrelevant."""
    t = (title or "").lower()
    if any(n in t for n in NOISE):
        return None, 0
    best_cat, best_score = None, 0
    for cat, (kws, score) in TAXONOMY.items():
        if any(k in t for k in kws) and score > best_score:
            best_cat, best_score = cat, score
    return best_cat, best_score


# ----------------------------------------------------------------------
# STAGE 2 — CLAUDE HAIKU SCORING (batched). Shared by RSS + WS + Telegram feeds.
# ----------------------------------------------------------------------
def llm_score(headlines):
    """headlines: list of (id, category, title). Returns {id: score_dict}.
    score_dict keys: sentiment (-5..5), impact (1..5), assets, rationale.
    Returns {} on any error or if the API key is not configured."""
    if not headlines:
        return {}
    if not have(ANTHROPIC_KEY):
        print("[llm skipped — no ANTHROPIC_API_KEY]")
        return {}
    items = "\n".join(f'{i+1}. [{c}] {t}' for i, (_, c, t) in enumerate(headlines))
    prompt = f"""You are a crypto/macro trading desk analyst. Score each headline.

Headlines:
{items}

Respond ONLY with a JSON array, one object per headline, same order:
[{{"n": 1, "sentiment": -5 to 5 (negative=bearish for BTC/crypto),
"impact": 1 to 5 (5=market-wide repricing, 1=ignorable),
"assets": "BTC" or "BTC,ETH,alts" etc,
"rationale": "one short sentence"}}]

Scoring guide: Fed policy surprises / CPI shocks / major regulation / big hacks / \
exchange listings of a new perp = 4-5. Routine commentary / minor project news = 1-2. \
Old or speculative news = lower impact."""
    try:
        import json
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": HAIKU_MODEL, "max_tokens": 1500,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30)
        text = r.json()["content"][0]["text"]
        text = text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(text)
        out = {}
        for s in scores:
            idx = s["n"] - 1
            if 0 <= idx < len(headlines):
                out[headlines[idx][0]] = s
        return out
    except Exception as e:
        print(f"[llm error] {e}")
        return {}


# ----------------------------------------------------------------------
# UNIFIED SIGNAL STORE  (signals.db)
# Every scored real-time item — from RSS, websockets, or Telegram — lands here
# in one normalized shape so the sentiment engine has a single table to read.
# ----------------------------------------------------------------------
def signal_id(source, title, link=""):
    return hashlib.sha256(f"{source}|{title}|{link}".encode()).hexdigest()[:16]


def signal_db():
    con = sqlite3.connect(SIGNAL_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS signals (
        id TEXT PRIMARY KEY,
        ts_utc TEXT,            -- ISO8601 UTC, event arrival time
        source TEXT,            -- 'tree', 'phoenix', 'telegram:<chan>', 'rss', ...
        category TEXT,
        title TEXT,
        link TEXT,
        sentiment INTEGER,      -- -5..5, NULL until scored
        impact INTEGER,         -- 1..5, NULL until scored
        assets TEXT,
        rationale TEXT,
        scored INTEGER DEFAULT 0)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc)")
    con.commit()
    return con


def record_raw(source, title, link="", category=None, base_score=0, con=None):
    """Insert an unscored item. Returns (signal_id, is_new)."""
    close = con is None
    con = con or signal_db()
    sid = signal_id(source, title, link)
    exists = con.execute("SELECT 1 FROM signals WHERE id=?", (sid,)).fetchone()
    if not exists:
        con.execute(
            "INSERT OR IGNORE INTO signals "
            "(id, ts_utc, source, category, title, link) VALUES (?,?,?,?,?,?)",
            (sid, datetime.now(UTC).isoformat(), source,
             category or "NOISE", title, link))
        con.commit()
    if close:
        con.close()
    return sid, (not exists)


def record_score(sid, score, con=None):
    """Attach a Haiku score dict to a stored signal."""
    close = con is None
    con = con or signal_db()
    con.execute(
        "UPDATE signals SET sentiment=?, impact=?, assets=?, rationale=?, "
        "category=COALESCE(NULLIF(category,'NOISE'), category), scored=1 WHERE id=?",
        (score.get("sentiment"), score.get("impact"),
         score.get("assets", "BTC"), score.get("rationale", ""), sid))
    con.commit()
    if close:
        con.close()


# ----------------------------------------------------------------------
# INGEST PIPELINE — the shared path for every real-time text source.
# RSS bot, websockets (feeds_ws.py) and Telegram (telegram_listener.py) all
# funnel through here: Stage 1 rule filter -> store -> Stage 2 Haiku -> alert.
# ----------------------------------------------------------------------
SENT_EMOJI = {-5: "🔴🔴", -4: "🔴🔴", -3: "🔴", -2: "🔴", -1: "🟠",
              0: "⚪", 1: "🟢", 2: "🟢", 3: "🟢", 4: "🟢🟢", 5: "🟢🟢"}

ALERT_THRESHOLD = int(os.environ.get("ALERT_THRESHOLD", "3"))
MAX_LLM_PER_BATCH = int(os.environ.get("MAX_LLM_PER_BATCH", "10"))


def alert(source, category, title, link, score):
    sent = score.get("sentiment", 0)
    impact = score.get("impact", 0)
    fire = "🚨" * max(1, impact - 2)
    src = source.split(":")[0]
    send(f"{fire} <b>[{category}] impact {impact}/5</b> "
         f"<i>via {src}</i>\n"
         f"{SENT_EMOJI.get(sent, '⚪')} sentiment {sent:+d} | "
         f"{score.get('assets', 'BTC')}\n\n"
         f"<b>{title}</b>\n{score.get('rationale', '')}\n\n"
         f"{btc_snapshot(short=True)}"
         + (f"\n<a href='{link}'>source</a>" if link else ""))


def ingest_batch(items, do_score=True, do_alert=True,
                 alert_threshold=None, con=None):
    """The Stage 1 -> Stage 2 -> alert path for a batch of raw items.

    items: iterable of (source, title, link).
    Returns list of dicts for items that were scored & cleared threshold:
        {source, category, title, link, sentiment, impact, assets, rationale}

    Stage 1 (classify) is free and runs on everything; only survivors with
    base_score >= 2 are stored as candidates and sent to Haiku (cost-guarded).
    """
    threshold = ALERT_THRESHOLD if alert_threshold is None else alert_threshold
    close = con is None
    con = con or signal_db()

    candidates = []   # (sid, category, title, link)
    for source, title, link in items:
        cat, base = classify(title)
        if not cat or base < 2:
            continue
        sid, is_new = record_raw(source, title, link, category=cat, con=con)
        if is_new:
            candidates.append((sid, source, cat, title, link))

    if not do_score or not candidates:
        if close:
            con.close()
        return []

    # Cost guard: score at most MAX_LLM_PER_BATCH, highest base_score first.
    batch = candidates[:MAX_LLM_PER_BATCH]
    scores = llm_score([(sid, cat, title) for sid, _, cat, title, _ in batch])

    alerts = []
    for sid, source, cat, title, link in batch:
        s = scores.get(sid)
        if not s:
            continue
        record_score(sid, s, con=con)
        if s.get("impact", 0) >= threshold:
            rec = {"source": source, "category": cat, "title": title,
                   "link": link, "sentiment": s.get("sentiment", 0),
                   "impact": s.get("impact", 0), "assets": s.get("assets", "BTC"),
                   "rationale": s.get("rationale", "")}
            alerts.append(rec)
            if do_alert:
                alert(source, cat, title, link, s)
    if close:
        con.close()
    return alerts


if __name__ == "__main__":
    # Smoke test — no keys required.
    print("shared.py self-test")
    print("  classify('Fed signals rate cut in September') ->", classify("Fed signals rate cut in September"))
    print("  classify('Top 10 altcoins to buy now')        ->", classify("Top 10 altcoins to buy now"))
    print("  btc snapshot:", btc_snapshot(short=True))
    sid, new = record_raw("selftest", "Binance will list NEWCOIN perpetual", "http://x")
    print(f"  record_raw -> id={sid} new={new}")
    record_score(sid, {"sentiment": 3, "impact": 4, "assets": "NEWCOIN", "rationale": "new perp listing"})
    con = signal_db()
    row = con.execute("SELECT source, category, sentiment, impact FROM signals WHERE id=?", (sid,)).fetchone()
    print("  stored+scored row:", row)
    con.close()
