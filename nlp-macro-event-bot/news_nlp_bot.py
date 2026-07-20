"""
NEWS NLP BOT — Real-time market news engine (Bloomberg-lite)
============================================================
Pipeline (runs every 60s):
  RSS feeds -> dedupe (SQLite) -> Stage 1 rule classifier (free, instant)
            -> Stage 2 Claude scoring (only for candidates) -> Telegram alert

Stage 1 kills ~95% of noise with keyword taxonomy.
Stage 2 sends surviving headlines to Claude (batched, 1 call per cycle)
and returns: sentiment (-5..+5), impact (1..5), assets, rationale.

Run:  py news_nlp_bot.py
Deps: py -m pip install requests feedparser apscheduler tzdata
Env:  ANTHROPIC_API_KEY, TG_BOT_TOKEN, TG_CHAT_ID (same bot as okx_spike_screener.py)
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import feedparser
import requests
from apscheduler.schedulers.background import BackgroundScheduler

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Same bot/chat as okx_spike_screener.py (the live OKX Reactive Spike Signal
# bot) so news/sentiment alerts land in one Telegram shell alongside spikes.
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN") or os.environ.get("MACRO_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN")
CHAT_ID   = os.environ.get("TG_CHAT_ID") or os.environ.get("MACRO_BOT_CHAT_ID", "PASTE_YOUR_CHAT_ID")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "PASTE_YOUR_API_KEY")
MODEL = "claude-haiku-4-5"          # cheap + fast; upgrade to sonnet if needed

WIB = ZoneInfo("Asia/Jakarta")
DB  = "news_nlp.db"

ALERT_THRESHOLD = 3                  # send Telegram alert if impact >= this
MAX_LLM_HEADLINES_PER_CYCLE = 10     # cost guard
POLL_SECONDS = 60

FEEDS = [
    # Crypto-native
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss.xml",
    "https://decrypt.co/feed",
    # Macro / Fed
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.forexlive.com/feed/news",
    # Google News queries (near-realtime aggregation)
    "https://news.google.com/rss/search?q=%22federal+reserve%22+OR+%22kevin+warsh%22&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=bitcoin+ETF+OR+%22CLARITY+act%22+OR+%22strategic+bitcoin+reserve%22&hl=en-US&gl=US&ceid=US:en",
]

# ----------------------------------------------------------------------
# STAGE 1 — RULE-BASED CLASSIFIER
# The taxonomy now lives in shared.py (single source of truth), so the RSS
# bot, the websocket feeds, and the Telegram listener all classify identically.
# ----------------------------------------------------------------------
from shared import TAXONOMY, NOISE, classify  # noqa: E402  (canonical Stage-1)


# ----------------------------------------------------------------------
# STORAGE
# ----------------------------------------------------------------------
def db_init():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS news (
        id TEXT PRIMARY KEY, ts TEXT, source TEXT, title TEXT, link TEXT,
        category TEXT, base_score INTEGER,
        sentiment INTEGER, impact INTEGER, assets TEXT, rationale TEXT,
        alerted INTEGER DEFAULT 0)""")
    con.commit()
    return con


def news_id(title, link):
    return hashlib.sha256((title + link).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# STAGE 2 — CLAUDE SCORING (batched)
# ----------------------------------------------------------------------
def llm_score(headlines):
    """headlines: list of (id, category, title). Returns {id: dict}."""
    if not headlines:
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

Scoring guide: Fed policy surprises / CPI shocks / major regulation / big hacks = 4-5.
Routine commentary / minor project news = 1-2. Old or speculative news = lower impact."""
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": MODEL, "max_tokens": 1500,
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
# TELEGRAM + BTC SNAPSHOT
# ----------------------------------------------------------------------
API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def send(text):
    try:
        requests.post(f"{API}/sendMessage", json={
            "chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": True}, timeout=15)
    except requests.RequestException as e:
        print(f"[send error] {e}")


def btc_price():
    try:
        t = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                         params={"symbol": "BTCUSDT"}, timeout=10).json()
        return f"BTC ${float(t['lastPrice']):,.0f} ({float(t['priceChangePercent']):+.2f}%)"
    except Exception:
        return ""


SENT_EMOJI = {-5: "🔴🔴", -4: "🔴🔴", -3: "🔴", -2: "🔴", -1: "🟠",
              0: "⚪", 1: "🟢", 2: "🟢", 3: "🟢", 4: "🟢🟢", 5: "🟢🟢"}


def alert(row):
    cat, title, link, sent, impact, assets, rationale = row
    fire = "🚨" * max(1, impact - 2)
    send(f"{fire} <b>[{cat}] impact {impact}/5</b>\n"
         f"{SENT_EMOJI.get(sent, '⚪')} sentiment {sent:+d} | {assets}\n\n"
         f"<b>{title}</b>\n{rationale}\n\n"
         f"{btc_price()}\n<a href='{link}'>source</a>")


# ----------------------------------------------------------------------
# MAIN CYCLE
# ----------------------------------------------------------------------
def cycle():
    con = db_init()
    candidates = []   # (id, category, title, link, base_score)

    for feed_url in FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"[feed error] {feed_url}: {e}")
            continue
        for entry in feed.entries[:20]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            if not title:
                continue
            nid = news_id(title, link)
            if con.execute("SELECT 1 FROM news WHERE id=?", (nid,)).fetchone():
                continue
            cat, score = classify(title)
            con.execute(
                "INSERT OR IGNORE INTO news (id, ts, source, title, link, category, base_score) "
                "VALUES (?,?,?,?,?,?,?)",
                (nid, datetime.now(tz=WIB).isoformat(), feed_url, title, link,
                 cat or "NOISE", score))
            if cat and score >= 2:
                candidates.append((nid, cat, title, link, score))
    con.commit()

    # Stage 2: score the top candidates (cost-guarded)
    candidates.sort(key=lambda x: -x[4])
    batch = candidates[:MAX_LLM_HEADLINES_PER_CYCLE]
    scores = llm_score([(c[0], c[1], c[2]) for c in batch])

    for nid, cat, title, link, _ in batch:
        s = scores.get(nid)
        if not s:
            continue
        con.execute("UPDATE news SET sentiment=?, impact=?, assets=?, rationale=? WHERE id=?",
                    (s["sentiment"], s["impact"], s.get("assets", "BTC"),
                     s.get("rationale", ""), nid))
        if s["impact"] >= ALERT_THRESHOLD:
            alert((cat, title, link, s["sentiment"], s["impact"],
                   s.get("assets", "BTC"), s.get("rationale", "")))
            con.execute("UPDATE news SET alerted=1 WHERE id=?", (nid,))
    con.commit()
    con.close()


def market_risk_state(minutes=60):
    """For your Perps engine: aggregate signed impact of recent scored news.
    Returns (risk_score, n_items). risk_score < -6 => strong bearish newsflow.
        from news_nlp_bot import market_risk_state
    """
    con = db_init()
    rows = con.execute(
        "SELECT sentiment, impact FROM news WHERE impact IS NOT NULL "
        "AND ts > datetime('now', ?)", (f"-{minutes} minutes",)).fetchall()
    con.close()
    score = sum(s * i for s, i in rows if s is not None)
    return score, len(rows)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if "--once" in sys.argv:
        # Single pass then exit — for schedulers that can't host a
        # persistent process (e.g. GitHub Actions cron).
        cycle()
        raise SystemExit

    print("News NLP Bot running. Ctrl+C to stop.")
    send("📡 News NLP Bot online — realtime feed monitoring active.")
    sched = BackgroundScheduler(timezone="Asia/Jakarta")
    sched.add_job(cycle, "interval", seconds=POLL_SECONDS,
                  max_instances=1, coalesce=True)
    sched.start()
    cycle()  # run immediately on start
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
