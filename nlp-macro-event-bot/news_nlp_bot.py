"""
NEWS NLP BOT — Real-time market news engine (Bloomberg-lite)
============================================================
Two ingestion paths feeding one pipeline:
  RSS feeds        -> Stage 1 keyword classifier (English-only) -> candidates
  Telegram channels -> (curated already, no keyword gate)        -> candidates
Then: dedupe (SQLite) -> Stage 2 LLM scoring (only for candidates) -> Telegram alert

Stage 1 kills ~95% of RSS noise with an English keyword taxonomy. It does NOT
gate Telegram channels (TELEGRAM_CHANNELS) — those are scraped from the public
t.me/s/<channel> preview (no login), assumed pre-curated, and every new post
becomes a candidate as long as an LLM key is set (needed for translation).

Stage 2 sends surviving items to an LLM (batched, 1 call per cycle) and
returns: sentiment (-5..+5), impact (1..5), assets, rationale, and title_en
(English translation, used for the alert instead of the raw text).
Provider is picked at import time, first available wins:
  1. Groq (GROQ_API_KEY)           — free tier, no credit card, llama-3.3-70b
  2. xAI (XAI_API_KEY)             — paid, grok-4-fast
  3. Anthropic (ANTHROPIC_API_KEY) — paid, Claude Haiku
  4. none — RSS falls back to Stage-1 keyword-tier alerts; Telegram channels
     are skipped entirely (no way to translate without an LLM)

Run:  py news_nlp_bot.py
Deps: py -m pip install requests feedparser apscheduler tzdata beautifulsoup4
Env:  GROQ_API_KEY or XAI_API_KEY or ANTHROPIC_API_KEY, TG_BOT_TOKEN, TG_CHAT_ID
      (same bot as okx_spike_screener.py)
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import feedparser
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Same bot/chat as okx_spike_screener.py (the live OKX Reactive Spike Signal
# bot) so news/sentiment alerts land in one Telegram shell alongside spikes.
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN") or os.environ.get("MACRO_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN")
CHAT_ID   = os.environ.get("TG_CHAT_ID") or os.environ.get("MACRO_BOT_CHAT_ID", "PASTE_YOUR_CHAT_ID")

GROQ_KEY      = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"   # free tier, no card required
XAI_KEY       = os.environ.get("XAI_API_KEY", "")
XAI_MODEL     = "grok-4-fast"               # paid; OpenAI-compatible endpoint
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "PASTE_YOUR_API_KEY")
ANTHROPIC_MODEL = "claude-haiku-4-5"

HAVE_GROQ      = bool(GROQ_KEY)
HAVE_XAI       = bool(XAI_KEY)
HAVE_ANTHROPIC = ANTHROPIC_KEY not in ("PASTE_YOUR_API_KEY", "")
HAVE_LLM       = HAVE_GROQ or HAVE_XAI or HAVE_ANTHROPIC
# Priority: Groq (free) > xAI (paid) > Anthropic (paid) — cheapest first.
LLM_PROVIDER   = ("groq" if HAVE_GROQ else
                   "xai" if HAVE_XAI else
                   "anthropic" if HAVE_ANTHROPIC else "none")

WIB = ZoneInfo("Asia/Jakarta")
# Absolute so the dedup history follows the script rather than the working
# directory — a scheduled run from a different cwd would otherwise create a
# second, empty DB and re-alert everything it had already sent.
DB = os.environ.get("NEWS_DB") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "news_nlp.db")

# Alert gate. impact is 1..5 where 5 = market-wide repricing and 1 = ignorable,
# so a threshold of 3 was alerting on "routine news" — it let through the ETF /
# regulation filler that made the feed unreadable. 4 means "this should move a
# position", which is the only kind of headline worth interrupting a swing trader.
ALERT_THRESHOLD = int(os.environ.get("NEWS_ALERT_IMPACT", "4"))
KEYWORD_ALERT_THRESHOLD = 4          # fallback alert gate when no LLM key
MAX_LLM_HEADLINES_PER_CYCLE = 10     # cost guard
MAX_KEYWORD_ALERTS_PER_CYCLE = 2     # fallback alert cap (per category, highest score)
POLL_SECONDS = 60

# Semantic dedup. The id is sha256(title+link), so the SAME story filed by two
# outlets under two wordings ("BlackRock debuts tokenized access to $311 billion
# of money market funds" vs "BlackRock debuts tokenized share classes for select
# European money market funds with $311 billion in assets") hashes differently
# and both fired. Stage 2 now also returns a story_key — a slug for the
# underlying EVENT — and only the first headline per story_key gets alerted.
STORY_DEDUP_HOURS = int(os.environ.get("NEWS_STORY_DEDUP_HOURS", "12"))
MAX_ALERTS_PER_DAY = int(os.environ.get("NEWS_MAX_ALERTS_PER_DAY", "8"))

# ---------------------------------------------------------------------------
# WHAT IS WORTH AN ALERT (measured on 1379 scored items over 18 days)
# ---------------------------------------------------------------------------
# Two independent filters, because impact and direction are different things and
# only the pair is tradeable:
#
#   CATEGORY — a swing trader on perps is positioned on macro, not on product
#   announcements. Fed / inflation (CPI, PPI, PCE) / jobs move the whole risk
#   complex; ETF flow stories, listings and corporate news do not move it in a
#   direction you can hold for days. ETF_FLOWS is deliberately NOT in this set.
#
#   DIRECTION — 26 of the 242 impact>=4 items scored sentiment 0: genuinely
#   big news with no directional read. Those are the worst kind of alert, since
#   they demand attention and imply no trade. Requiring |sentiment| >= 3 drops
#   them along with the merely-mildly-slanted.
#
# Measured alert rates per day on the same data:
#   impact>=3 (original)                  31.4/day
#   impact>=4                             13.4/day
#   impact>=4 + |sentiment|>=3             7.1/day
#   MACRO + impact>=4 + |sentiment|>=3     1.2/day   <- this configuration
ALERT_CATEGORIES = {c.strip().upper() for c in os.environ.get(
    "NEWS_CATEGORIES", "FED,INFLATION,JOBS,GEO").split(",") if c.strip()}
MIN_ABS_SENTIMENT = int(os.environ.get("NEWS_MIN_ABS_SENTIMENT", "3"))

print(f"[news_nlp_bot] Stage 2 provider: {LLM_PROVIDER}"
      + ("" if HAVE_LLM else " — alerting on Stage-1 keyword score only."))

FEEDS = [
    # Crypto-native
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss.xml",
    "https://decrypt.co/feed",
    # Macro / Fed
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.forexlive.com/feed/news",
    # Google News queries (near-realtime aggregation). These are aimed at the
    # categories that actually clear ALERT_CATEGORIES — there is no point paying
    # to fetch and LLM-score stories the macro filter will discard. The old
    # "bitcoin ETF OR CLARITY act" query was doing exactly that: every item it
    # returned classified as ETF_FLOWS or REGULATION and was then dropped.
    "https://news.google.com/rss/search?q=%22federal+reserve%22+OR+%22kevin+warsh%22&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=CPI+OR+inflation+OR+PPI+OR+%22core+PCE%22&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=FOMC+OR+%22rate+decision%22+OR+%22nonfarm+payrolls%22+OR+%22jobs+report%22&hl=en-US&gl=US&ceid=US:en",
]

# Public Telegram channels, scraped via the unauthenticated t.me/s/<channel>
# web preview (no bot/user login needed for public channels — same trick
# Telegram itself uses for link-preview embeds). Posts may be in any
# language; Stage 2 (if an LLM key is set) returns an English translation
# alongside the score, used for the alert text instead of the raw post.
TELEGRAM_CHANNELS = [
    "CryptoMarketAggregator",
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
    # story_key was added after the table shipped, so migrate in place rather
    # than forcing a rebuild that would drop the dedup history.
    cols = {r[1] for r in con.execute("PRAGMA table_info(news)")}
    if "story_key" not in cols:
        con.execute("ALTER TABLE news ADD COLUMN story_key TEXT")
        print("[news_nlp_bot] migrated: added news.story_key")
    con.execute("CREATE INDEX IF NOT EXISTS ix_news_story ON news(story_key, ts)")
    con.commit()
    return con


def recent_story_keys(con, hours):
    """story_keys already ALERTED inside the window — the semantic dedup set."""
    cutoff = (datetime.now(tz=WIB) - timedelta(hours=hours)).isoformat()
    return {r[0] for r in con.execute(
        "SELECT story_key FROM news WHERE alerted=1 AND story_key IS NOT NULL AND ts > ?",
        (cutoff,)) if r[0]}


def alerts_today(con):
    today = datetime.now(tz=WIB).date().isoformat()
    return con.execute("SELECT COUNT(*) FROM news WHERE alerted=1 AND ts >= ?",
                       (today,)).fetchone()[0]


def news_id(title, link):
    return hashlib.sha256((title + link).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# TELEGRAM CHANNEL SCRAPE (public channels, no login required)
# ----------------------------------------------------------------------
def fetch_telegram_channel(channel, limit=15):
    """Scrape the public t.me/s/<channel> preview page. Returns list of
    (text, link) for the most recent posts. No auth, no session file —
    works statelessly in CI. Fragile against Telegram markup changes since
    it's HTML scraping, not an API; fails soft (empty list) on error."""
    try:
        r = requests.get(f"https://t.me/s/{channel}", timeout=15,
                          headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[telegram scrape error] {channel}: {e}")
        return []

    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"[telegram parse error] {channel}: {e}")
        return []

    posts = []
    for msg in soup.select("div.tgme_widget_message"):
        post_id = msg.get("data-post")
        text_div = msg.select_one("div.tgme_widget_message_text")
        if not post_id or not text_div:
            continue
        text = text_div.get_text("\n", strip=True)
        if not text:
            continue
        posts.append((text, f"https://t.me/{post_id}"))
    return posts[-limit:]


# ----------------------------------------------------------------------
# STAGE 2 — LLM SCORING (batched, provider-agnostic)
# ----------------------------------------------------------------------
def _score_prompt(headlines):
    items = "\n".join(f'{i+1}. [{c}] {t}' for i, (_, c, t) in enumerate(headlines))
    return f"""You are a crypto/macro trading desk analyst. Score each headline.

Headlines:
{items}

Respond ONLY with a JSON array, one object per headline, same order:
[{{"n": 1, "sentiment": -5 to 5 (negative=bearish for BTC/crypto),
"impact": 1 to 5 (5=market-wide repricing, 1=ignorable),
"assets": "BTC" or "BTC,ETH,alts" etc,
"rationale": "one short sentence",
"story_key": "short-lowercase-slug of the underlying EVENT",
"title_en": "the headline translated to English (verbatim if already English)"}}]

Scoring guide — the reader is a swing trader holding perps for days, so score by
whether this changes a POSITION, not by whether it is interesting:
  5 = market-wide repricing (FOMC/CPI surprise, major exchange insolvency or hack,
      landmark law actually passing)
  4 = a real directional catalyst a trader should act on
  3 = context worth knowing, not worth an interruption
  1-2 = routine commentary, product launches, partnerships, price-prediction
      pieces, "X could hit $Y", recycled or speculative news
Anything phrased as a question, a forecast, or an opinion column is at most 2.

story_key rule: two headlines describing the SAME underlying event MUST get the
IDENTICAL story_key, even when the wording, outlet, or numbers differ. Use the
actor + action, e.g. "blackrock-tokenized-mmf-europe", "clarity-act-senate-vote".
Do not put the outlet name or the headline's exact phrasing in the slug."""


def _parse_score_response(text, headlines):
    text = text.replace("```json", "").replace("```", "").strip()
    scores = json.loads(text)
    out = {}
    for s in scores:
        idx = s["n"] - 1
        if 0 <= idx < len(headlines):
            out[headlines[idx][0]] = s
    return out


def llm_score_groq(headlines):
    """Free tier, no credit card. OpenAI-compatible chat completions API."""
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}",
                 "content-type": "application/json"},
        json={"model": GROQ_MODEL, "max_tokens": 1500, "temperature": 0,
              "messages": [{"role": "user", "content": _score_prompt(headlines)}]},
        timeout=30)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return _parse_score_response(text, headlines)


def llm_score_xai(headlines):
    """xAI Grok — OpenAI-compatible chat completions API. Paid."""
    r = requests.post("https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {XAI_KEY}",
                 "content-type": "application/json"},
        json={"model": XAI_MODEL, "max_tokens": 1500, "temperature": 0,
              "messages": [{"role": "user", "content": _score_prompt(headlines)}]},
        timeout=30)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return _parse_score_response(text, headlines)


def llm_score_anthropic(headlines):
    r = requests.post("https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": ANTHROPIC_MODEL, "max_tokens": 1500,
              "messages": [{"role": "user", "content": _score_prompt(headlines)}]},
        timeout=30)
    r.raise_for_status()
    text = r.json()["content"][0]["text"]
    return _parse_score_response(text, headlines)


def llm_score(headlines):
    """headlines: list of (id, category, title). Returns {id: dict}."""
    if not headlines or not HAVE_LLM:
        return {}
    try:
        if HAVE_GROQ:
            return llm_score_groq(headlines)
        if HAVE_XAI:
            return llm_score_xai(headlines)
        return llm_score_anthropic(headlines)
    except Exception as e:
        print(f"[llm error, provider={LLM_PROVIDER}] {e}")
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


def bias_of(sent):
    """Direction only. The old 'power N/10' was |sentiment| + impact, so an
    impact-3 item with mild sentiment scored 6/10 and read as conviction when it
    was really the alert floor wearing a costume. Impact and sentiment are now
    shown as themselves."""
    if sent > 0:
        return "LONG", "📈"
    if sent < 0:
        return "SHORT", "📉"
    return "NEUTRAL", "⚪"


def alert(row):
    cat, title, link, sent, impact, assets, rationale = row
    bias, emoji = bias_of(sent)
    # Direction first: this only fires on macro with a strong directional read,
    # so the bias IS the message and the headline is the supporting evidence.
    strength = "STRONG" if abs(sent) >= 4 else "CLEAR"
    send(f"{emoji} <b>MACRO {bias}</b> — {strength} bias\n"
         f"[{cat}] impact {impact}/5 · sentiment {sent:+d} · {assets}\n\n"
         f"<b>{title}</b>\n{rationale}\n\n"
         f"{btc_price()}\n<a href='{link}'>source</a>")


def alert_keyword_tier(cat, title, link, score):
    """Fallback when no LLM key (Groq/Anthropic) is set: no sentiment/impact/
    rationale, just the Stage-1 keyword hit — still better than silence."""
    send(f"⚠️ <b>[{cat}] keyword score {score}</b> (no LLM key — unscored)\n\n"
         f"<b>{title}</b>\n\n{btc_price()}\n<a href='{link}'>source</a>")


# ----------------------------------------------------------------------
# MAIN CYCLE
# ----------------------------------------------------------------------
def cycle():
    con = db_init()
    candidates = []   # (id, category, title, link, base_score)

    # First-ever run: the dedup table is empty, so every current headline
    # across all feeds would look "new" and fire at once. Seed silently —
    # insert everything as seen, alert on nothing this cycle — then alert
    # normally from the next cycle on as genuinely new items appear.
    is_seed_run = con.execute("SELECT COUNT(*) FROM news").fetchone()[0] == 0
    if is_seed_run:
        print("[news_nlp_bot] Empty DB — seeding dedup table silently "
              "(no alerts this cycle) to avoid an RSS-backfill flood.")

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

    # Telegram channels aren't gated through the Stage-1 keyword classifier:
    # its taxonomy is English-only substring matching ("hack", "fed ", ...),
    # so non-English posts (e.g. Chinese aggregator channels) would almost
    # never match and would silently produce zero candidates. These channels
    # are themselves already curated, so every new post is a candidate —
    # but only when an LLM key is set, since untranslated non-English text
    # alerted as "unscored keyword tier" wouldn't be useful.
    if HAVE_LLM:
        for channel in TELEGRAM_CHANNELS:
            source = f"telegram:{channel}"
            for text, link in fetch_telegram_channel(channel):
                title = text.splitlines()[0][:300]  # DB/alert display title
                nid = news_id(title, link)
                if con.execute("SELECT 1 FROM news WHERE id=?", (nid,)).fetchone():
                    continue
                con.execute(
                    "INSERT OR IGNORE INTO news (id, ts, source, title, link, category, base_score) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (nid, datetime.now(tz=WIB).isoformat(), source, title, link,
                     "TELEGRAM", 3))
                candidates.append((nid, "TELEGRAM", text, link, 3))
    con.commit()
    print(f"[news_nlp_bot] {len(candidates)} new candidate(s) this cycle "
          f"(seed_run={is_seed_run}, llm_provider={LLM_PROVIDER})")

    if is_seed_run:
        con.commit()
        con.close()
        return

    # Stage 2: score the top candidates (cost-guarded) — skipped entirely if
    # no LLM key, so we don't burn a network call to a 401 every cycle.
    candidates.sort(key=lambda x: -x[4])
    batch = candidates[:MAX_LLM_HEADLINES_PER_CYCLE]

    fired = 0
    if HAVE_LLM:
        scores = llm_score([(c[0], c[1], c[2]) for c in batch])
        # Stories already alerted in the dedup window, plus the ones fired in
        # THIS batch — two wordings of one event usually arrive in the same
        # cycle, so an in-memory set is what actually catches the screenshot case.
        seen_stories = recent_story_keys(con, STORY_DEDUP_HOURS)
        budget = MAX_ALERTS_PER_DAY - alerts_today(con)
        for nid, cat, title, link, _ in batch:
            s = scores.get(nid)
            if not s:
                continue
            impact = int(s.get("impact", 0))
            skey = (s.get("story_key") or "").strip().lower() or None
            con.execute("UPDATE news SET sentiment=?, impact=?, assets=?, rationale=?, "
                        "story_key=? WHERE id=?",
                        (s["sentiment"], impact, s.get("assets", "BTC"),
                         s.get("rationale", ""), skey, nid))
            if impact < ALERT_THRESHOLD:
                continue
            if ALERT_CATEGORIES and cat.upper() not in ALERT_CATEGORIES:
                print(f"[news_nlp_bot] skipped [{cat}] (not a macro category): "
                      f"{title[:60]}")
                continue
            if abs(int(s.get("sentiment", 0))) < MIN_ABS_SENTIMENT:
                # High impact but no directional read — nothing to act on.
                print(f"[news_nlp_bot] skipped (sentiment {s.get('sentiment')} "
                      f"too weak to trade): {title[:60]}")
                continue
            if skey and skey in seen_stories:
                print(f"[news_nlp_bot] deduped '{title[:60]}' -> story '{skey}'")
                continue
            if budget <= 0:
                print(f"[news_nlp_bot] daily cap {MAX_ALERTS_PER_DAY} reached — "
                      f"suppressing '{title[:60]}'")
                continue
            alert((cat, s.get("title_en") or title, link, s["sentiment"],
                   impact, s.get("assets", "BTC"), s.get("rationale", "")))
            con.execute("UPDATE news SET alerted=1 WHERE id=?", (nid,))
            if skey:
                seen_stories.add(skey)
            budget -= 1
            fired += 1
    else:
        # No sentiment/impact to write — market_risk_state() only reads rows
        # with impact set, so these are honestly excluded from that aggregate.
        # Different feeds often carry the same story under different
        # headlines (e.g. "US agencies miss..." vs "US regulators miss...");
        # without Claude to notice that, cap to one alert per category
        # (highest keyword score) and MAX_KEYWORD_ALERTS_PER_CYCLE overall.
        seen_categories = set()
        fired = 0
        for nid, cat, title, link, score in batch:
            if score < KEYWORD_ALERT_THRESHOLD or cat in seen_categories:
                continue
            if fired >= MAX_KEYWORD_ALERTS_PER_CYCLE:
                break
            alert_keyword_tier(cat, title, link, score)
            con.execute("UPDATE news SET alerted=1 WHERE id=?", (nid,))
            seen_categories.add(cat)
            fired += 1
    con.commit()
    con.close()
    print(f"[news_nlp_bot] {fired} alert(s) sent this cycle")


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
