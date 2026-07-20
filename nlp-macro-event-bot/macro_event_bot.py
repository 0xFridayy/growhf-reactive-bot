"""
MACRO EVENT BOT — Crypto macro calendar + blackout alerts via Telegram
======================================================================
What it does:
  1. Daily digest 07:30 WIB  -> today's + next 7 days macro events + BTC snapshot
  2. Pre-event warning       -> T-60min before every Tier 1 event
  3. Blackout checker        -> is_blackout() importable by your Perps engine
  4. Commands                -> /events /today /blackout /btc

Run:  py macro_event_bot.py
Deps: py -m pip install requests apscheduler tzdata

Event dates verified against BLS official schedule (bls.gov) + Fed calendar,
fetched 2026-07-07. FOMC dates beyond July are the standard published
calendar — reconfirm on federalreserve.gov if a meeting gets moved.
PCE dates (BEA) are approximate — edit if needed.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from apscheduler.schedulers.background import BackgroundScheduler

# ----------------------------------------------------------------------
# CONFIG — fill these in (or set as environment variables)
# ----------------------------------------------------------------------
# Same bot/chat as okx_spike_screener.py (the live OKX Reactive Spike Signal
# bot) so macro/event alerts land in one Telegram shell alongside spikes.
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN") or os.environ.get("MACRO_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN")
CHAT_ID   = os.environ.get("TG_CHAT_ID") or os.environ.get("MACRO_BOT_CHAT_ID", "PASTE_YOUR_CHAT_ID")

ET  = ZoneInfo("America/New_York")   # release times are US Eastern
WIB = ZoneInfo("Asia/Jakarta")

STATE_FILE = "macro_bot_state.json"  # remembers sent alerts + telegram offset

# Blackout window per event type: (hours_before, hours_after)
BLACKOUT = {
    "FOMC":    (12, 6),
    "CPI":     (12, 4),
    "NFP":     (12, 4),
    "MINUTES": (3, 3),
    "PPI":     (2, 2),
    "PCE":     (3, 3),
}

# ----------------------------------------------------------------------
# EVENT CALENDAR — H2 2026 (all times US Eastern, auto-converted to WIB)
# type, date, time_ET, label, tier
# ----------------------------------------------------------------------
EVENTS = [
    # --- FOMC minutes (2:00 PM ET, 3 weeks after each meeting) ---
    ("MINUTES", "2026-07-08", "14:00", "FOMC Minutes (June mtg — Warsh #1)", 1),
    ("MINUTES", "2026-08-19", "14:00", "FOMC Minutes (July mtg)",            2),
    ("MINUTES", "2026-10-07", "14:00", "FOMC Minutes (Sep mtg)",             2),
    ("MINUTES", "2026-11-18", "14:00", "FOMC Minutes (Oct mtg)",             2),
    ("MINUTES", "2026-12-30", "14:00", "FOMC Minutes (Dec mtg)",             2),

    # --- FOMC rate decisions (statement 2:00 PM ET, presser 2:30 PM) ---
    ("FOMC", "2026-07-29", "14:00", "FOMC Rate Decision + Presser", 1),
    ("FOMC", "2026-09-16", "14:00", "FOMC Rate Decision + Presser (+ projections)", 1),
    ("FOMC", "2026-10-28", "14:00", "FOMC Rate Decision + Presser", 1),
    ("FOMC", "2026-12-09", "14:00", "FOMC Rate Decision + Presser (+ projections)", 1),

    # --- CPI (8:30 AM ET — BLS verified) ---
    ("CPI", "2026-07-14", "08:30", "US CPI (June)",      1),
    ("CPI", "2026-08-12", "08:30", "US CPI (July)",      1),
    ("CPI", "2026-09-11", "08:30", "US CPI (August)",    1),
    ("CPI", "2026-10-14", "08:30", "US CPI (September)", 1),
    ("CPI", "2026-11-10", "08:30", "US CPI (October)",   1),
    ("CPI", "2026-12-10", "08:30", "US CPI (November)",  1),

    # --- NFP / Employment Situation (8:30 AM ET — BLS verified) ---
    ("NFP", "2026-08-07", "08:30", "US NFP / Jobs Report (July)",      1),
    ("NFP", "2026-09-04", "08:30", "US NFP / Jobs Report (August)",    1),
    ("NFP", "2026-10-02", "08:30", "US NFP / Jobs Report (September)", 1),
    ("NFP", "2026-11-06", "08:30", "US NFP / Jobs Report (October)",   1),
    ("NFP", "2026-12-04", "08:30", "US NFP / Jobs Report (November)",  1),

    # --- PPI (8:30 AM ET — BLS verified) ---
    ("PPI", "2026-07-15", "08:30", "US PPI (June)",      2),
    ("PPI", "2026-08-13", "08:30", "US PPI (July)",      2),
    ("PPI", "2026-09-10", "08:30", "US PPI (August)",    2),
    ("PPI", "2026-10-15", "08:30", "US PPI (September)", 2),
    ("PPI", "2026-11-13", "08:30", "US PPI (October)",   2),
    ("PPI", "2026-12-15", "08:30", "US PPI (November)",  2),

    # --- PCE (BEA, ~8:30 AM ET — APPROXIMATE, verify on bea.gov) ---
    ("PCE", "2026-07-31", "08:30", "US PCE Inflation (June) [verify date]",      2),
    ("PCE", "2026-08-28", "08:30", "US PCE Inflation (July) [verify date]",      2),
    ("PCE", "2026-09-25", "08:30", "US PCE Inflation (August) [verify date]",    2),
    ("PCE", "2026-10-30", "08:30", "US PCE Inflation (September) [verify date]", 2),
    ("PCE", "2026-11-25", "08:30", "US PCE Inflation (October) [verify date]",   2),
    ("PCE", "2026-12-23", "08:30", "US PCE Inflation (November) [verify date]",  2),
]


def parse_events():
    """Return sorted list of dicts with tz-aware datetimes (ET + WIB)."""
    out = []
    for etype, d, t, label, tier in EVENTS:
        dt_et = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        out.append({
            "type": etype, "label": label, "tier": tier,
            "dt_et": dt_et, "dt_wib": dt_et.astimezone(WIB),
        })
    return sorted(out, key=lambda e: e["dt_et"])


ALL_EVENTS = parse_events()

# ----------------------------------------------------------------------
# STATE (sent alerts + telegram update offset)
# ----------------------------------------------------------------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"alerted": [], "offset": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


STATE = load_state()

# ----------------------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------------------
API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def send(text, chat_id=None):
    try:
        requests.post(f"{API}/sendMessage", json={
            "chat_id": chat_id or CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
    except requests.RequestException as e:
        print(f"[send error] {e}")


# ----------------------------------------------------------------------
# BINANCE SNAPSHOT (public endpoints, no key needed)
# ----------------------------------------------------------------------
def btc_snapshot():
    try:
        t = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                         params={"symbol": "BTCUSDT"}, timeout=10).json()
        p = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": "BTCUSDT"}, timeout=10).json()
        price = float(t["lastPrice"])
        chg = float(t["priceChangePercent"])
        funding = float(p["lastFundingRate"]) * 100
        arrow = "🟢" if chg >= 0 else "🔴"
        return (f"{arrow} <b>BTC</b> ${price:,.0f}  ({chg:+.2f}% 24h)\n"
                f"⚙️ Funding: {funding:+.4f}%")
    except Exception as e:
        return f"BTC snapshot unavailable ({e})"


# ----------------------------------------------------------------------
# CORE LOGIC
# ----------------------------------------------------------------------
def is_blackout(now=None):
    """Returns (True, event_label) if inside a blackout window, else (False, None).
    Import this from your Perps engine:
        from macro_event_bot import is_blackout
        blocked, why = is_blackout()
    """
    now = now or datetime.now(tz=WIB)
    for e in ALL_EVENTS:
        before, after = BLACKOUT.get(e["type"], (2, 2))
        start = e["dt_wib"] - timedelta(hours=before)
        end   = e["dt_wib"] + timedelta(hours=after)
        if start <= now <= end:
            return True, e["label"]
    return False, None


def fmt_event(e, now):
    dt = e["dt_wib"]
    delta = dt - now
    days = delta.days
    hrs = delta.seconds // 3600
    eta = f"in {days}d {hrs}h" if days > 0 else f"in {hrs}h" if delta.total_seconds() > 0 else "NOW"
    tier = "🔴" if e["tier"] == 1 else "🟡"
    return f"{tier} {dt:%a %d %b %H:%M} WIB — {e['label']} ({eta})"


def upcoming(n=10, within_days=None, now=None):
    now = now or datetime.now(tz=WIB)
    evs = [e for e in ALL_EVENTS if e["dt_wib"] > now]
    if within_days:
        evs = [e for e in evs if e["dt_wib"] <= now + timedelta(days=within_days)]
    return evs[:n]


# ----------------------------------------------------------------------
# SCHEDULED JOBS
# ----------------------------------------------------------------------
def daily_digest():
    now = datetime.now(tz=WIB)
    evs = upcoming(within_days=7, n=15)
    lines = [f"📅 <b>MACRO DIGEST</b> — {now:%a %d %b %Y}", ""]
    lines.append(btc_snapshot())
    lines.append("")
    blocked, why = is_blackout(now)
    if blocked:
        lines.append(f"⛔ <b>BLACKOUT ACTIVE</b>: {why}")
        lines.append("")
    if evs:
        lines.append("<b>Next 7 days:</b>")
        lines += [fmt_event(e, now) for e in evs]
    else:
        lines.append("No macro events in the next 7 days. Clean tape.")
    send("\n".join(lines))


def pre_event_check():
    """Runs every 10 min. Fires a warning 50-70 min before Tier 1 events."""
    now = datetime.now(tz=WIB)
    for e in ALL_EVENTS:
        if e["tier"] != 1:
            continue
        key = f"{e['type']}_{e['dt_et']:%Y%m%d%H%M}"
        mins_to = (e["dt_wib"] - now).total_seconds() / 60
        if 50 <= mins_to <= 70 and key not in STATE["alerted"]:
            before, after = BLACKOUT[e["type"]]
            send(f"⚠️ <b>T-60min: {e['label']}</b>\n"
                 f"🕐 {e['dt_wib']:%H:%M} WIB\n"
                 f"⛔ Blackout window: -{before}h / +{after}h\n"
                 f"Reduce leverage / widen stops / no fresh entries.\n\n"
                 f"{btc_snapshot()}")
            STATE["alerted"].append(key)
            save_state(STATE)


# ----------------------------------------------------------------------
# COMMAND POLLING  (/events /today /blackout /btc)
# ----------------------------------------------------------------------
def handle_command(text, chat_id):
    now = datetime.now(tz=WIB)
    cmd = text.strip().lower().split("@")[0]
    if cmd == "/events":
        evs = upcoming(n=10)
        send("📅 <b>Next 10 events:</b>\n" +
             "\n".join(fmt_event(e, now) for e in evs), chat_id)
    elif cmd == "/today":
        evs = [e for e in upcoming(n=20, within_days=1)
               if e["dt_wib"].date() == now.date()]
        send("📅 <b>Today:</b>\n" + ("\n".join(fmt_event(e, now) for e in evs)
             if evs else "No events today. Clean tape."), chat_id)
    elif cmd == "/blackout":
        blocked, why = is_blackout(now)
        send(f"⛔ BLACKOUT: {why}" if blocked else "✅ No blackout. Trade normal.", chat_id)
    elif cmd == "/btc":
        send(btc_snapshot(), chat_id)


def poll_commands():
    try:
        r = requests.get(f"{API}/getUpdates", params={
            "offset": STATE["offset"] + 1, "timeout": 0}, timeout=15).json()
        for upd in r.get("result", []):
            STATE["offset"] = upd["update_id"]
            msg = upd.get("message") or {}
            text = msg.get("text", "")
            if text.startswith("/"):
                handle_command(text, msg["chat"]["id"])
        save_state(STATE)
    except requests.RequestException as e:
        print(f"[poll error] {e}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if "--once" in sys.argv:
        # Single pass then exit — for schedulers that can't host a
        # persistent process (e.g. GitHub Actions cron).
        now = datetime.now(tz=WIB)
        if now.hour == 7 and 25 <= now.minute <= 34 and STATE.get("last_digest") != now.strftime("%Y-%m-%d"):
            daily_digest()
            STATE["last_digest"] = now.strftime("%Y-%m-%d")
            save_state(STATE)
        pre_event_check()
        poll_commands()
        raise SystemExit

    sched = BackgroundScheduler(timezone="Asia/Jakarta")
    sched.add_job(daily_digest, "cron", hour=7, minute=30)
    sched.add_job(pre_event_check, "interval", minutes=10)
    sched.add_job(poll_commands, "interval", seconds=5)
    sched.start()

    print("Macro Event Bot running. Ctrl+C to stop.")
    send("🤖 Macro Event Bot online.\n\n" +
         "\n".join(fmt_event(e, datetime.now(tz=WIB)) for e in upcoming(n=5)))
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
