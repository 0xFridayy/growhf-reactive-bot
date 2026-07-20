"""
SENTIMENT ENGINE — discrete scored events -> continuous ML feature (step 3)
===========================================================================
Consumes every scored item (RSS via news_nlp.db, plus websocket/Telegram via
signals.db), applies an EXPONENTIAL TIME-DECAY, and maintains rolling
aggregation windows. The output is what the Wasserstein K-Means regime engine
reads on its bar cadence.

Core transform per item:   weight = impact * exp(-Δt / τ)
                           contribution = sentiment * weight
τ (tau) is the decay half-life-ish constant, tuned per horizon:
  scalp  τ=5min, swing τ=1h, macro τ=6h.

Feature vector (per horizon), written to signals.db table `sentiment_features`:
  decayed_impact     Σ impact·exp(-Δt/τ)                 (attention / intensity)
  decayed_sentiment  Σ sentiment·impact·exp(-Δt/τ)       (signed pressure)
  net_tone           decayed_sentiment / decayed_impact  (bounded -5..5)
  max_impact         strongest single recent event
  n_events           count in the raw window
  top_category       category carrying the most decayed impact

Public API for the regime engine:
    from sentiment_engine import sentiment_state
    feat = sentiment_state("swing")     # dict of features
    if feat["net_tone"] < -2 and feat["decayed_impact"] > 8:
        ...  # heavy bearish newsflow -> cut size / block longs

Run `py sentiment_engine.py` to (a) ingest, (b) print the current state, and
(c) run a self-test on synthetic data (no external services needed).
"""

import argparse
import math
import sqlite3
from datetime import datetime, timedelta

from shared import UTC, SIGNAL_DB, NEWS_DB, WIB

# Horizon -> decay constant tau (seconds) and raw lookback window (seconds).
HORIZONS = {
    "scalp": {"tau": 5 * 60,      "window": 30 * 60},
    "swing": {"tau": 60 * 60,     "window": 6 * 3600},
    "macro": {"tau": 6 * 3600,    "window": 48 * 3600},
}


# ----------------------------------------------------------------------
# INGEST — normalize scored rows from all sources into signals.db.
# news_nlp.db (RSS bot) stores ts as WIB isoformat; signals.db stores UTC.
# We copy RSS rows in so the engine reads ONE table.
# ----------------------------------------------------------------------
def _ensure_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS signals (
        id TEXT PRIMARY KEY, ts_utc TEXT, source TEXT, category TEXT,
        title TEXT, link TEXT, sentiment INTEGER, impact INTEGER,
        assets TEXT, rationale TEXT, scored INTEGER DEFAULT 0)""")
    con.execute("""CREATE TABLE IF NOT EXISTS sentiment_features (
        ts_utc TEXT, horizon TEXT,
        decayed_impact REAL, decayed_sentiment REAL, net_tone REAL,
        max_impact INTEGER, n_events INTEGER, top_category TEXT,
        PRIMARY KEY (ts_utc, horizon))""")
    con.commit()


def ingest_rss(signal_con):
    """Pull scored rows out of news_nlp.db into the unified signals table.
    Idempotent (INSERT OR IGNORE on the RSS row id). Safe if news_nlp.db
    doesn't exist yet."""
    try:
        rss = sqlite3.connect(f"file:{NEWS_DB}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return 0
    try:
        rows = rss.execute(
            "SELECT id, ts, category, title, link, sentiment, impact, assets, rationale "
            "FROM news WHERE impact IS NOT NULL").fetchall()
    except sqlite3.OperationalError:
        rss.close()
        return 0
    rss.close()

    added = 0
    for rid, ts, cat, title, link, sent, impact, assets, rationale in rows:
        # news_nlp.db ts is WIB isoformat -> convert to UTC iso.
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=WIB)
            ts_utc = dt.astimezone(UTC).isoformat()
        except (ValueError, TypeError):
            ts_utc = datetime.now(UTC).isoformat()
        cur = signal_con.execute(
            "INSERT OR IGNORE INTO signals "
            "(id, ts_utc, source, category, title, link, sentiment, impact, "
            "assets, rationale, scored) VALUES (?,?,?,?,?,?,?,?,?,?,1)",
            (f"rss:{rid}", ts_utc, "rss", cat, title, link, sent, impact,
             assets, rationale))
        added += cur.rowcount
    signal_con.commit()
    return added


# ----------------------------------------------------------------------
# COMPUTE — the decay + aggregation transform.
# ----------------------------------------------------------------------
def _decayed_features(rows, now, tau):
    """rows: list of (ts_utc, category, sentiment, impact). Returns feature dict."""
    decayed_impact = 0.0
    decayed_sentiment = 0.0
    max_impact = 0
    cat_impact = {}
    for ts_utc, category, sentiment, impact in rows:
        if sentiment is None or impact is None:
            continue
        try:
            dt = datetime.fromisoformat(ts_utc)
        except (ValueError, TypeError):
            continue
        age = (now - dt).total_seconds()
        if age < 0:
            age = 0
        w = impact * math.exp(-age / tau)
        decayed_impact += w
        decayed_sentiment += sentiment * w
        max_impact = max(max_impact, impact)
        cat_impact[category] = cat_impact.get(category, 0.0) + w
    net_tone = (decayed_sentiment / decayed_impact) if decayed_impact > 1e-9 else 0.0
    top_cat = max(cat_impact, key=cat_impact.get) if cat_impact else None
    return {
        "decayed_impact": round(decayed_impact, 4),
        "decayed_sentiment": round(decayed_sentiment, 4),
        "net_tone": round(net_tone, 4),
        "max_impact": max_impact,
        "n_events": len(rows),
        "top_category": top_cat,
    }


def compute(horizon="swing", now=None, con=None):
    """Compute the feature vector for a horizon from current signals."""
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}; pick {list(HORIZONS)}")
    cfg = HORIZONS[horizon]
    now = now or datetime.now(UTC)
    close = con is None
    con = con or sqlite3.connect(SIGNAL_DB)
    _ensure_tables(con)
    cutoff = (now - timedelta(seconds=cfg["window"])).isoformat()
    rows = con.execute(
        "SELECT ts_utc, category, sentiment, impact FROM signals "
        "WHERE scored=1 AND ts_utc > ?", (cutoff,)).fetchall()
    feat = _decayed_features(rows, now, cfg["tau"])
    if close:
        con.close()
    return feat


def sentiment_state(horizon="swing"):
    """PUBLIC API for the regime engine. Ingests fresh RSS rows first, then
    returns the current decayed feature vector for the requested horizon."""
    con = sqlite3.connect(SIGNAL_DB)
    _ensure_tables(con)
    ingest_rss(con)
    feat = compute(horizon, con=con)
    con.close()
    return feat


def recompute_and_store(now=None):
    """Recompute all horizons and persist a snapshot row per horizon.
    Call this on your bar cadence (e.g. each 1m/5m bar close)."""
    now = now or datetime.now(UTC)
    con = sqlite3.connect(SIGNAL_DB)
    _ensure_tables(con)
    ingest_rss(con)
    snap = {}
    for h in HORIZONS:
        feat = compute(h, now=now, con=con)
        con.execute(
            "INSERT OR REPLACE INTO sentiment_features "
            "(ts_utc, horizon, decayed_impact, decayed_sentiment, net_tone, "
            "max_impact, n_events, top_category) VALUES (?,?,?,?,?,?,?,?)",
            (now.isoformat(), h, feat["decayed_impact"], feat["decayed_sentiment"],
             feat["net_tone"], feat["max_impact"], feat["n_events"],
             feat["top_category"]))
        snap[h] = feat
    con.commit()
    con.close()
    return snap


# ----------------------------------------------------------------------
# SELF-TEST — synthetic events, no external services.
# ----------------------------------------------------------------------
def _selftest():
    print("sentiment_engine self-test (synthetic, in-memory)\n")
    con = sqlite3.connect(":memory:")
    _ensure_tables(con)
    now = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)
    synthetic = [
        # (minutes_ago, category, sentiment, impact)
        (2,   "HACK",     -4, 5),   # very recent, big, bearish
        (10,  "FED",      +2, 4),   # recent, bullish
        (90,  "ETF_FLOWS",+3, 3),   # older bullish
        (400, "REGULATION",-5, 5),  # stale bearish (should barely register at swing)
    ]
    for i, (mins, cat, sent, imp) in enumerate(synthetic):
        ts = (now - timedelta(minutes=mins)).isoformat()
        con.execute("INSERT INTO signals (id, ts_utc, source, category, title, "
                    "sentiment, impact, scored) VALUES (?,?,?,?,?,?,?,1)",
                    (f"t{i}", ts, "test", cat, f"event {i}", sent, imp))
    con.commit()
    for h in HORIZONS:
        feat = compute(h, now=now, con=con)
        print(f"  [{h:5}] net_tone={feat['net_tone']:+.2f}  "
              f"decayed_impact={feat['decayed_impact']:6.2f}  "
              f"top={feat['top_category']:<11} n={feat['n_events']}")
    con.close()
    print("\n  Expect: scalp dominated by the fresh HACK (net_tone strongly "
          "negative), macro more balanced as the stale REGULATION event still "
          "carries weight. PASS if scalp net_tone < swing net_tone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--horizon", default=None, help="scalp|swing|macro")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.horizon:
        print(f"{args.horizon}:", sentiment_state(args.horizon))
    else:
        snap = recompute_and_store()
        now_wib = datetime.now(UTC).astimezone(WIB)
        print(f"Sentiment state @ {now_wib:%Y-%m-%d %H:%M} WIB")
        for h, f in snap.items():
            print(f"  [{h:5}] net_tone={f['net_tone']:+.2f}  "
                  f"decayed_impact={f['decayed_impact']:.2f}  "
                  f"n={f['n_events']}  top={f['top_category']}")
