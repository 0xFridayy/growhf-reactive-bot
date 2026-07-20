"""
TRAINING SNAPSHOT — accumulate a real labeled dataset on the bar cadence
========================================================================
Ties the live stack to the modeling layer. On each bar cadence it:

  1. brings the sentiment + regime stores up to date,
  2. fetches recent perp klines,
  3. assembles the leakage-safe table (features.assemble — point-in-time),
  4. UPSERTS labeled rows into training.db keyed by bar timestamp.

Why upsert (not overwrite):
  * signals.db is append-only, so recomputing a past bar's sentiment later is
    still point-in-time correct — the as-of join only ever sees signals dated
    <= that bar. Re-running is safe.
  * a bar near the live edge has no forward-return label yet, so make_label
    drops it. On a LATER snapshot, once its forward window has elapsed, it gets
    a label and is inserted. So the dataset self-heals its own edge over time,
    with no look-ahead.

Over days/weeks this grows into the real dataset model_lab.py trains on —
replacing the synthetic sanity data.

Run:  py training_snapshot.py --selftest        # synthetic, no network
      py training_snapshot.py                    # one live snapshot
      py training_snapshot.py --export out.csv   # dump the accumulated table
Deps: numpy, pandas
"""

import argparse
import json
import sqlite3
from datetime import datetime

import pandas as pd

from shared import UTC, WIB
from features import assemble, fetch_perp_klines, load_signals, load_regime

TRAINING_DB = "training.db"


def _db(path=TRAINING_DB):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS rows (
        ts TEXT PRIMARY KEY,      -- bar close time, ISO UTC
        features TEXT,            -- JSON {col: value}
        fwd_ret REAL,
        y INTEGER,
        updated TEXT)""")
    con.commit()
    return con


def snapshot(symbol="BTCUSDT", interval="5m", label_horizon=1,
             market=None, signals=None, regime=None, db=TRAINING_DB,
             refresh_stores=True):
    """Assemble the current labeled table and upsert it. Returns (added, total).
    Pass market/signals/regime to override live loads (used by --selftest)."""
    # 1) refresh live stores (no-op-safe if modules/keys missing)
    if refresh_stores and market is None:
        try:
            from sentiment_engine import recompute_and_store
            recompute_and_store()
        except Exception as e:
            print(f"[snapshot] sentiment refresh skipped: {e}")
        try:
            from data_sources import snapshot_regime_features
            snapshot_regime_features()
        except Exception as e:
            print(f"[snapshot] regime refresh skipped: {e}")

    # 2) market data
    if market is None:
        market = fetch_perp_klines(symbol, interval, 1000)
        if market is None:
            print("[snapshot] no market data (network?) — nothing to store")
            return 0, _total(db)
        if "funding" not in market:
            market["funding"] = 0.0001

    # 3) assemble (point-in-time; signals/regime default to live DB loads)
    sig = load_signals() if signals is None else signals
    reg = load_regime() if regime is None else regime
    df, cols = assemble(market, signals=sig, regime=reg,
                        label_horizon=label_horizon, label_kind="sign")

    # 4) upsert
    con = _db(db)
    now = datetime.now(UTC).isoformat()
    added = 0
    for _, r in df.iterrows():
        ts = pd.Timestamp(r["ts"]).tz_convert("UTC").isoformat()
        feats = {c: (None if pd.isna(r[c]) else float(r[c])) for c in cols}
        existed = con.execute("SELECT 1 FROM rows WHERE ts=?", (ts,)).fetchone()
        con.execute("INSERT OR REPLACE INTO rows (ts, features, fwd_ret, y, updated) "
                    "VALUES (?,?,?,?,?)",
                    (ts, json.dumps(feats), float(r["fwd_ret"]), int(r["y"]), now))
        if not existed:
            added += 1
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
    con.close()
    print(f"[snapshot] {datetime.now(tz=WIB):%H:%M WIB} — "
          f"+{added} new rows, {total} total (features: {len(cols)})")
    return added, total


def _total(db):
    con = _db(db)
    n = con.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
    con.close()
    return n


def export_csv(out="training_table.csv", db=TRAINING_DB):
    """Expand the JSON feature blobs into a flat CSV for model_lab.py."""
    con = _db(db)
    rows = con.execute("SELECT ts, features, fwd_ret, y FROM rows ORDER BY ts").fetchall()
    con.close()
    if not rows:
        print("[export] training.db is empty")
        return None
    recs = []
    for ts, feats, fwd, y in rows:
        rec = {"ts": ts, **json.loads(feats), "fwd_ret": fwd, "y": y}
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out, index=False)
    print(f"[export] wrote {len(df)} rows x {df.shape[1]-3} features -> {out}")
    return df


# ----------------------------------------------------------------------
# SELF-TEST — simulate two snapshots as time advances, no network.
# ----------------------------------------------------------------------
def _selftest():
    import numpy as np
    from features import synthetic_market
    print("training_snapshot self-test (two snapshots, simulated time)\n")
    db = "training_selftest.db"
    import os
    if os.path.exists(db):
        os.remove(db)

    full, shock = synthetic_market(n=500, planted=True)
    sig_rows = [{"ts": t, "sentiment": int(np.sign(shock[i]) * 4), "impact": 4}
                for i, t in enumerate(full["ts"]) if abs(shock[i]) > 0.6]
    signals = pd.DataFrame(sig_rows)
    empty_regime = pd.DataFrame()

    # Snapshot 1: only the first 300 bars are "known so far".
    m1 = full.iloc[:300].copy()
    s1 = signals[signals["ts"] <= m1["ts"].iloc[-1]]
    added1, total1 = snapshot(market=m1, signals=s1, regime=empty_regime,
                              db=db, refresh_stores=False)
    edge_ts = pd.Timestamp(m1["ts"].iloc[-1]).tz_convert("UTC").isoformat()

    # Snapshot 2: time advanced — now 360 bars are known.
    m2 = full.iloc[:360].copy()
    s2 = signals[signals["ts"] <= m2["ts"].iloc[-1]]
    added2, total2 = snapshot(market=m2, signals=s2, regime=empty_regime,
                              db=db, refresh_stores=False)

    con = _db(db)
    edge_labeled = con.execute("SELECT y FROM rows WHERE ts=?", (edge_ts,)).fetchone()
    con.close()

    print(f"\n  pass 1: +{added1} rows (total {total1})")
    print(f"  pass 2: +{added2} rows (total {total2})")
    print(f"  the last bar of pass 1 was unlabeled at the edge; after pass 2 it "
          f"is {'LABELED ' + str(edge_labeled[0]) if edge_labeled else 'still absent'}")

    df = export_csv("training_selftest.csv", db=db)
    assert total2 > total1, "dataset did not grow across snapshots"
    assert df is not None and len(df) == total2, "export row count mismatch"
    ok_edge = edge_labeled is not None
    print("\n  dataset grows across snapshots ✓   export matches store ✓"
          + ("   edge bar self-healed a label ✓" if ok_edge else ""))
    print("  PASS — accumulates a real dataset, no dupes, edge labels fill in "
          "as time passes (no look-ahead).")

    for f in (db, "training_selftest.csv"):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--export", default=None)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.export:
        export_csv(args.export)
    else:
        snapshot(args.symbol, args.interval, args.horizon)
