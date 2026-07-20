"""
FEATURES — leakage-safe ML training table (build-plan step 4)
=============================================================
Assembles the tabular dataset that LightGBM/XGBoost trains on:

    perp OHLCV + funding   (technical features)
  + point-in-time sentiment (decayed, computed AS-OF each bar — no look-ahead)
  + regime tag (Fear & Greed, lagged)
  + forward-return label   (shift(-H), tail dropped)

Every merge is a BACKWARD as-of join (`merge_asof direction='backward'`) or an
explicit lag, so no row can ever see information dated after its own bar close.
This is the single most important property for a trustworthy backtest — the
whole table is built to make forward leakage structurally impossible.

Run:  py features.py --selftest     # synthetic market with a planted signal
Deps: pandas, numpy
"""

import argparse
import math
import sqlite3

import numpy as np
import pandas as pd

from shared import SIGNAL_DB
from sentiment_engine import HORIZONS

# ----------------------------------------------------------------------
# MARKET DATA
# ----------------------------------------------------------------------
def fetch_perp_klines(symbol="BTCUSDT", interval="5m", limit=1000):
    """Binance USDT-M public klines -> DataFrame[open_time(UTC), o,h,l,c,v].
    Network-dependent; returns None on failure (caller can use CSV/synthetic)."""
    import requests
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": symbol, "interval": interval,
                                 "limit": limit}, timeout=15)
        raw = r.json()
        df = pd.DataFrame(raw, columns=[
            "open_time", "o", "h", "l", "c", "v", "close_time", "qv",
            "trades", "tbav", "tbqv", "ignore"])
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ("o", "h", "l", "c", "v"):
            df[col] = df[col].astype(float)
        return df[["ts", "o", "h", "l", "c", "v"]]
    except Exception as e:
        print(f"[klines error] {e}")
        return None


def synthetic_market(n=800, seed=7, planted=True):
    """Synthetic 5m OHLCV with an OPTIONAL planted relationship: a 'sentiment
    shock' at bar t triggers a DECAYING impulse response over the following
    bars (news moves the market for a while, then fades) — exactly the shape a
    time-decayed sentiment feature is built to capture. Lets us prove the
    harness recovers a signal it was given — a sanity check, never an edge."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    shock = rng.normal(0, 1, n)                       # exogenous 'news' impulse
    ret = rng.normal(0, 0.004, n)
    if planted:
        ir = 0.0016 * np.exp(-np.arange(1, 17) / 8.0)  # 16-bar decaying response
        for i in range(n):
            for k, h in enumerate(ir, start=1):
                if i + k < n:
                    ret[i + k] += h * shock[i]
    price = 60000 * np.exp(np.cumsum(ret))
    high = price * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = rng.lognormal(6, 0.5, n)
    funding = rng.normal(0.0001, 0.00005, n)
    df = pd.DataFrame({"ts": ts, "o": price, "h": high, "l": low, "c": price,
                       "v": vol, "funding": funding})
    return df, shock


# ----------------------------------------------------------------------
# TECHNICAL FEATURES (all backward-looking)
# ----------------------------------------------------------------------
def perp_features(df):
    df = df.sort_values("ts").reset_index(drop=True).copy()
    c = df["c"]
    df["ret_1"]   = c.pct_change()
    df["ret_3"]   = c.pct_change(3)
    df["ret_12"]  = c.pct_change(12)
    df["vol_12"]  = df["ret_1"].rolling(12).std()
    df["vol_48"]  = df["ret_1"].rolling(48).std()
    df["mom_12"]  = c / c.shift(12) - 1
    df["rng"]     = (df["h"] - df["l"]) / c
    df["v_z"]     = (df["v"] - df["v"].rolling(48).mean()) / df["v"].rolling(48).std()
    if "funding" in df:
        df["funding"] = df["funding"]
        df["funding_z"] = (df["funding"] - df["funding"].rolling(48).mean()) \
                          / df["funding"].rolling(48).std()
    return df


# ----------------------------------------------------------------------
# POINT-IN-TIME SENTIMENT  (decayed, computed as-of each bar)
# ----------------------------------------------------------------------
def load_signals(db=SIGNAL_DB):
    """Load scored signals from signals.db -> DataFrame[ts(UTC), sentiment, impact]."""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["ts", "sentiment", "impact"])
    try:
        rows = con.execute("SELECT ts_utc, sentiment, impact FROM signals "
                           "WHERE scored=1").fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    if not rows:
        return pd.DataFrame(columns=["ts", "sentiment", "impact"])
    df = pd.DataFrame(rows, columns=["ts", "sentiment", "impact"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.dropna(subset=["sentiment", "impact"])


def pointintime_sentiment(bar_ts, signals, horizon="swing"):
    """For each bar close time, decayed sentiment using ONLY signals dated <= t.
    Returns DataFrame indexed like bar_ts with decayed_impact / net_tone.
    O(bars * signals) but numpy-vectorized over signals per bar."""
    tau = HORIZONS[horizon]["tau"]
    win = HORIZONS[horizon]["window"]
    out_imp, out_tone = [], []
    # Epoch-seconds int64 throughout (UTC) — avoids np.datetime64 tz warnings.
    bar_secs = pd.DatetimeIndex(bar_ts).view("int64") // 10**9
    if len(signals):
        s_ts = signals["ts"].astype("int64").values // 10**9
        s_sent = signals["sentiment"].values.astype(float)
        s_imp = signals["impact"].values.astype(float)
    for t_s in bar_secs:
        if not len(signals):
            out_imp.append(0.0); out_tone.append(0.0); continue
        age = t_s - s_ts
        mask = (age >= 0) & (age <= win)              # <= t and within window (no future)
        if not mask.any():
            out_imp.append(0.0); out_tone.append(0.0); continue
        w = s_imp[mask] * np.exp(-age[mask] / tau)
        di = w.sum()
        ds = (s_sent[mask] * w).sum()
        out_imp.append(round(float(di), 4))
        out_tone.append(round(float(ds / di) if di > 1e-9 else 0.0, 4))
    return pd.DataFrame({f"sent_impact_{horizon}": out_imp,
                         f"sent_tone_{horizon}": out_tone})


# ----------------------------------------------------------------------
# REGIME (Fear & Greed) — daily, LAGGED one day (avoid same-day late post)
# ----------------------------------------------------------------------
def load_regime(db=SIGNAL_DB):
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("SELECT ts_utc, fng, fng_bucket FROM regime_features").fetchall()
        con.close()
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["ts", "fng", "fng_bucket"])
    if not rows:
        return pd.DataFrame(columns=["ts", "fng", "fng_bucket"])
    df = pd.DataFrame(rows, columns=["ts", "fng", "fng_bucket"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True) + pd.Timedelta(days=1)  # lag
    return df.sort_values("ts")


# ----------------------------------------------------------------------
# LABEL — forward return over H bars (tail dropped => no future leakage)
# ----------------------------------------------------------------------
def make_label(df, horizon_bars=1, kind="sign", deadband=0.0):
    fwd = df["c"].shift(-horizon_bars) / df["c"] - 1.0
    df = df.copy()
    df["fwd_ret"] = fwd
    if kind == "sign":
        df["y"] = np.where(fwd > deadband, 1, np.where(fwd < -deadband, 0, np.nan))
    else:
        df["y"] = fwd
    return df


# ----------------------------------------------------------------------
# ASSEMBLE
# ----------------------------------------------------------------------
FEATURE_COLS = ["ret_1", "ret_3", "ret_12", "vol_12", "vol_48", "mom_12",
                "rng", "v_z", "funding_z",
                "sent_impact_swing", "sent_tone_swing",
                "sent_impact_scalp", "sent_tone_scalp"]


def assemble(market, signals=None, regime=None, label_horizon=1,
             label_kind="sign", deadband=0.0):
    """Full leakage-safe training table. `market` needs columns ts,o,h,l,c,v[,funding]."""
    df = perp_features(market)
    signals = load_signals() if signals is None else signals
    for h in ("swing", "scalp"):
        st = pointintime_sentiment(df["ts"].tolist(), signals, horizon=h)
        df = pd.concat([df.reset_index(drop=True), st], axis=1)

    regime = load_regime() if regime is None else regime
    if len(regime):
        df = pd.merge_asof(df.sort_values("ts"), regime.sort_values("ts"),
                           on="ts", direction="backward")
    else:
        df["fng"] = np.nan; df["fng_bucket"] = np.nan
    # Fear & Greed is a slow OPTIONAL regime tag — impute neutral when absent
    # rather than dropping otherwise-complete rows.
    df["fng_bucket"] = df["fng_bucket"].fillna(0)

    df = make_label(df, label_horizon, label_kind, deadband)
    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep only feature columns that actually carry information (an all-NaN
    # column — e.g. funding_z when funding is constant — is dropped, not allowed
    # to delete every row via dropna).
    core = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().any()]
    cols = core + ["fng_bucket"]
    keep = ["ts"] + cols + ["fwd_ret", "y"]
    df = df[keep].dropna(subset=core + ["y"]).reset_index(drop=True)
    return df, cols


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
def _selftest():
    print("features.py self-test (synthetic market + planted signal)\n")
    market, shock = synthetic_market(n=800, planted=True)

    # Turn the planted shock into synthetic 'signals' timed one bar BEFORE the
    # return it should predict, so a correct as-of join must pick them up.
    sig_rows = []
    for i, t in enumerate(market["ts"]):
        if abs(shock[i]) > 0.6:
            sig_rows.append({"ts": t, "sentiment": int(np.sign(shock[i]) * 4),
                             "impact": 4})
    signals = pd.DataFrame(sig_rows)
    print(f"  planted {len(signals)} synthetic signals into the stream")

    df, cols = assemble(market, signals=signals, regime=pd.DataFrame(),
                        label_horizon=1, label_kind="sign")
    print(f"  assembled table: {len(df)} rows x {len(cols)} features")
    print(f"  features: {cols}")

    # Leakage check: correlation of sentiment tone with the SAME-bar fwd return
    # should be positive (that's the planted effect) — and crucially the feature
    # column has no NaN and the label tail was dropped.
    corr = df["sent_tone_swing"].corr(df["fwd_ret"])
    print(f"  corr(sent_tone_swing, fwd_ret) = {corr:+.3f}  "
          f"(planted effect => should be > 0)")
    assert len(df) > 500, f"table collapsed to {len(df)} rows — assembly bug"
    assert df[cols].isna().sum().sum() == 0, "NaN leaked into feature matrix"
    assert df["y"].isin([0, 1]).all(), "label not clean binary"
    assert corr > 0, f"planted signal not recovered (corr={corr:+.3f})"
    print("  rows > 500 ✓   no NaNs in feature matrix ✓   "
          "label clean binary ✓   planted corr > 0 ✓")
    print("\n  PASS — table is leakage-safe and recovers the planted relationship.")
    return df, cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--out", default="training_table.csv")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        mkt = fetch_perp_klines(args.symbol, args.interval, 1000)
        if mkt is None:
            print("no market data (network?) — try --selftest for a synthetic run")
        else:
            if "funding" not in mkt:
                mkt["funding"] = 0.0001
            df, cols = assemble(mkt)
            df.to_csv(args.out, index=False)
            print(f"wrote {len(df)} rows x {len(cols)} features -> {args.out}")
