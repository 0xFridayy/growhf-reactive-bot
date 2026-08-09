"""
alpha_ml/features.py — leakage-safe feature table for the alpha-research pipeline.

Every feature is the SAME signal family the live bots already alert on
(okx_tele_bot.py's momentum/funding/TPO/mean-reversion/regime read, and
growhf_reactive_bot.py's price-spike + volume-ratio read) — just computed
across a full historical series so XGBoost (phase 1) and the DDQN env
(phase 2) train on the identical vocabulary the Telegram bot already speaks.

All technical features are backward-looking only (pandas .shift()/.rolling()),
and the TPO/mean-reversion/regime columns at bar t are built from a rolling
window that ends AT bar t (reusing market_profile.py, unmodified) — never
future candles. That structural property is what makes the resulting table
safe to label with a forward-looking outcome (labels.py) without leakage.

Run:  py features.py --selftest        # synthetic market, fast
Deps: numpy, pandas
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from market_profile import build_tpo_profile, mean_reversion, regime  # noqa: E402

TECH_COLS = ["ret_1", "ret_3", "ret_12", "vol_12", "vol_48", "mom_12", "rng",
             "v_z", "vol_ratio", "chg_short", "chg_long"]
PROFILE_COLS = ["poc_dist_pct", "in_value_area", "value_area_pct", "mr_zscore",
                "regime_er", "regime_momentum", "regime_meanrev"]
FUNDING_COLS = ["funding_z"]
ALL_COLS = TECH_COLS + PROFILE_COLS + FUNDING_COLS


# ----------------------------------------------------------------------
# SYNTHETIC MARKET — same shape as nlp-macro-event-bot/features.py so the
# selftest doesn't depend on network access.
# ----------------------------------------------------------------------
def synthetic_market(n=1000, seed=7, planted=True):
    """Synthetic OHLCV with an OPTIONAL planted spike->drift relationship: a
    volume/price shock at bar t is followed by a decaying continuation, the
    exact shape a reactive spike-follow signal is built to capture."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    shock = rng.normal(0, 1, n)
    ret = rng.normal(0, 0.003, n)
    vol_base = rng.lognormal(6, 0.4, n)
    if planted:
        ir = 0.0018 * np.exp(-np.arange(1, 13) / 6.0)
        for i in range(n):
            if abs(shock[i]) > 1.2:
                vol_base[i] *= 4.0  # the volume spike itself
                # The spike bar's OWN return also moves with the shock — a
                # reactive momentum-follow strategy only has an edge if the
                # spike's direction is observable AT the spike bar (exactly
                # what growhf_reactive_bot.py's price+volume confirmation
                # checks), not just in bars that haven't happened yet.
                ret[i] += 0.006 * np.sign(shock[i])
                for k, h in enumerate(ir, start=1):
                    if i + k < n:
                        ret[i + k] += h * np.sign(shock[i])
    price = 50000 * np.exp(np.cumsum(ret))
    high = price * (1 + np.abs(rng.normal(0, 0.0015, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.0015, n)))
    funding = rng.normal(0.0001, 0.00005, n)
    df = pd.DataFrame({"ts": ts, "o": price, "h": high, "l": low, "c": price,
                       "v": vol_base, "funding": funding})
    return df


# ----------------------------------------------------------------------
# TECHNICAL FEATURES — vectorized, backward-looking only
# ----------------------------------------------------------------------
def technical_features(df, short_bars=4, long_bars=16, vol_lookback=10):
    df = df.sort_values("ts").reset_index(drop=True).copy()
    c, v = df["c"], df["v"]
    df["ret_1"] = c.pct_change()
    df["ret_3"] = c.pct_change(3)
    df["ret_12"] = c.pct_change(12)
    df["vol_12"] = df["ret_1"].rolling(12).std()
    df["vol_48"] = df["ret_1"].rolling(48).std()
    df["mom_12"] = c / c.shift(12) - 1
    df["rng"] = (df["h"] - df["l"]) / c
    df["v_z"] = (v - v.rolling(48).mean()) / v.rolling(48).std()
    # Same ratio growhf_reactive_bot.py's process_okx_spike() uses: current bar
    # volume vs the trailing average BEFORE this bar (shift(1) keeps it causal).
    prior_avg_vol = v.rolling(vol_lookback).mean().shift(1)
    df["vol_ratio"] = v / prior_avg_vol
    # Same momentum reads okx_tele_bot.py's analyze() uses (chg1h/chg4h at its
    # 15m bar size); expressed here in bar counts so it works at any --bar.
    df["chg_short"] = c.pct_change(short_bars) * 100.0
    df["chg_long"] = c.pct_change(long_bars) * 100.0
    if "funding" in df:
        df["funding_z"] = (df["funding"] - df["funding"].rolling(48).mean()) \
                          / df["funding"].rolling(48).std()
    return df


# ----------------------------------------------------------------------
# TPO / MEAN-REVERSION / REGIME — reuses market_profile.py's pure functions
# over a rolling window ENDING at each bar (causal by construction).
# ----------------------------------------------------------------------
def _window_candles(rows, i, window):
    """rows: numpy array [ts_ns,o,h,l,c,v] oldest-first. Returns the last
    `window` rows up to and including i, newest-first (market_profile format)."""
    lo = max(0, i - window + 1)
    return rows[lo:i + 1][::-1].tolist()


def profile_features(df, window=96, mr_lookback=20, regime_lookback=20, min_candles=5):
    df = df.reset_index(drop=True)
    rows = df[["ts", "o", "h", "l", "c", "v"]].copy()
    rows["ts"] = rows["ts"].astype("int64")
    rows = rows.values
    n = len(df)
    poc_dist = np.full(n, np.nan); in_va = np.full(n, np.nan); va_pct = np.full(n, np.nan)
    mr_z = np.full(n, np.nan); er = np.full(n, np.nan)
    mom_flag = np.full(n, np.nan); mrv_flag = np.full(n, np.nan)

    for i in range(min_candles - 1, n):
        candles = _window_candles(rows, i, window)
        prof = build_tpo_profile(candles)
        if prof is not None:
            poc_dist[i] = prof.poc_dist_pct
            in_va[i] = 1.0 if prof.in_value_area else 0.0
            va_pct[i] = prof.value_area_pct
        mr = mean_reversion(candles, lookback=mr_lookback)
        if mr is not None:
            mr_z[i] = mr.zscore
        reg = regime(candles, lookback=regime_lookback)
        if reg is not None:
            er[i] = reg.efficiency_ratio
            mom_flag[i] = 1.0 if reg.favored == "momentum" else 0.0
            mrv_flag[i] = 1.0 if reg.favored == "mean-reversion" else 0.0

    out = df.copy()
    out["poc_dist_pct"] = poc_dist
    out["in_value_area"] = in_va
    out["value_area_pct"] = va_pct
    out["mr_zscore"] = mr_z
    out["regime_er"] = er
    out["regime_momentum"] = mom_flag
    out["regime_meanrev"] = mrv_flag
    return out


# ----------------------------------------------------------------------
# ASSEMBLE
# ----------------------------------------------------------------------
def assemble(market, window=96, short_bars=4, long_bars=16):
    """market needs columns ts,o,h,l,c,v[,funding]. Returns (df, feature_cols)
    with every feature column backward-looking only; no label attached here —
    see labels.py for the forward-looking triple-barrier outcome."""
    df = technical_features(market, short_bars=short_bars, long_bars=long_bars)
    df = profile_features(df, window=window)
    df = df.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in ALL_COLS if c in df.columns and df[c].notna().any()]
    keep = ["ts", "o", "h", "l", "c", "v"] + cols
    df = df[keep].dropna(subset=cols).reset_index(drop=True)
    return df, cols


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
def _selftest():
    print("alpha_ml/features.py self-test (synthetic market)\n")
    market = synthetic_market(n=600, planted=True)
    df, cols = assemble(market, window=48, short_bars=4, long_bars=16)
    print(f"  assembled table: {len(df)} rows x {len(cols)} features")
    print(f"  features: {cols}")

    assert len(df) > 300, f"table collapsed to {len(df)} rows — assembly bug"
    assert df[cols].isna().sum().sum() == 0, "NaN leaked into feature matrix"
    # Causality spot-check: profile features computed from a window ending at
    # bar i must not depend on bar i+1's price. Perturbing the future shouldn't
    # change a past row's features.
    market2 = market.copy()
    future_idx = len(market2) - 1
    market2.loc[future_idx, ["o", "h", "l", "c"]] *= 1.5
    df2, _ = assemble(market2, window=48, short_bars=4, long_bars=16)
    past_row = df.iloc[100][PROFILE_COLS].values
    past_row2 = df2.iloc[100][PROFILE_COLS].values
    assert np.allclose(past_row.astype(float), past_row2.astype(float)), \
        "future bar leaked into a past row's profile features"
    print("  rows > 300 ✓   no NaNs in feature matrix ✓   "
          "no future leakage into past rows ✓")
    print("\n  PASS")
    return df, cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="5m")
    ap.add_argument("--bars", type=int, default=2000)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--out", default="features_table.csv")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        from data import fetch_ohlcv
        mkt = fetch_ohlcv(args.inst, args.bar, args.bars)
        if mkt.empty:
            print("no market data (network?) — try --selftest for a synthetic run")
        else:
            df, cols = assemble(mkt, window=args.window)
            df.to_csv(args.out, index=False)
            print(f"wrote {len(df)} rows x {len(cols)} features -> {args.out}")
