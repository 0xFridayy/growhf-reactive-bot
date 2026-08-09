"""
alpha_ml/labels.py — triple-barrier labeling (the "was this entry profitable"
ground truth both phase 1 and phase 2 train against).

For a hypothetical entry at bar i's close, walk forward until price hits a
take-profit, a stop-loss, or a max-holding-period vertical barrier — whichever
comes first — the standard triple-barrier method (Lopez de Prado). This is
what turns raw historical price action into the same yes/no "was this a good
entry" signal a profitable discretionary trader would have made, without
needing actual trade records.

Fully forward-looking by construction (it's a label, not a feature) — callers
must drop the tail rows where fewer than max_hold forward bars exist, or join
against features.py's table with an embargo, exactly like
nlp-macro-event-bot/features.py's make_label() does.

Run:  py labels.py --selftest
Deps: numpy, pandas
"""

import argparse

import numpy as np
import pandas as pd


def _as_array(x, n):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if len(arr) != n:
        raise ValueError(f"expected length {n}, got {len(arr)}")
    return arr


def triple_barrier(df, tp_frac=0.01, sl_frac=0.01, max_hold=12, direction=1):
    """df needs columns h,l,c, oldest-first. tp_frac/sl_frac: scalar or
    per-bar array (e.g. ATR-scaled). direction: +1 long, -1 short.

    Returns a DataFrame aligned to df with:
      y            1.0 win / 0.0 loss / NaN (not enough forward bars yet)
      fwd_ret      realized signed return at exit (already sign-adjusted for direction)
      exit_bars    bars until exit
      exit_reason  "tp" | "sl" | "sl(ambiguous)" | "time"

    Same-bar TP+SL (can't be ordered from OHLC alone) resolves to SL — the
    conservative assumption, same spirit as costed backtests that assume the
    worst intrabar path when they can't observe it.
    """
    c = df["c"].to_numpy(dtype=float)
    h = df["h"].to_numpy(dtype=float)
    l = df["l"].to_numpy(dtype=float)
    n = len(df)
    tp_frac = _as_array(tp_frac, n)
    sl_frac = _as_array(sl_frac, n)

    y = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)
    exit_bars = np.full(n, np.nan)
    exit_reason = np.array([""] * n, dtype=object)

    for i in range(n):
        entry = c[i]
        if entry <= 0 or np.isnan(tp_frac[i]) or np.isnan(sl_frac[i]):
            continue
        if direction > 0:
            tp_px, sl_px = entry * (1 + tp_frac[i]), entry * (1 - sl_frac[i])
        else:
            tp_px, sl_px = entry * (1 - tp_frac[i]), entry * (1 + sl_frac[i])

        resolved = False
        for k in range(1, max_hold + 1):
            j = i + k
            if j >= n:
                break
            hi, lo = h[j], l[j]
            tp_hit = (hi >= tp_px) if direction > 0 else (lo <= tp_px)
            sl_hit = (lo <= sl_px) if direction > 0 else (hi >= sl_px)
            if tp_hit and sl_hit:
                y[i], fwd_ret[i] = 0.0, (sl_px - entry) / entry * direction
                exit_bars[i], exit_reason[i] = k, "sl(ambiguous)"
                resolved = True
                break
            if tp_hit:
                y[i], fwd_ret[i] = 1.0, (tp_px - entry) / entry * direction
                exit_bars[i], exit_reason[i] = k, "tp"
                resolved = True
                break
            if sl_hit:
                y[i], fwd_ret[i] = 0.0, (sl_px - entry) / entry * direction
                exit_bars[i], exit_reason[i] = k, "sl"
                resolved = True
                break
        if not resolved:
            j = i + max_hold
            if j < n:
                ret = (c[j] - entry) / entry * direction
                y[i], fwd_ret[i] = (1.0 if ret > 0 else 0.0), ret
                exit_bars[i], exit_reason[i] = max_hold, "time"
            # else: tail row, fewer than max_hold forward bars exist -> leave NaN

    return pd.DataFrame({"y": y, "fwd_ret": fwd_ret, "exit_bars": exit_bars,
                         "exit_reason": exit_reason})


def triple_barrier_signed(df, direction_arr, tp_frac=0.01, sl_frac=0.01, max_hold=12):
    """Like triple_barrier, but the trade direction varies per bar (e.g. the
    sign of a momentum feature). direction_arr entries of 0 are left unlabeled
    (NaN) — no entry was signaled there."""
    n = len(df)
    direction_arr = np.asarray(direction_arr)
    long_res = triple_barrier(df, tp_frac, sl_frac, max_hold, direction=1)
    short_res = triple_barrier(df, tp_frac, sl_frac, max_hold, direction=-1)
    out = pd.DataFrame({
        "y": np.full(n, np.nan), "fwd_ret": np.full(n, np.nan),
        "exit_bars": np.full(n, np.nan),
        "exit_reason": np.array([""] * n, dtype=object),
    })
    is_long = direction_arr > 0
    is_short = direction_arr < 0
    out.loc[is_long] = long_res.loc[is_long]
    out.loc[is_short] = short_res.loc[is_short]
    return out


def atr_fractions(df, lookback=14, mult=1.5, floor=0.002, ceil=0.08):
    """Rolling ATR (as a fraction of close) * mult, causal (shift(1) so bar i's
    barrier is set from volatility BEFORE bar i, not including it)."""
    prev_c = df["c"].shift(1)
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - prev_c).abs(),
        (df["l"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean().shift(1)
    frac = (atr / df["c"] * mult).clip(lower=floor, upper=ceil)
    return frac.to_numpy(dtype=float)


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
def _selftest():
    print("alpha_ml/labels.py self-test\n")
    from features import synthetic_market
    market = synthetic_market(n=400, planted=True)

    res = triple_barrier(market, tp_frac=0.01, sl_frac=0.01, max_hold=10, direction=1)
    labeled = res.dropna(subset=["y"])
    print(f"  long labels: {len(labeled)}/{len(res)} rows resolved "
          f"(tail of {len(res) - len(labeled)} awaiting forward bars)")
    # Every row with a full max_hold forward bars available must resolve (via
    # tp/sl or the time barrier); only the last max_hold rows may be NaN, and
    # only if neither tp nor sl fired before the data ran out.
    guaranteed = res.iloc[:len(res) - 10]
    assert guaranteed["y"].notna().all(), "a row with full forward data was left unlabeled"
    assert labeled["y"].isin([0.0, 1.0]).all(), "label not clean binary"
    assert (labeled["exit_bars"] >= 1).all() and (labeled["exit_bars"] <= 10).all()

    # Sanity: a dead-simple deterministic ramp always hits TP before SL.
    ramp = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC"),
        "o": np.linspace(100, 120, 20), "h": np.linspace(100.5, 120.5, 20),
        "l": np.linspace(99.5, 119.5, 20), "c": np.linspace(100, 120, 20),
        "v": np.ones(20),
    })
    r = triple_barrier(ramp, tp_frac=0.01, sl_frac=0.01, max_hold=5, direction=1)
    assert r.dropna(subset=["y"])["y"].eq(1.0).all(), \
        "a monotonic uptrend must always resolve to a TP win going long"

    # Direction flip: shorting the same ramp must always lose.
    r_short = triple_barrier(ramp, tp_frac=0.01, sl_frac=0.01, max_hold=5, direction=-1)
    assert r_short.dropna(subset=["y"])["y"].eq(0.0).all(), \
        "shorting a monotonic uptrend must always resolve to a loss"

    frac = atr_fractions(market, lookback=14, mult=1.5)
    assert np.isnan(frac[:14]).all(), "ATR fraction should be NaN before lookback fills"
    assert np.nanmin(frac) >= 0.002 and np.nanmax(frac) <= 0.08, "ATR fraction not clipped"

    print("  clean binary labels ✓   full-forward-data rows always resolved ✓   "
          "deterministic ramp always wins long / loses short ✓   ATR causal+clipped ✓")
    print("\n  PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("give --selftest")
