"""
BACKTEST — validation battery for the modeling harness
======================================================
Real-data backtesting isn't the first question for a system validated only on
synthetic data — HARNESS CORRECTNESS is. Before any Sharpe number means
anything, you have to prove the pipeline (a) doesn't leak the future, (b) finds
nothing when there's nothing to find, and (c) recovers a signal in proportion
to how strong it really is. This module runs those experiments and writes
backtest_results.json for the results dashboard.

Experiments (all on synthetic data, well-powered):
  1. PERMUTATION / leakage   shuffle labels -> OOS AUC must collapse to ~0.50
  2. EMBARGO sensitivity      AUC vs purge gap between train/test
  3. HORIZON sweep            which forward horizon is most predictable
  4. FEATURE ablation         marginal value of sentiment vs technical features
  5. PER-FOLD stability       is the edge consistent across time, not one fold?
  6. SNR recovery curve       AUC rises with planted signal strength; 0 -> ~0.50

Run:  py backtest.py            # full battery (~1-2 min)
      py backtest.py --quick    # smaller n for a fast pass
Deps: numpy, pandas, scikit-learn, xgboost (or lightgbm)
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import synthetic_market, assemble
from model_lab import walkforward_splits, make_model, BARS_PER_YEAR

SEED = 12


def build(n, effect_scale=1.0, seed=SEED, sig_threshold=0.6, label_horizon=1,
          persistent=True):
    """Synthetic market + signals -> assembled table. effect_scale scales the
    planted signal (0.0 => none). persistent=True embeds a DECAYING impulse into
    price (momentum features can read it); persistent=False embeds only a sparse
    one-bar bump under heavy noise, so the edge lives in SENTIMENT, not price."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    shock = rng.normal(0, 1, n)
    ret = rng.normal(0, 0.004, n)
    if effect_scale:
        if persistent:
            ir = effect_scale * 0.0016 * np.exp(-np.arange(1, 17) / 8.0)
        else:
            ir = np.array([effect_scale * 0.0011])   # single next-bar bump only
        for i in range(n):
            for k, h in enumerate(ir, start=1):
                if i + k < n:
                    ret[i + k] += h * shock[i]
    price = 60000 * np.exp(np.cumsum(ret))
    high = price * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = rng.lognormal(6, 0.5, n)
    funding = rng.normal(0.0001, 0.00005, n)
    market = pd.DataFrame({"ts": ts, "o": price, "h": high, "l": low,
                           "c": price, "v": vol, "funding": funding})
    sig_rows = [{"ts": t, "sentiment": int(np.sign(shock[i]) * 4), "impact": 4}
                for i, t in enumerate(ts) if abs(shock[i]) > sig_threshold]
    signals = pd.DataFrame(sig_rows) if sig_rows else \
        pd.DataFrame(columns=["ts", "sentiment", "impact"])
    df, cols = assemble(market, signals=signals, regime=pd.DataFrame(),
                        label_horizon=label_horizon, label_kind="sign")
    return df, cols


def run_wf(df, cols, embargo=1, folds=5, shuffle_y=False, seed=SEED):
    """Walk-forward eval. Returns (pooled_auc, per_fold_aucs, expectancy)."""
    X = df[cols].values
    y = df["y"].values.astype(int)
    fwd = df["fwd_ret"].values
    if shuffle_y:
        y = np.random.default_rng(seed).permutation(y)
    oos_p, oos_y, oos_ret, per_fold = [], [], [], []
    for tr, te in walkforward_splits(len(df), folds, embargo):
        _, model = make_model()
        model.fit(X[tr], y[tr])
        try:
            p = model.predict_proba(X[te])[:, 1]
        except (AttributeError, IndexError):
            p = model.predict(X[te]).astype(float)
        oos_p.append(p); oos_y.append(y[te])
        oos_ret.append(np.where(p > 0.5, fwd[te], 0.0))
        if len(np.unique(y[te])) > 1:
            per_fold.append(round(roc_auc_score(y[te], p), 4))
    p = np.concatenate(oos_p); yv = np.concatenate(oos_y); r = np.concatenate(oos_ret)
    auc = round(roc_auc_score(yv, p), 4) if len(np.unique(yv)) > 1 else float("nan")
    sharpe = round(r.mean() / r.std() * np.sqrt(BARS_PER_YEAR), 2) if r.std() > 1e-12 else 0.0
    return auc, per_fold, sharpe


def _tech_sent(cols):
    sent = [c for c in cols if c.startswith("sent_")]
    tech = [c for c in cols if c not in sent]
    return tech, sent


# ----------------------------------------------------------------------
# EXPERIMENTS
# ----------------------------------------------------------------------
def battery(n=4000, folds=5):
    R = {"config": {"n": n, "folds": folds, "seed": SEED}}
    print(f"Building base table (n={n}) …")
    df, cols = build(n)
    tech, sent = _tech_sent(cols)
    R["n_rows"] = len(df); R["features"] = cols

    # 1) PERMUTATION / LEAKAGE
    print("1) permutation (label-shuffle) leakage test …")
    real_auc, _, _ = run_wf(df, cols, embargo=1)
    shuf = [run_wf(df, cols, embargo=1, shuffle_y=True, seed=SEED + k)[0]
            for k in range(5)]
    R["permutation"] = {"real_auc": real_auc,
                        "shuffled_auc_mean": round(float(np.mean(shuf)), 4),
                        "shuffled_auc_max": round(float(np.max(shuf)), 4),
                        "shuffled_runs": shuf}

    # 2) EMBARGO SENSITIVITY
    print("2) embargo sensitivity …")
    R["embargo"] = {str(e): run_wf(df, cols, embargo=e)[0]
                    for e in (0, 1, 3, 6, 12)}

    # 3) HORIZON SWEEP  (embargo = horizon each time)
    print("3) horizon sweep …")
    R["horizon"] = {}
    for h in (1, 3, 6, 12):
        dfh, colsh = build(n, label_horizon=h)
        auc, _, sharpe = run_wf(dfh, colsh, embargo=h)
        R["horizon"][str(h)] = {"auc": auc, "naive_sharpe": sharpe, "rows": len(dfh)}

    # 4) FEATURE ABLATION
    print("4) feature ablation …")
    R["ablation"] = {
        "all": run_wf(df, cols, embargo=1)[0],
        "technical_only": run_wf(df, tech, embargo=1)[0],
        "sentiment_only": run_wf(df, sent, embargo=1)[0],
    }

    # 5) PER-FOLD STABILITY
    print("5) per-fold stability …")
    auc, per_fold, _ = run_wf(df, cols, embargo=1)
    R["per_fold"] = {"pooled": auc, "folds": per_fold,
                     "min": min(per_fold), "max": max(per_fold)}

    # 6) SNR RECOVERY CURVE
    print("6) SNR recovery curve …")
    R["snr"] = {}
    for scale in (0.0, 0.5, 1.0, 2.0):
        dfs, colss = build(n, effect_scale=scale)
        R["snr"][str(scale)] = run_wf(dfs, colss, embargo=1)[0]

    return R


def report(R):
    p = R["permutation"]
    print("\n" + "=" * 60)
    print("  VALIDATION BATTERY RESULTS")
    print("=" * 60)
    print(f"  rows={R['n_rows']}  features={len(R['features'])}\n")
    leak_ok = p["shuffled_auc_max"] < 0.55 and p["real_auc"] > 0.55
    print(f"1) PERMUTATION / LEAKAGE   {'PASS' if leak_ok else 'CHECK'}")
    print(f"     real AUC        {p['real_auc']:.3f}")
    print(f"     shuffled AUC    mean {p['shuffled_auc_mean']:.3f}  "
          f"max {p['shuffled_auc_max']:.3f}   (must be ~0.50)")
    print("2) EMBARGO SENSITIVITY")
    for e, a in R["embargo"].items():
        print(f"     embargo {e:>2} bars -> AUC {a:.3f}")
    print("3) HORIZON SWEEP")
    for h, d in R["horizon"].items():
        print(f"     H={h:>2} bars     -> AUC {d['auc']:.3f}  "
              f"naive_sharpe {d['naive_sharpe']:+.1f}")
    print("4) FEATURE ABLATION (signal-in-price regime)")
    for k, a in R["ablation"].items():
        print(f"     {k:<15} -> AUC {a:.3f}")
    print("5) PER-FOLD STABILITY")
    print(f"     folds {R['per_fold']['folds']}  "
          f"(min {R['per_fold']['min']:.3f}, max {R['per_fold']['max']:.3f})")
    print("6) SNR RECOVERY CURVE")
    for s, a in R["snr"].items():
        print(f"     effect x{s:<4} -> AUC {a:.3f}")
    snr0 = R["snr"]["0.0"]
    print("=" * 60)
    print(f"  Headlines: no-signal AUC {snr0:.3f} (~0.50 = no false edge), "
          f"label-shuffle max {p['shuffled_auc_max']:.3f} (no leakage).")
    print("  Synthetic sanity — NOT an edge claim on real data.")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="backtest_results.json")
    args = ap.parse_args()
    R = battery(n=1500 if args.quick else 4000, folds=5)
    report(R)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nwrote {args.out}")
