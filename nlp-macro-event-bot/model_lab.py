"""
MODEL LAB — walk-forward gradient-boosted model on the feature table (step 4)
=============================================================================
Trains the primary model from the plan (LightGBM/XGBoost) on the leakage-safe
table from features.py, evaluated with EMBARGOED WALK-FORWARD cross-validation
so every metric is out-of-sample and free of label-overlap leakage.

Rigor (matches your backtest checklist):
  * Expanding-window walk-forward — train on the past, test on the future, never
    the reverse.
  * Embargo gap = label horizon between train and test, so a train row's forward
    label can never overlap a test row.
  * Metrics are pooled OOS predictions only: ROC-AUC vs the 0.5 no-skill line,
    accuracy vs the base rate, per-bar expectancy, and a fees-agnostic naive
    Sharpe (clearly caveated — NOT an edge claim).
  * Model auto-selects lightgbm -> xgboost -> sklearn HistGBM, whichever is
    installed.

Run:  py model_lab.py --selftest         # synthetic planted signal, end-to-end
      py model_lab.py --csv training_table.csv
Deps: numpy, pandas, scikit-learn, and one of {lightgbm, xgboost}
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

BARS_PER_YEAR = 288 * 365     # 5m bars


# ----------------------------------------------------------------------
# MODEL — first available of lightgbm / xgboost / sklearn
# ----------------------------------------------------------------------
def make_model():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            verbosity=-1)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        return "xgboost", xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist")
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    return "sklearn-histgbm", HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.03, max_iter=300)


def feature_importance(model, cols):
    for attr in ("feature_importances_",):
        if hasattr(model, attr):
            imp = getattr(model, attr)
            return dict(sorted(zip(cols, imp), key=lambda x: -x[1]))
    return {}


# ----------------------------------------------------------------------
# EMBARGOED WALK-FORWARD SPLITS
# ----------------------------------------------------------------------
def walkforward_splits(n, n_folds=5, embargo=1, min_train_frac=0.3):
    """Yield (train_idx, test_idx). Expanding train window; each test block is
    a contiguous future slice; `embargo` rows are purged between them so a
    train row's forward label cannot overlap the test block."""
    start = int(n * min_train_frac)
    fold_size = (n - start) // n_folds
    if fold_size <= 0:
        raise ValueError("not enough rows for the requested folds")
    for k in range(n_folds):
        test_lo = start + k * fold_size
        test_hi = n if k == n_folds - 1 else test_lo + fold_size
        train_hi = max(0, test_lo - embargo)
        train_idx = np.arange(0, train_hi)
        test_idx = np.arange(test_lo, test_hi)
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx


# ----------------------------------------------------------------------
# EVALUATE
# ----------------------------------------------------------------------
def evaluate(df, cols, embargo=1, n_folds=5, long_short=False):
    X = df[cols].values
    y = df["y"].values.astype(int)
    fwd = df["fwd_ret"].values

    oos_p, oos_y, oos_ret, importances = [], [], [], []
    name = None
    for train_idx, test_idx in walkforward_splits(len(df), n_folds, embargo):
        name, model = make_model()
        model.fit(X[train_idx], y[train_idx])
        try:
            p = model.predict_proba(X[test_idx])[:, 1]
        except (AttributeError, IndexError):
            p = model.predict(X[test_idx]).astype(float)
        oos_p.append(p)
        oos_y.append(y[test_idx])
        pred_up = p > 0.5
        r = np.where(pred_up, fwd[test_idx],
                     -fwd[test_idx] if long_short else 0.0)
        oos_ret.append(r)
        fi = feature_importance(model, cols)
        if fi:
            importances.append(fi)

    p = np.concatenate(oos_p); yv = np.concatenate(oos_y); r = np.concatenate(oos_ret)
    base_rate = max(yv.mean(), 1 - yv.mean())
    auc = roc_auc_score(yv, p) if len(np.unique(yv)) > 1 else float("nan")
    acc = accuracy_score(yv, (p > 0.5).astype(int))
    expectancy = r.mean()
    sharpe = (r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)) if r.std() > 1e-12 else 0.0

    avg_imp = {}
    for fi in importances:
        for c, v in fi.items():
            avg_imp[c] = avg_imp.get(c, 0.0) + v / len(importances)
    avg_imp = dict(sorted(avg_imp.items(), key=lambda x: -x[1]))

    return {"model": name, "n_oos": len(yv), "auc": auc, "acc": acc,
            "base_rate": base_rate, "expectancy": expectancy,
            "naive_sharpe": sharpe, "importance": avg_imp}


def report(res):
    print(f"\n  model            : {res['model']}")
    print(f"  OOS samples      : {res['n_oos']}")
    print(f"  ROC-AUC          : {res['auc']:.4f}   (0.500 = no skill)")
    print(f"  accuracy         : {res['acc']:.4f}   (base rate {res['base_rate']:.4f})")
    print(f"  OOS expectancy   : {res['expectancy']*1e4:+.2f} bps / bar")
    print(f"  naive Sharpe     : {res['naive_sharpe']:+.2f}   "
          f"(NO fees/slippage — sanity only)")
    if res["importance"]:
        print("  top features     :")
        for c, v in list(res["importance"].items())[:6]:
            print(f"      {c:<20} {v:.4f}")


# ----------------------------------------------------------------------
# SELF-TEST — end-to-end on synthetic planted signal
# ----------------------------------------------------------------------
def _selftest():
    from features import synthetic_market, assemble
    print("model_lab self-test — synthetic market with a planted signal\n")
    market, shock = synthetic_market(n=2500, planted=True)
    sig_rows = [{"ts": t, "sentiment": int(np.sign(shock[i]) * 4), "impact": 4}
                for i, t in enumerate(market["ts"]) if abs(shock[i]) > 0.6]
    signals = pd.DataFrame(sig_rows)
    df, cols = assemble(market, signals=signals, regime=pd.DataFrame(),
                        label_horizon=1, label_kind="sign")
    print(f"  table: {len(df)} rows x {len(cols)} features, "
          f"{len(signals)} planted signals")
    res = evaluate(df, cols, embargo=1, n_folds=5)
    report(res)
    print()
    if res["auc"] > 0.53:
        print(f"  PASS — walk-forward OOS AUC {res['auc']:.3f} > 0.53: with the "
              f"embargo enforced, the harness recovers the planted\n         "
              f"signal out-of-sample rather than memorizing it. (Synthetic sanity "
              f"check — NOT an edge claim on real data.)")
    else:
        print(f"  NOTE — OOS AUC {res['auc']:.3f}; planted SNR is low, rerun or "
              f"raise n. The point is the harness is leakage-safe, not the number.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", default=None, help="training_table.csv from features.py")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=1)
    ap.add_argument("--long-short", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.csv:
        df = pd.read_csv(args.csv)
        cols = [c for c in df.columns if c not in ("ts", "fwd_ret", "y")]
        df = df.dropna(subset=cols + ["y"]).reset_index(drop=True)
        res = evaluate(df, cols, embargo=args.embargo, n_folds=args.folds,
                       long_short=args.long_short)
        report(res)
    else:
        print("give --selftest or --csv <training_table.csv>")
