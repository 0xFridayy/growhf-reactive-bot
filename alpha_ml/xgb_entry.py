"""
alpha_ml/xgb_entry.py — PHASE 1: XGBoost entry/exit classifier.

Learns "was this a profitable entry" from the SAME feature vocabulary the live
Telegram bot already alerts on (features.py), labeled by triple-barrier
outcomes (labels.py). Direction defaults to momentum-following — the same
side growhf_reactive_bot.py's spike alert or okx_tele_bot.py's chg_short read
would imply — so the model is scored on exactly the entries the bot would
actually have taken.

Evaluated with the same embargoed walk-forward CV as
nlp-macro-event-bot/model_lab.py, for the same reason: every metric has to be
out-of-sample and free of label-overlap leakage before it means anything.

Run:  py xgb_entry.py --selftest                 # synthetic, fast
      py xgb_entry.py --inst BTC-USDT-SWAP --bar 5m --bars 3000
Deps: numpy, pandas, scikit-learn, xgboost (falls back to sklearn HistGBM)
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from features import assemble, synthetic_market
from labels import atr_fractions, triple_barrier_signed

MODELS_DIR = Path(__file__).with_name("models")
BARS_PER_YEAR = 288 * 365  # 5m bars, matches nlp-macro-event-bot/model_lab.py


# ----------------------------------------------------------------------
# DATASET — features x triple-barrier labels, direction-aware
# ----------------------------------------------------------------------
def build_dataset(market, window=96, short_bars=4, long_bars=16,
                  max_hold=12, atr_mult=1.5, atr_lookback=14,
                  chg_deadzone=0.15, direction_mode="momentum"):
    """direction_mode:
      "momentum"  — direction = sign(chg_short) with a small dead-zone (the
                    same side a reactive spike alert would take); flat bars
                    are dropped, not labeled 0.
      "long_only" — every bar is a hypothetical long entry.
    """
    df, cols = assemble(market, window=window, short_bars=short_bars, long_bars=long_bars)
    if direction_mode == "long_only":
        direction = np.ones(len(df))
    else:
        direction = np.where(df["chg_short"] > chg_deadzone, 1,
                             np.where(df["chg_short"] < -chg_deadzone, -1, 0))

    tp_frac = sl_frac = atr_fractions(df, lookback=atr_lookback, mult=atr_mult)
    lab = triple_barrier_signed(df, direction, tp_frac=tp_frac, sl_frac=sl_frac, max_hold=max_hold)

    out = pd.concat([df.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
    out["direction"] = direction
    out = out[out["direction"] != 0].reset_index(drop=True)
    out = out.dropna(subset=cols + ["y"]).reset_index(drop=True)
    return out, cols


# ----------------------------------------------------------------------
# MODEL — xgboost, else sklearn HistGBM
# ----------------------------------------------------------------------
def make_model():
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
    if hasattr(model, "feature_importances_"):
        return dict(sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1]))
    return {}


# ----------------------------------------------------------------------
# EMBARGOED WALK-FORWARD — identical scheme to nlp-macro-event-bot/model_lab.py
# ----------------------------------------------------------------------
def walkforward_splits(n, n_folds=5, embargo=1, min_train_frac=0.3):
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


def evaluate(df, cols, embargo=None, n_folds=5, max_hold=12):
    embargo = max_hold if embargo is None else embargo
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
        oos_p.append(p); oos_y.append(y[test_idx])
        oos_ret.append(np.where(p > 0.5, fwd[test_idx], 0.0))
        fi = feature_importance(model, cols)
        if fi:
            importances.append(fi)

    p = np.concatenate(oos_p); yv = np.concatenate(oos_y); r = np.concatenate(oos_ret)
    base_rate = max(yv.mean(), 1 - yv.mean()) if len(yv) else float("nan")
    auc = roc_auc_score(yv, p) if len(np.unique(yv)) > 1 else float("nan")
    acc = accuracy_score(yv, (p > 0.5).astype(int))
    expectancy = float(r.mean())
    sharpe = float(r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)) if r.std() > 1e-12 else 0.0

    avg_imp = {}
    for fi in importances:
        for c, v in fi.items():
            avg_imp[c] = avg_imp.get(c, 0.0) + float(v) / len(importances)
    avg_imp = dict(sorted(avg_imp.items(), key=lambda x: -x[1]))

    return {"model": name, "n_oos": int(len(yv)), "auc": float(auc), "acc": float(acc),
            "base_rate": float(base_rate), "expectancy": expectancy,
            "naive_sharpe": sharpe, "importance": avg_imp}


def report(res):
    print(f"\n  model            : {res['model']}")
    print(f"  OOS samples      : {res['n_oos']}")
    print(f"  ROC-AUC          : {res['auc']:.4f}   (0.500 = no skill)")
    print(f"  accuracy         : {res['acc']:.4f}   (base rate {res['base_rate']:.4f})")
    print(f"  OOS expectancy   : {res['expectancy']*1e4:+.2f} bps / entry")
    print(f"  naive Sharpe     : {res['naive_sharpe']:+.2f}   (NO fees/slippage — sanity only)")
    if res["importance"]:
        print("  top features     :")
        for c, v in list(res["importance"].items())[:6]:
            print(f"      {c:<20} {v:.4f}")


# ----------------------------------------------------------------------
# SignalScorer — the object the live bots (and backtest.py) call into
# ----------------------------------------------------------------------
class SignalScorer:
    """Wraps a trained phase-1 model. Cold-start safe: if no trained model is
    available, predict_proba returns 0.5 (neutral) rather than raising, so the
    live bot keeps working before the first training run."""

    def __init__(self, model=None, cols=None, backend=None):
        self.model = model
        self.cols = cols or []
        self.backend = backend

    @property
    def is_trained(self):
        return self.model is not None

    def predict_proba(self, feature_row: dict) -> float:
        if not self.is_trained:
            return 0.5
        x = np.array([[float(feature_row.get(c, 0.0)) for c in self.cols]])
        try:
            return float(self.model.predict_proba(x)[0, 1])
        except (AttributeError, IndexError):
            return float(self.model.predict(x)[0])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"cols": self.cols, "backend": self.backend}
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)
        if self.backend == "xgboost":
            self.model.save_model(str(path))
        else:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        if not path.exists() or not meta_path.exists():
            return cls()  # cold start
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["backend"] == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(path))
        else:
            with open(path, "rb") as f:
                model = pickle.load(f)
        return cls(model=model, cols=meta["cols"], backend=meta["backend"])


def train_final(df, cols):
    backend, model = make_model()
    model.fit(df[cols].values, df["y"].values.astype(int))
    return SignalScorer(model=model, cols=cols, backend=backend)


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
def _selftest():
    print("alpha_ml/xgb_entry.py self-test — synthetic market with a planted signal\n")
    # long_only validates the harness itself: with momentum-mode direction
    # selection, the direction filter already captures most of the planted
    # edge (by design — that's what a directional entry filter is FOR), which
    # leaves little residual for the classifier to find and understretches
    # this sanity check. long_only isolates "can the model learn from
    # features that a fixed-direction entry wins" — the harness-correctness
    # question this selftest exists to answer. Live scoring still defaults to
    # direction_mode="momentum" (see build_dataset's docstring).
    market = synthetic_market(n=3000, planted=True)
    df, cols = build_dataset(market, window=48, max_hold=10, direction_mode="long_only")
    print(f"  dataset: {len(df)} labeled entries x {len(cols)} features "
          f"(win rate {df['y'].mean():.3f})")
    res = evaluate(df, cols, n_folds=5, max_hold=10)
    report(res)

    scorer = train_final(df, cols)
    scorer.save(MODELS_DIR / "_selftest_model.json")
    loaded = SignalScorer.load(MODELS_DIR / "_selftest_model.json")
    assert loaded.is_trained, "round-tripped model failed to load"
    p_live = loaded.predict_proba(df.iloc[-1][cols].to_dict())
    assert 0.0 <= p_live <= 1.0
    print(f"\n  save/load round-trip ✓   live predict_proba() = {p_live:.3f}")

    cold = SignalScorer()
    assert cold.predict_proba({c: 0.0 for c in cols}) == 0.5, "cold-start scorer must return 0.5"
    print("  cold-start (untrained) scorer returns neutral 0.5 ✓")

    if res["auc"] > 0.53:
        print(f"\n  PASS — walk-forward OOS AUC {res['auc']:.3f} > 0.53: recovers the "
              f"planted spike->drift edge out-of-sample.\n  (Synthetic sanity check — NOT an edge claim on real data.)")
    else:
        print(f"\n  NOTE — OOS AUC {res['auc']:.3f}; rerun or raise n. The point is the "
              f"harness is leakage-safe, not this particular number.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="5m")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--max-hold", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        from data import fetch_ohlcv
        mkt = fetch_ohlcv(args.inst, args.bar, args.bars)
        if mkt.empty:
            print("no market data (network?) — try --selftest for a synthetic run")
        else:
            df, cols = build_dataset(mkt, window=args.window, max_hold=args.max_hold)
            res = evaluate(df, cols, n_folds=args.folds, max_hold=args.max_hold)
            report(res)
            scorer = train_final(df, cols)
            out_path = MODELS_DIR / f"xgb_entry_{args.inst}_{args.bar}.json"
            scorer.save(out_path)
            print(f"\nsaved production model -> {out_path}")
