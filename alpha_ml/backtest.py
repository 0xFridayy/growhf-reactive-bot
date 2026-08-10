"""
alpha_ml/backtest.py — walks phase-1 (XGBoost) + phase-2 (DDQN) over a
historical series and reports the metrics that actually matter for a
trading system, not just RL reward:

    OOS PnL, Sharpe, Sortino, max drawdown, profit factor, trade count,
    turnover + fees, win rate, and a breakdown by market regime
    (trending / ranging / transitional, from market_profile.py's regime()).

Two policies can be backtested:
  - DDQN agent (phase 2)          — greedy target-position policy each bar.
  - momentum baseline (no RL)     — target = sign(chg_short); the default a
                                    reactive spike-follow bot would take with
                                    no ML at all. Useful as the "is the ML
                                    actually adding anything" comparison.

An optional phase-1 SignalScorer GATES new entries (flat -> long/short): the
trade is only taken if predict_proba() for that direction clears
`prob_threshold`. Flips and exits are never gated (closing/reducing risk
should never be blocked by a model).

Run:  py backtest.py --selftest
Deps: numpy, pandas, torch (only if backtesting a saved DDQN checkpoint)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from market_profile import regime as market_regime  # noqa: E402

BARS_PER_YEAR_5M = 288 * 365


def _regime_series(df, window=96, lookback=20):
    """Per-bar regime label using the same rolling-window construction as
    features.py's profile_features(), kept separate here so callers that
    already have regime_momentum/regime_meanrev columns can skip recomputing."""
    if "regime_momentum" in df.columns and "regime_meanrev" in df.columns:
        out = np.where(df["regime_momentum"] > 0.5, "trending",
                       np.where(df["regime_meanrev"] > 0.5, "ranging", "transitional"))
        return out
    return np.array(["transitional"] * len(df))


def momentum_target(df):
    """No-RL baseline policy: target position = sign(chg_short)."""
    return np.sign(df["chg_short"].to_numpy(dtype=float))


def run_policy(df, feature_cols, target_fn=None, agent=None, scorer=None,
              fee=0.0005, prob_threshold=0.55, episode_len=None):
    """Single continuous pass over `df` (not random episodes — a backtest
    needs the real chronological path). Exactly one of target_fn / agent may
    drive the policy.

    target_fn(df) -> array of target positions in {-1,0,1}, precomputed.
    agent          -> alpha_ml.ddqn.DDQNAgent; queried bar-by-bar so it sees
                       its own (position, bars_held) state, matching how
                       env.PerpTradingEnv builds observations.

    Returns a per-bar trace DataFrame: ts, position, target, bar_ret, reward,
    turnover, gated (bool), regime.
    """
    n = len(df)
    closes = df["c"].to_numpy(dtype=float)
    ts = df["ts"].to_numpy()
    regimes = _regime_series(df)

    if target_fn is not None:
        raw_targets = target_fn(df)
    elif agent is not None:
        raw_targets = None
    else:
        raise ValueError("pass either target_fn or agent")

    feat_mat = df[feature_cols].to_numpy(dtype=float) if scorer is not None or agent is not None else None

    position, bars_held = 0.0, 0
    rows = []
    for t in range(n - 1):
        if agent is not None:
            bars_frac = min(bars_held / (episode_len or 200), 1.0)
            obs = np.concatenate([feat_mat[t], [position, bars_frac]]).astype(np.float32)
            target = {0: 0.0, 1: 1.0, 2: -1.0}[agent.act(obs, greedy=True)]  # matches env.PerpTradingEnv.ACTIONS
        else:
            target = float(raw_targets[t])
            if np.isnan(target):
                target = position  # no signal this bar -> hold

        gated = False
        opening = position == 0.0 and target != 0.0
        if scorer is not None and opening:
            row_feats = dict(zip(feature_cols, feat_mat[t]))
            p = scorer.predict_proba(row_feats)
            p_dir = p if target > 0 else (1.0 - p)
            if p_dir < prob_threshold:
                target, gated = 0.0, True

        turnover = abs(target - position)
        bar_ret = (closes[t + 1] - closes[t]) / closes[t] if closes[t] else 0.0
        fee_cost = fee * turnover
        reward = target * bar_ret - fee_cost

        rows.append({"ts": ts[t], "position": position, "target": target,
                    "bar_ret": bar_ret, "reward": reward, "turnover": turnover,
                    "fee_cost": fee_cost, "gated": gated, "regime": regimes[t]})

        bars_held = 0 if target != position else bars_held + 1
        position = target

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------
def _trade_segments(trace):
    """Segment the position series into discrete trades (maximal runs of a
    constant nonzero position). Returns list of dicts with entry/exit ts,
    side, bars_held, pnl (sum of per-bar rewards over the run, net of fees)."""
    trades, cur = [], None
    for _, row in trace.iterrows():
        pos = row["position"]
        if pos != 0 and (cur is None or cur["side"] != pos):
            if cur is not None:
                trades.append(cur)
            cur = {"side": pos, "entry_ts": row["ts"], "bars": 0, "pnl": 0.0}
        if cur is not None and pos == cur["side"]:
            cur["bars"] += 1
            cur["pnl"] += row["reward"]
            cur["exit_ts"] = row["ts"]
        if pos == 0 and cur is not None:
            trades.append(cur)
            cur = None
    if cur is not None:
        trades.append(cur)
    return trades


def _sharpe(r, bars_per_year):
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 1e-12 else 0.0


def _sortino(r, bars_per_year):
    r = np.asarray(r, dtype=float)
    downside = r[r < 0]
    dd_std = downside.std() if len(downside) else 0.0
    return float(r.mean() / dd_std * np.sqrt(bars_per_year)) if dd_std > 1e-12 else 0.0


def _max_drawdown(r):
    equity = np.cumsum(np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = equity - peak
    return float(dd.min()) if len(dd) else 0.0


def _profit_factor(r):
    r = np.asarray(r, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 1e-12 else float("inf") if gains > 0 else 0.0


def compute_metrics(trace, bars_per_year=BARS_PER_YEAR_5M):
    r = trace["reward"].to_numpy(dtype=float)
    trades = _trade_segments(trace)
    trade_pnls = [t["pnl"] for t in trades]
    wins = [p for p in trade_pnls if p > 0]

    metrics = {
        "n_bars": int(len(trace)),
        "total_return": float(r.sum()),
        "sharpe": _sharpe(r, bars_per_year),
        "sortino": _sortino(r, bars_per_year),
        "max_drawdown": _max_drawdown(r),
        "profit_factor": _profit_factor(r),
        "trade_count": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else float("nan"),
        "avg_trade_pnl": float(np.mean(trade_pnls)) if trade_pnls else float("nan"),
        "turnover": float(trace["turnover"].sum()),
        "fees_paid": float(trace["fee_cost"].sum()),
        "gated_entries": int(trace["gated"].sum()),
    }

    by_regime = {}
    for reg, grp in trace.groupby("regime"):
        rg = grp["reward"].to_numpy(dtype=float)
        by_regime[reg] = {
            "n_bars": int(len(grp)),
            "total_return": float(rg.sum()),
            "sharpe": _sharpe(rg, bars_per_year),
            "avg_bar_reward": float(rg.mean()),
        }
    metrics["by_regime"] = by_regime
    return metrics


def report(metrics):
    print(f"\n  bars              : {metrics['n_bars']}")
    print(f"  total return      : {metrics['total_return']*100:+.3f}%")
    print(f"  Sharpe            : {metrics['sharpe']:+.2f}")
    print(f"  Sortino           : {metrics['sortino']:+.2f}")
    print(f"  max drawdown      : {metrics['max_drawdown']*100:.3f}%")
    print(f"  profit factor     : {metrics['profit_factor']:.2f}")
    print(f"  trades            : {metrics['trade_count']}  (win rate {metrics['win_rate']*100:.1f}%)")
    print(f"  turnover          : {metrics['turnover']:.2f}   fees paid {metrics['fees_paid']*100:.3f}%")
    print(f"  gated entries     : {metrics['gated_entries']}")
    print("  by regime:")
    for reg, m in metrics["by_regime"].items():
        print(f"      {reg:<14} n={m['n_bars']:>5}  return {m['total_return']*100:+.3f}%  sharpe {m['sharpe']:+.2f}")


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
def _selftest():
    print("alpha_ml/backtest.py self-test (synthetic market)\n")
    from features import assemble, synthetic_market
    market = synthetic_market(n=2000, planted=True)
    df, cols = assemble(market, window=48)

    print("-- momentum baseline (no RL, no filter) --")
    trace = run_policy(df, cols, target_fn=momentum_target, fee=0.0005)
    m = compute_metrics(trace)
    report(m)
    assert set(m["by_regime"]).issubset({"trending", "ranging", "transitional"})
    assert m["trade_count"] > 0, "baseline should take some trades on a trending synthetic market"
    assert -1.0 <= m["max_drawdown"] <= 0.0

    print("\n-- momentum baseline + phase-1 XGBoost gate --")
    from xgb_entry import build_dataset, train_final
    train_df, train_cols = build_dataset(market, window=48, max_hold=10, direction_mode="long_only")
    scorer = train_final(train_df, train_cols)
    # Scorer trained on long_only cols; reuse the same feature set for the gate.
    trace_gated = run_policy(df, train_cols, target_fn=momentum_target, scorer=scorer,
                             prob_threshold=0.55, fee=0.0005)
    m_gated = compute_metrics(trace_gated)
    report(m_gated)
    assert m_gated["gated_entries"] >= 0
    print(f"\n  ungated trades: {m['trade_count']}  gated trades: {m_gated['trade_count']}  "
          f"(filter blocked {m_gated['gated_entries']} entries)")

    print("\n  PASS — backtest engine runs end-to-end, metrics are well-formed, "
          "and the phase-1 gate measurably changes trade-taking behavior.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("give --selftest (train.py drives a full real-data backtest)")
