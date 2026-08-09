"""
alpha_ml/train.py — orchestrates the full alpha-research pipeline:

    OKX historical candles (data.py)
      -> leakage-safe feature table (features.py, same signal vocabulary
         okx_tele_bot.py / growhf_reactive_bot.py already alert on)
      -> phase 1: XGBoost entry classifier, embargoed walk-forward CV (xgb_entry.py)
      -> phase 2: DDQN position policy trained against a trading env (env.py, ddqn.py)
      -> backtest the DDQN policy gated by the XGBoost model (backtest.py)
      -> persist progress (status.py) so /alpha_status in okx_tele_bot.py can
         report it on demand, any time, without re-running anything.

Run:  py train.py --selftest                                   # fast, synthetic, no network
      py train.py --inst BTC-USDT-SWAP --bar 5m --bars 5000     # real data
Deps: numpy, pandas, scikit-learn, xgboost, torch
"""

import argparse
from pathlib import Path

import numpy as np

from backtest import compute_metrics, run_policy
from env import PerpTradingEnv
from ddqn import DDQNAgent
from features import assemble, synthetic_market
from status import format_telegram, write_status
from xgb_entry import build_dataset, evaluate as xgb_evaluate, train_final

MODELS_DIR = Path(__file__).with_name("models")

BAR_MINUTES = {"m": 1, "H": 60, "D": 60 * 24}


def bars_per_year(bar):
    """'5m' -> 60*24*365/5, '1H' -> 24*365, etc. Falls back to 5m cadence for
    anything unrecognized (only affects the Sharpe/Sortino annualization
    constant, not the backtest itself)."""
    try:
        unit = bar[-1] if bar[-1].isalpha() else "m"
        n = int(bar[:-1]) if bar[:-1] else 1
        minutes = BAR_MINUTES.get(unit.upper() if unit.upper() in ("H", "D") else "m", 1) * n
        return int(365 * 24 * 60 / minutes)
    except (ValueError, IndexError, ZeroDivisionError):
        return 288 * 365


def run(inst, bar, market, window=96, max_hold=12, xgb_folds=5,
       ddqn_episodes=200, episode_len=200, fee=0.0005, prob_threshold=0.55,
       seed=0):
    print(f"[1/4] building feature table (window={window}) ...")
    df, cols = assemble(market, window=window)
    print(f"      {len(df)} rows x {len(cols)} features")

    print("[2/4] phase 1 — XGBoost entry classifier (embargoed walk-forward) ...")
    xgb_df, xgb_cols = build_dataset(market, window=window, max_hold=max_hold,
                                     direction_mode="momentum")
    xgb_res = xgb_evaluate(xgb_df, xgb_cols, n_folds=xgb_folds, max_hold=max_hold)
    scorer = train_final(xgb_df, xgb_cols)
    xgb_path = MODELS_DIR / f"xgb_entry_{inst}_{bar}.json"
    scorer.save(xgb_path)
    print(f"      OOS AUC {xgb_res['auc']:.4f}  acc {xgb_res['acc']:.4f}  -> {xgb_path}")

    print(f"[3/4] phase 2 — DDQN training ({ddqn_episodes} episodes, "
          f"episode_len={episode_len}) ...")
    env = PerpTradingEnv(df, cols, episode_len=episode_len, fee=fee, seed=seed)
    agent = DDQNAgent(env.state_dim, env.ACTION_DIM, seed=seed,
                      eps_decay_episodes=max(int(ddqn_episodes * 0.6), 1))
    history = agent.train(env, episodes=ddqn_episodes, log_every=max(ddqn_episodes // 5, 1))
    ddqn_path = MODELS_DIR / f"ddqn_{inst}_{bar}.pt"
    agent.save(ddqn_path)
    tail = max(len(history) // 5, 1)
    avg_reward = float(np.mean(history[-tail:]))
    print(f"      last-{tail}-episode avg reward {avg_reward:+.5f}  -> {ddqn_path}")

    print("[4/4] backtest — DDQN policy gated by the XGBoost model, over the full series ...")
    trace = run_policy(df, cols, agent=agent, scorer=scorer, fee=fee,
                       prob_threshold=prob_threshold, episode_len=episode_len)
    metrics = compute_metrics(trace, bars_per_year=bars_per_year(bar))

    status = {
        "inst": inst, "bar": bar,
        "xgb_auc": xgb_res["auc"], "xgb_acc": xgb_res["acc"],
        "xgb_expectancy_bps": xgb_res["expectancy"] * 1e4,
        "ddqn_episodes": ddqn_episodes, "ddqn_avg_reward": avg_reward,
        "bt_total_return": metrics["total_return"], "bt_sharpe": metrics["sharpe"],
        "bt_sortino": metrics["sortino"], "bt_max_drawdown": metrics["max_drawdown"],
        "bt_profit_factor": metrics["profit_factor"], "bt_trade_count": metrics["trade_count"],
        "bt_win_rate": metrics["win_rate"], "bt_turnover": metrics["turnover"],
        "bt_fees_paid": metrics["fees_paid"], "bt_gated_entries": metrics["gated_entries"],
        "by_regime": metrics["by_regime"],
    }
    write_status(status)
    return status, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="5m")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--max-hold", type=int, default=12)
    ap.add_argument("--xgb-folds", type=int, default=5)
    ap.add_argument("--ddqn-episodes", type=int, default=200)
    ap.add_argument("--episode-len", type=int, default=200)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--prob-threshold", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        print("alpha_ml/train.py self-test — full pipeline on synthetic data\n")
        market = synthetic_market(n=3000, planted=True)
        status, metrics = run("SELFTEST", "5m", market, window=48, max_hold=10,
                              xgb_folds=3, ddqn_episodes=60, episode_len=80, seed=0)
        assert status["xgb_auc"] == status["xgb_auc"], "xgb_auc is NaN"       # NaN != NaN
        assert status["bt_trade_count"] >= 0
        print("\n" + format_telegram(status))
        print("\n  PASS — phase 1 + phase 2 + backtest + status all ran end-to-end.")
    else:
        from data import fetch_ohlcv
        market = fetch_ohlcv(args.inst, args.bar, args.bars)
        if market.empty:
            raise SystemExit("no market data (network?) — try --selftest for a synthetic run")
        status, metrics = run(args.inst, args.bar, market, window=args.window,
                              max_hold=args.max_hold, xgb_folds=args.xgb_folds,
                              ddqn_episodes=args.ddqn_episodes, episode_len=args.episode_len,
                              fee=args.fee, prob_threshold=args.prob_threshold, seed=args.seed)
        print("\n" + format_telegram(status))
