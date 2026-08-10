"""
alpha_ml/train.py — orchestrates the full alpha-research pipeline:

    OKX historical candles (data.py)
      -> leakage-safe feature table (features.py, same signal vocabulary
         okx_tele_bot.py / growhf_reactive_bot.py already alert on)
      -> chronological search/holdout split (search_frac, default 70/30)
      -> phase 1: XGBoost entry classifier, embargoed walk-forward CV,
         fit only on the search period (xgb_entry.py)
      -> phase 2: DDQN position policy, trained only on the search period
         (env.py, ddqn.py)
      -> backtest BOTH periods (backtest.py) — search is in-sample and
         expected to look better than reality; holdout is the number that
         matters, since it's the only one the agent never trained against.
      -> persist progress (status.py) so /alpha_status in okx_tele_bot.py can
         report it on demand, any time, without re-running anything.

The search/holdout split exists because DailyScraper/NEOBDM's own
ddqn_entry_exit.py already demonstrated the failure mode it guards against:
that repo's DDQN agent scored a search-period Sharpe of 1.68, which collapsed
to -0.65 on an untouched holdout — a clean overfit, not a real edge, and it
would have looked like a working strategy if only the in-sample number had
been reported. Splitting before training is what caught it. Same discipline
here: never trust bt_search_* over bt_holdout_*.

Run:  py train.py --selftest                                   # fast, synthetic, no network
      py train.py --inst BTC-USDT-SWAP --bar 5m --bars 5000     # real data
Deps: numpy, pandas, scikit-learn, xgboost, torch
"""

import argparse
from pathlib import Path

import numpy as np

from backtest import compute_metrics, report as backtest_report, run_policy
from env import PerpTradingEnv
from ddqn import DDQNAgent
from features import assemble, synthetic_market
from status import format_telegram, write_status
from xgb_entry import build_dataset, evaluate as xgb_evaluate, train_final

MODELS_DIR = Path(__file__).with_name("models")

BT_METRIC_KEYS = [
    "total_return", "sharpe", "sortino", "max_drawdown", "profit_factor",
    "trade_count", "win_rate", "turnover", "fees_paid", "gated_entries",
]


def _prefixed(prefix, metrics):
    return {f"{prefix}_{k}": metrics[k] for k in BT_METRIC_KEYS}

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
       search_frac=0.7, seed=0):
    print(f"[1/5] building feature table (window={window}) ...")
    df, cols = assemble(market, window=window)
    print(f"      {len(df)} rows x {len(cols)} features")

    split = int(len(df) * search_frac)
    train_df = df.iloc[:split].reset_index(drop=True)
    holdout_df = df.iloc[split:].reset_index(drop=True)
    print(f"      chronological split: {len(train_df)} search / {len(holdout_df)} holdout "
          f"({search_frac:.0%}/{1 - search_frac:.0%}) — holdout is never touched during "
          f"training, same discipline as DailyScraper/NEOBDM's ddqn_entry_exit.py")

    print("[2/5] phase 1 — XGBoost entry classifier (embargoed walk-forward, search period only) ...")
    xgb_df, xgb_cols = build_dataset(market, window=window, max_hold=max_hold,
                                     direction_mode="momentum")
    xgb_split = int(len(xgb_df) * search_frac)
    xgb_train_df = xgb_df.iloc[:xgb_split].reset_index(drop=True)
    xgb_res = xgb_evaluate(xgb_train_df, xgb_cols, n_folds=xgb_folds, max_hold=max_hold)
    scorer = train_final(xgb_train_df, xgb_cols)
    xgb_path = MODELS_DIR / f"xgb_entry_{inst}_{bar}.json"
    scorer.save(xgb_path)
    print(f"      search-period walk-forward OOS AUC {xgb_res['auc']:.4f}  "
          f"acc {xgb_res['acc']:.4f}  -> {xgb_path}")

    print(f"[3/5] phase 2 — DDQN training on search period only "
          f"({ddqn_episodes} episodes, episode_len={episode_len}) ...")
    env = PerpTradingEnv(train_df, cols, episode_len=episode_len, fee=fee, seed=seed)
    agent = DDQNAgent(env.state_dim, env.ACTION_DIM, seed=seed,
                      eps_decay_episodes=max(int(ddqn_episodes * 0.6), 1))
    history = agent.train(env, episodes=ddqn_episodes, log_every=max(ddqn_episodes // 5, 1))
    ddqn_path = MODELS_DIR / f"ddqn_{inst}_{bar}.pt"
    agent.save(ddqn_path)
    tail = max(len(history) // 5, 1)
    avg_reward = float(np.mean(history[-tail:]))
    print(f"      last-{tail}-episode avg reward {avg_reward:+.5f}  -> {ddqn_path}")

    bpy = bars_per_year(bar)

    print("\n[4/5] backtest — SEARCH period (in-sample; expect this to flatter the policy) ...")
    trace_search = run_policy(train_df, cols, agent=agent, scorer=scorer, fee=fee,
                              prob_threshold=prob_threshold, episode_len=episode_len)
    m_search = compute_metrics(trace_search, bars_per_year=bpy)
    backtest_report(m_search)

    print("\n[5/5] backtest — HOLDOUT period (never seen in training; this is the number that matters) ...")
    trace_holdout = run_policy(holdout_df, cols, agent=agent, scorer=scorer, fee=fee,
                               prob_threshold=prob_threshold, episode_len=episode_len)
    m_holdout = compute_metrics(trace_holdout, bars_per_year=bpy)
    backtest_report(m_holdout)

    gap = m_search["sharpe"] - m_holdout["sharpe"]
    overfit = m_holdout["sharpe"] <= 0 <= m_search["sharpe"] or gap > 1.0
    if overfit:
        print(f"\n  ⚠ overfit warning: search Sharpe {m_search['sharpe']:+.2f} vs holdout "
              f"{m_holdout['sharpe']:+.2f} (gap {gap:+.2f}) — same collapse pattern seen in "
              f"DailyScraper/NEOBDM's ddqn_entry_exit.py. Trust holdout, not search.")

    status = {
        "inst": inst, "bar": bar, "search_frac": search_frac,
        "xgb_auc": xgb_res["auc"], "xgb_acc": xgb_res["acc"],
        "xgb_expectancy_bps": xgb_res["expectancy"] * 1e4,
        "ddqn_episodes": ddqn_episodes, "ddqn_avg_reward": avg_reward,
        "overfit_flag": overfit,
        **_prefixed("bt_search", m_search),
        **_prefixed("bt_holdout", m_holdout),
        "by_regime_search": m_search["by_regime"],
        "by_regime_holdout": m_holdout["by_regime"],
    }
    write_status(status)
    return status, {"search": m_search, "holdout": m_holdout}


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
    ap.add_argument("--search-frac", type=float, default=0.7,
                    help="fraction of bars used to train (search); rest is untouched holdout")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        print("alpha_ml/train.py self-test — full pipeline on synthetic data\n")
        market = synthetic_market(n=3000, planted=True)
        status, metrics = run("SELFTEST", "5m", market, window=48, max_hold=10,
                              xgb_folds=3, ddqn_episodes=60, episode_len=80, seed=0)
        assert status["xgb_auc"] == status["xgb_auc"], "xgb_auc is NaN"       # NaN != NaN
        assert status["bt_search_trade_count"] >= 0
        assert status["bt_holdout_trade_count"] >= 0
        print("\n" + format_telegram(status))
        print("\n  PASS — phase 1 + phase 2 + search/holdout backtest + status all ran end-to-end.")
    else:
        from data import fetch_ohlcv
        market = fetch_ohlcv(args.inst, args.bar, args.bars)
        if market.empty:
            raise SystemExit("no market data (network?) — try --selftest for a synthetic run")
        status, metrics = run(args.inst, args.bar, market, window=args.window,
                              max_hold=args.max_hold, xgb_folds=args.xgb_folds,
                              ddqn_episodes=args.ddqn_episodes, episode_len=args.episode_len,
                              fee=args.fee, prob_threshold=args.prob_threshold,
                              search_frac=args.search_frac, seed=args.seed)
        print("\n" + format_telegram(status))
