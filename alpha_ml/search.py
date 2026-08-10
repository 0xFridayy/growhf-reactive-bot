"""
alpha_ml/search.py — the daily entry/exit strategy search.

Runs every candidate in strategies.default_grid() over real candles, ranks
them, confirms the winner on data the ranking never saw, logs every trade the
winner took with the reason it was taken, and says plainly whether the result
clears the bar you set. Meant to run once a day, forever, until it does.

THREE-WAY CHRONOLOGICAL SPLIT — the part that makes the number mean something
-----------------------------------------------------------------------------
    |------------ train ------------|---- validation ----|--- holdout ---|
     phase-1 XGBoost is fitted here   every candidate is    ONLY the top-k
     and nowhere else                 ranked here          are run here

Two splits are not enough once you are searching. With a plain search/holdout
split, picking the best of N candidates on the holdout makes the holdout an
in-sample number — you have simply moved the overfitting one level up, into the
selection step. So the winner is chosen on VALIDATION, and holdout is touched
only to confirm the already-made choice. Holdout never picks anything.

THE LUCK BAR
------------
Trying hundreds of strategies a day is a great way to find something that never
worked. If you draw N candidates from pure noise, the best of them still posts
a positive Sharpe of roughly

    SE * sqrt(2 * ln N),   SE = sqrt(bars_per_year / n_bars)

so every run reports that threshold next to the winner's validation Sharpe. A
winner below the luck bar is not a finding, however good its numbers look, and
gets flagged as such.

Holdout also wears out: re-testing against the same tail every day slowly turns
it in-sample too. The leaderboard therefore counts how many times each strategy
has been confirmed on a holdout, and the daily run pulls fresh candles so the
tail contains bars that did not exist yesterday.

Run:  py search.py --selftest                                   # synthetic, fast, no network
      py search.py --inst BTC-USDT-SWAP --bar 5m --bars 8000     # real data
Deps: numpy, pandas, scikit-learn, xgboost
"""

import argparse
import json
import math
from datetime import date, datetime, timezone

import numpy as np

from features import assemble, synthetic_market
from status import (CHAMPION_TRADES_CSV, append_leaderboard, holdout_use_count,
                    write_search_status)
from strategies import (baseline_strategy, default_grid, run_strategy,
                        strategy_metrics, trades_to_frame)
from train import bars_per_year
from xgb_entry import build_dataset, train_final

# What "good enough to trade" means. Every one of these must hold on the
# HOLDOUT before a run reports itself satisfied; override from the CLI.
DEFAULT_TARGETS = {
    "sharpe": 1.0,
    "profit_factor": 1.2,
    "trade_count": 20,
    "max_drawdown": -0.10,   # no worse than a 10% peak-to-trough bleed
}


def luck_bar(n_bars, n_configs, bpy):
    """The validation Sharpe a best-of-N search would post on pure noise."""
    if n_bars <= 1 or n_configs < 2:
        return 0.0
    se = math.sqrt(bpy / n_bars)
    return se * math.sqrt(2.0 * math.log(n_configs))


def split_three(df, train_frac=0.5, val_frac=0.25):
    n = len(df)
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return (df.iloc[:a].reset_index(drop=True),
            df.iloc[a:b].reset_index(drop=True),
            df.iloc[b:].reset_index(drop=True))


def score(metrics, min_trades):
    """Rank key. Sharpe is the objective; anything that barely traded is
    unrankable rather than good, so it sorts to the bottom instead of winning
    on three lucky fills."""
    if metrics["trade_count"] < min_trades:
        return -float("inf")
    return metrics["sharpe"]


def evaluate_all(candidates, df, cols, fee, probs, min_trades, bpy, label=""):
    out = []
    for i, strat in enumerate(candidates):
        trace, trades = run_strategy(df, strat, feature_cols=cols, fee=fee, probs=probs)
        m = strategy_metrics(trace, trades, bars_per_year=bpy)
        out.append({"strategy": strat, "metrics": m, "trades": trades,
                    "score": score(m, min_trades)})
        if label and (i + 1) % 50 == 0:
            print(f"      {label}: {i + 1}/{len(candidates)} evaluated")
    return out


def run_search(inst, bar, market, fee=0.0005, train_frac=0.5, val_frac=0.25,
               min_trades=15, top_k=5, max_configs=None, window=96, max_hold=12,
               seed=None, targets=None, persist=True):
    targets = {**DEFAULT_TARGETS, **(targets or {})}
    bpy = bars_per_year(bar)

    print(f"[1/5] building feature table for {inst} {bar} ...")
    df, cols = assemble(market, window=window)
    train_df, val_df, hold_df = split_three(df, train_frac, val_frac)
    print(f"      {len(df)} bars -> {len(train_df)} train / {len(val_df)} validation / "
          f"{len(hold_df)} holdout")
    if min(len(val_df), len(hold_df)) < 50:
        raise SystemExit(f"not enough bars to search on ({len(df)} total) — fetch more")

    print("[2/5] fitting the phase-1 gate on the TRAIN split only ...")
    xdf, xcols = build_dataset(market, window=window, max_hold=max_hold,
                               direction_mode="momentum")
    # Cut the labeled set at the same timestamp the train split ends, so no
    # gate-training row can carry information from a validation-period bar.
    cutoff = train_df["ts"].iloc[-1]
    xtrain = xdf[xdf["ts"] <= cutoff]
    scorer = train_final(xtrain, xcols) if len(xtrain) > 100 else None
    if scorer is None:
        print("      too few labeled rows before the cutoff — running ungated")
    val_probs = scorer.predict_proba_batch(val_df) if scorer else None
    hold_probs = scorer.predict_proba_batch(hold_df) if scorer else None

    candidates = default_grid()
    if max_configs and len(candidates) > max_configs:
        rng = np.random.default_rng(seed if seed is not None else 0)
        pick = rng.choice(len(candidates), size=max_configs, replace=False)
        candidates = [candidates[i] for i in sorted(pick)]
        print(f"      sampled {max_configs} of the full grid (seed {seed}) — a different "
              f"slice each day")
    print(f"[3/5] scoring {len(candidates)} strategies on VALIDATION ...")
    ranked = evaluate_all(candidates, val_df, cols, fee, val_probs, min_trades, bpy,
                          label="validation")
    ranked.sort(key=lambda r: r["score"], reverse=True)
    tradeable = [r for r in ranked if r["score"] > -float("inf")]
    print(f"      {len(tradeable)}/{len(ranked)} took at least {min_trades} trades")
    if not tradeable:
        raise SystemExit(f"no strategy took {min_trades}+ trades on the validation split")

    bar_sharpe = luck_bar(len(val_df), len(candidates), bpy)
    print(f"      luck bar for {len(candidates)} tries over {len(val_df)} bars: "
          f"Sharpe {bar_sharpe:+.2f}")

    print(f"[4/5] confirming the top {top_k} on HOLDOUT (never used for ranking) ...")
    finalists = tradeable[:top_k]
    for r in finalists:
        h_trace, h_trades = run_strategy(hold_df, r["strategy"], feature_cols=cols,
                                         fee=fee, probs=hold_probs)
        r["holdout"] = strategy_metrics(h_trace, h_trades, bars_per_year=bpy)
        r["holdout_trades"] = h_trades

    # The champion is the VALIDATION winner. Deliberately not the best holdout
    # number — picking on holdout is the exact mistake this split exists to
    # prevent, and it would quietly burn the only honest data left.
    champ = finalists[0]

    base = baseline_strategy()
    b_trace, b_trades = run_strategy(hold_df, base, feature_cols=cols, fee=fee)
    base_m = strategy_metrics(b_trace, b_trades, bars_per_year=bpy)

    print("[5/5] writing results ...")
    # A second luck bar for the holdout. Only the top_k finalists are run
    # there, so the multiple-testing penalty is much smaller than validation's
    # — but with 5m bars an annualized Sharpe is a noisy statistic no matter
    # what, and a target of "Sharpe >= 1" is meaningless if noise alone scores
    # 10. Clearing this bar is therefore part of being satisfied, not a footnote.
    hold_bar = luck_bar(len(hold_df), max(len(finalists), 2), bpy)
    passed, checks = check_targets(champ["holdout"], targets, hold_bar)
    prior_uses = holdout_use_count(champ["strategy"].name) if persist else 0

    status = {
        "inst": inst, "bar": bar, "ts_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": date.today().isoformat(),
        "n_bars": len(df), "n_train": len(train_df), "n_val": len(val_df),
        "n_holdout": len(hold_df),
        "fee": fee, "min_trades": min_trades,
        "n_configs_tried": len(candidates),
        "n_configs_tradeable": len(tradeable),
        "luck_bar_sharpe": bar_sharpe,
        "holdout_luck_bar": hold_bar,
        "champion": champ["strategy"].name,
        "champion_spec": champ["strategy"].to_dict(),
        "champion_describe": champ["strategy"].describe(),
        "champion_val": _slim(champ["metrics"]),
        "champion_holdout": _slim(champ["holdout"]),
        "champion_exit_breakdown": champ["holdout"]["exit_breakdown"],
        "champion_holdout_uses": prior_uses + 1,
        "beats_luck_bar": bool(champ["metrics"]["sharpe"] > bar_sharpe),
        "baseline": _slim(base_m),
        "baseline_name": base.name,
        "targets": targets,
        "target_checks": checks,
        "satisfied": passed,
        "top": [{"name": r["strategy"].name,
                 "val": _slim(r["metrics"]),
                 "holdout": _slim(r["holdout"])} for r in finalists],
    }

    if persist:
        write_search_status(status)
        append_leaderboard(status, finalists)
        trades_to_frame(champ["holdout_trades"]).to_csv(CHAMPION_TRADES_CSV, index=False)
        print(f"      champion's {len(champ['holdout_trades'])} holdout trades "
              f"-> {CHAMPION_TRADES_CSV.name}")

    return status, finalists


def _slim(m):
    keys = ["total_return", "sharpe", "sortino", "max_drawdown", "profit_factor",
            "trade_count", "win_rate", "avg_trade_pnl", "avg_bars_held",
            "turnover", "fees_paid", "gated_entries"]
    return {k: (None if k not in m or (isinstance(m[k], float) and not np.isfinite(m[k]))
                else m[k]) for k in keys}


def check_targets(m, targets, holdout_luck_bar=None):
    checks = {
        "sharpe": (m["sharpe"], targets["sharpe"], m["sharpe"] >= targets["sharpe"]),
        "profit_factor": (m["profit_factor"], targets["profit_factor"],
                          m["profit_factor"] >= targets["profit_factor"]),
        "trade_count": (m["trade_count"], targets["trade_count"],
                        m["trade_count"] >= targets["trade_count"]),
        "max_drawdown": (m["max_drawdown"], targets["max_drawdown"],
                         m["max_drawdown"] >= targets["max_drawdown"]),
    }
    if holdout_luck_bar is not None:
        checks["above_noise"] = (m["sharpe"], holdout_luck_bar,
                                 m["sharpe"] >= holdout_luck_bar)
    checks = {k: {"value": v, "target": t, "pass": bool(ok)} for k, (v, t, ok) in checks.items()}
    return all(c["pass"] for c in checks.values()), checks


def report(status, finalists, n_trades_shown=5):
    print("\n" + "=" * 72)
    print(f"  {status['inst']} {status['bar']}   {status['n_configs_tried']} strategies tried   "
          f"{status['run_date']}")
    print("=" * 72)
    print(f"\n  CHAMPION  {status['champion']}")
    print("  " + status["champion_describe"].replace("\n", "\n  "))
    v, h = status["champion_val"], status["champion_holdout"]
    print(f"\n  {'':<14}{'validation (picked here)':>26}{'holdout (confirms it)':>26}")
    for key, fmt in (("total_return", "{:+.2%}"), ("sharpe", "{:+.2f}"),
                     ("profit_factor", "{:.2f}"), ("max_drawdown", "{:.2%}"),
                     ("trade_count", "{:.0f}"), ("win_rate", "{:.1%}")):
        vs = fmt.format(v[key]) if v[key] is not None else "n/a"
        hs = fmt.format(h[key]) if h[key] is not None else "n/a"
        print(f"  {key:<14}{vs:>26}{hs:>26}")

    print(f"\n  luck bar (best of {status['n_configs_tried']} on noise): "
          f"Sharpe {status['luck_bar_sharpe']:+.2f}  -> champion "
          f"{'CLEARS it' if status['beats_luck_bar'] else 'DOES NOT clear it'}")
    b = status["baseline"]
    print(f"  baseline ({status['baseline_name']}) on the same holdout: "
          f"return {b['total_return']:+.2%}  sharpe {b['sharpe']:+.2f}")

    print("\n  how the champion's holdout trades ended:")
    for kind, s in sorted(status["champion_exit_breakdown"].items(),
                          key=lambda kv: -kv[1]["n"]):
        print(f"      {kind:<14} {s['n']:>4} trades   avg {s['avg_pnl']*100:+.3f}%")

    print("\n  target check:")
    for k, c in status["target_checks"].items():
        mark = "PASS" if c["pass"] else "fail"
        note = "  <- noise floor for a best-of-k pick" if k == "above_noise" else ""
        print(f"      {mark}  {k:<14} {c['value']:.3f}  (need {c['target']:.3f}){note}")
    print(f"\n  ==> {'SATISFIED' if status['satisfied'] else 'NOT satisfied yet — keep searching'}")

    champ_trades = finalists[0].get("holdout_trades", [])
    if champ_trades:
        print(f"\n  last {min(n_trades_shown, len(champ_trades))} holdout trades, with reasoning:")
        for t in champ_trades[-n_trades_shown:]:
            print(f"\n    {t['side']:<5} {t['entry_px']:.2f} -> {t['exit_px']:.2f}   "
                  f"{t['pnl']*100:+.3f}% net over {t['bars_held']} bars")
            print(f"      in : {t['entry_reason']}")
            print(f"      out: {t['exit_reason']}")

    print("\n  runners-up on validation:")
    for r in status["top"][1:]:
        print(f"      val sharpe {r['val']['sharpe']:+6.2f}  holdout {r['holdout']['sharpe']:+6.2f}"
              f"   {r['name']}")
    print()


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
def _selftest():
    print("alpha_ml/search.py self-test — full search on synthetic data\n")
    market = synthetic_market(n=4000, planted=True)
    status, finalists = run_search("SELFTEST", "5m", market, window=48,
                                   max_configs=40, min_trades=5, top_k=3,
                                   seed=0, persist=False)
    report(status, finalists, n_trades_shown=2)

    assert status["n_configs_tried"] == 40
    assert status["champion_val"]["trade_count"] >= 5
    assert status["champion"] == status["top"][0]["name"], "champion is not the validation winner"
    # The champion must be the VALIDATION winner even when a runner-up posts a
    # better holdout — that ordering is the whole point of the three-way split.
    best_val = max(status["top"], key=lambda r: r["val"]["sharpe"])
    assert best_val["name"] == status["champion"], "ranking did not come from validation"
    assert set(status["target_checks"]) == set(DEFAULT_TARGETS) | {"above_noise"}
    assert isinstance(status["satisfied"], bool)
    assert status["luck_bar_sharpe"] > 0
    # More candidates ranked over the same bars must raise the noise floor —
    # that is the multiple-testing penalty doing its job.
    assert luck_bar(1000, 400, 288 * 365) > luck_bar(1000, 10, 288 * 365) > 0
    assert luck_bar(20000, 400, 288 * 365) < luck_bar(1000, 400, 288 * 365), \
        "more bars must lower the noise floor"

    trades = finalists[0]["holdout_trades"]
    assert all(t["entry_reason"] and t["exit_reason"] for t in trades), \
        "a holdout trade is missing its reasoning"

    print("  champion is the validation winner, not the holdout winner ✓")
    print("  luck bar computed ✓   targets evaluated ✓   every trade explained ✓")
    print("\n  PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="5m")
    ap.add_argument("--bars", type=int, default=8000)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--max-hold", type=int, default=12)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--min-trades", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-configs", type=int, default=None,
                    help="sample this many from the full grid (default: try all)")
    ap.add_argument("--seed", type=int, default=None,
                    help="grid-sampling seed (default: derived from today's date, so "
                         "each day explores a different slice)")
    ap.add_argument("--target-sharpe", type=float, default=DEFAULT_TARGETS["sharpe"])
    ap.add_argument("--target-profit-factor", type=float, default=DEFAULT_TARGETS["profit_factor"])
    ap.add_argument("--target-trades", type=int, default=DEFAULT_TARGETS["trade_count"])
    ap.add_argument("--target-max-dd", type=float, default=DEFAULT_TARGETS["max_drawdown"])
    args = ap.parse_args()

    if args.selftest:
        _selftest()
    else:
        from data import fetch_ohlcv
        seed = args.seed if args.seed is not None else date.today().toordinal()
        market = fetch_ohlcv(args.inst, args.bar, args.bars)
        if market.empty:
            raise SystemExit("no market data (network?) — try --selftest for a synthetic run")
        status, finalists = run_search(
            args.inst, args.bar, market, fee=args.fee, train_frac=args.train_frac,
            val_frac=args.val_frac, min_trades=args.min_trades, top_k=args.top_k,
            max_configs=args.max_configs, window=args.window, max_hold=args.max_hold,
            seed=seed,
            targets={"sharpe": args.target_sharpe,
                     "profit_factor": args.target_profit_factor,
                     "trade_count": args.target_trades,
                     "max_drawdown": args.target_max_dd})
        report(status, finalists)
        print(json.dumps({"satisfied": status["satisfied"],
                          "champion": status["champion"]}, indent=2))
