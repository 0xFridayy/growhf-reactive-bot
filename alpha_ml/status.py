"""
alpha_ml/status.py — persisted training/backtest status.

Every alpha_ml/train.py run writes the latest status to alpha_status.json AND
appends a row to alpha_status_history.csv, so history survives bot restarts
even though only the latest snapshot is shown on demand. okx_tele_bot.py's
/alpha_status command reads this module ONLY (json/csv stdlib, zero heavy
deps) so the live Telegram process never needs torch/xgboost installed just
to answer "how's training going".

Deps: stdlib only
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

STATUS_JSON = Path(__file__).with_name("alpha_status.json")
STATUS_HISTORY_CSV = Path(__file__).with_name("alpha_status_history.csv")
# search.py's daily entry/exit strategy search writes these three.
SEARCH_JSON = Path(__file__).with_name("search_status.json")
LEADERBOARD_CSV = Path(__file__).with_name("leaderboard.csv")
CHAMPION_TRADES_CSV = Path(__file__).with_name("champion_trades.csv")
WIB = ZoneInfo("Asia/Jakarta")

_BT_SUFFIXES = [
    "total_return", "sharpe", "sortino", "max_drawdown", "profit_factor",
    "trade_count", "win_rate", "turnover", "fees_paid", "gated_entries",
]

HISTORY_FIELDS = [
    "ts_utc", "inst", "bar", "search_frac",
    "xgb_auc", "xgb_acc", "xgb_expectancy_bps",
    "ddqn_episodes", "ddqn_avg_reward", "overfit_flag",
    *[f"bt_search_{s}" for s in _BT_SUFFIXES],
    *[f"bt_holdout_{s}" for s in _BT_SUFFIXES],
]


def write_status(status: dict):
    """status may include any keys (e.g. by_regime) — only HISTORY_FIELDS are
    appended to the CSV trend log; everything is kept in the latest JSON.
    Mutates and returns `status` with ts_utc filled in, so callers that want
    to format/display it (e.g. train.py) see the same timestamp that was
    persisted, without a second read_status() round-trip."""
    status.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())

    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_JSON, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)

    is_new = not STATUS_HISTORY_CSV.exists()
    with open(STATUS_HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(status)
    return status


def read_status():
    if not STATUS_JSON.exists():
        return None
    with open(STATUS_JSON, encoding="utf-8") as f:
        return json.load(f)


def read_history(limit=20):
    if not STATUS_HISTORY_CSV.exists():
        return []
    with open(STATUS_HISTORY_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:]


def _fmt(status, key, spec=None, default="n/a"):
    v = status.get(key)
    if v is None or v == "":
        return default
    try:
        return spec.format(float(v)) if spec else str(v)
    except (TypeError, ValueError):
        return str(v)


def format_telegram(status=None):
    """Telegram-ready HTML status message (matches okx_tele_bot.py's parse_mode)."""
    status = status if status is not None else read_status()
    if status is None:
        # Deliberately points at Actions, not at this host: training needs
        # torch/xgboost/sklearn, which the live bot never installs (that's the
        # whole reason this module is stdlib-only). Telling you to run it here
        # would just produce an ImportError.
        return ("🤖 <b>Alpha ML status</b>\n"
                "No training run yet.\n\n"
                "Training runs on GitHub Actions, not on this machine — "
                "start one from the repo's <b>Actions</b> tab → "
                "<b>alpha_ml training</b> → <b>Run workflow</b>. "
                "It reports back here when it finishes.\n\n"
                "Meanwhile <code>/alpha_search</code> has the daily entry/exit "
                "strategy results, which run separately.")

    when = "n/a"
    ts_utc = status.get("ts_utc")
    if ts_utc:
        try:
            when = datetime.fromisoformat(ts_utc).astimezone(WIB).strftime("%H:%M WIB, %d %b")
        except ValueError:
            when = ts_utc

    search_frac = status.get("search_frac")
    split_s = (f"{search_frac:.0%}/{1 - search_frac:.0%}"
               if isinstance(search_frac, (int, float)) else "n/a")

    lines = [
        "🤖 <b>Alpha ML status</b>",
        f"{status.get('inst', 'n/a')} {status.get('bar', '')}   "
        f"(search/holdout split: {split_s})",
        "",
    ]
    # train.py --selftest runs on synthetic data and no longer persists status,
    # but an old status file (or a hand-run selftest) can still be sitting here.
    # Say so loudly rather than letting fake numbers read as a BTC result.
    if str(status.get("inst", "")).upper() == "SELFTEST":
        lines += [
            "🧪 <b>This is the synthetic self-test, not a real run.</b>",
            "Every number below comes from generated data — ignore it. "
            "Kick off a real run from the alpha_ml training workflow.",
            "",
        ]
    lines += [
        f"XGBoost AUC: {_fmt(status, 'xgb_auc', '{:.3f}')}   "
        f"acc: {_fmt(status, 'xgb_acc', '{:.3f}')}",
        f"DDQN avg reward: {_fmt(status, 'ddqn_avg_reward', '{:+.4f}')}   "
        f"episode: {_fmt(status, 'ddqn_episodes')}",
        "",
        "<b>Search (in-sample — expect this to flatter the policy):</b>",
        f"  PnL: {_fmt(status, 'bt_search_total_return', '{:+.2%}')}   "
        f"Sharpe: {_fmt(status, 'bt_search_sharpe', '{:+.2f}')}   "
        f"Sortino: {_fmt(status, 'bt_search_sortino', '{:+.2f}')}",
        f"  Max DD: {_fmt(status, 'bt_search_max_drawdown', '{:.2%}')}   "
        f"Trades: {_fmt(status, 'bt_search_trade_count')}   "
        f"Win rate: {_fmt(status, 'bt_search_win_rate', '{:.1%}')}",
        "",
        "<b>Holdout (never seen in training — trust this number):</b>",
        f"  PnL: {_fmt(status, 'bt_holdout_total_return', '{:+.2%}')}   "
        f"Sharpe: {_fmt(status, 'bt_holdout_sharpe', '{:+.2f}')}   "
        f"Sortino: {_fmt(status, 'bt_holdout_sortino', '{:+.2f}')}",
        f"  Max DD: {_fmt(status, 'bt_holdout_max_drawdown', '{:.2%}')}   "
        f"Profit factor: {_fmt(status, 'bt_holdout_profit_factor', '{:.2f}')}",
        f"  Trades: {_fmt(status, 'bt_holdout_trade_count')}   "
        f"Win rate: {_fmt(status, 'bt_holdout_win_rate', '{:.1%}')}",
        f"  Turnover: {_fmt(status, 'bt_holdout_turnover', '{:.1f}')}   "
        f"Fees paid: {_fmt(status, 'bt_holdout_fees_paid', '{:.2%}')}   "
        f"Gated entries: {_fmt(status, 'bt_holdout_gated_entries')}",
    ]
    if status.get("overfit_flag"):
        lines.append("")
        lines.append("⚠️ <b>Overfit warning</b>: search Sharpe far exceeds holdout Sharpe — "
                     "same collapse DailyScraper/NEOBDM's DDQN showed. Don't trade this yet.")
    lines.append("")
    lines.append(f"Last checkpoint: {when}")

    by_regime = status.get("by_regime_holdout")
    if by_regime:
        lines.append("")
        lines.append("<b>Holdout by regime:</b>")
        for reg, m in by_regime.items():
            ret = m.get("total_return")
            sharpe = m.get("sharpe")
            ret_s = f"{ret:+.2%}" if isinstance(ret, (int, float)) else "n/a"
            sharpe_s = f"{sharpe:+.2f}" if isinstance(sharpe, (int, float)) else "n/a"
            lines.append(f"  {reg}: return {ret_s}, sharpe {sharpe_s}")
    return "\n".join(lines)


# ======================================================================
# DAILY STRATEGY SEARCH — written by search.py, read here by the bot
# ======================================================================
LEADERBOARD_FIELDS = [
    "run_date", "ts_utc", "inst", "bar", "rank", "strategy",
    "val_sharpe", "val_return", "val_trades",
    "hold_sharpe", "hold_return", "hold_trades", "hold_profit_factor",
    "hold_max_drawdown", "luck_bar_sharpe", "satisfied",
]


def write_search_status(status: dict):
    SEARCH_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SEARCH_JSON, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)
    return status


def read_search():
    if not SEARCH_JSON.exists():
        return None
    with open(SEARCH_JSON, encoding="utf-8") as f:
        return json.load(f)


def append_leaderboard(status: dict, finalists):
    """One row per finalist per run. Rows accumulate forever: the point is to
    see whether a strategy keeps winning across days and fresh candles, which
    is the only evidence that separates an edge from a lucky slice."""
    is_new = not LEADERBOARD_CSV.exists()
    with open(LEADERBOARD_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        for rank, r in enumerate(finalists, start=1):
            m, h = r["metrics"], r.get("holdout", {})
            w.writerow({
                "run_date": status.get("run_date"), "ts_utc": status.get("ts_utc"),
                "inst": status.get("inst"), "bar": status.get("bar"),
                "rank": rank, "strategy": r["strategy"].name,
                "val_sharpe": m.get("sharpe"), "val_return": m.get("total_return"),
                "val_trades": m.get("trade_count"),
                "hold_sharpe": h.get("sharpe"), "hold_return": h.get("total_return"),
                "hold_trades": h.get("trade_count"),
                "hold_profit_factor": h.get("profit_factor"),
                "hold_max_drawdown": h.get("max_drawdown"),
                "luck_bar_sharpe": status.get("luck_bar_sharpe"),
                "satisfied": status.get("satisfied"),
            })


def read_leaderboard(limit=None):
    if not LEADERBOARD_CSV.exists():
        return []
    with open(LEADERBOARD_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:] if limit else rows


def holdout_use_count(strategy_name):
    """How many times this exact strategy has already been confirmed against a
    holdout. Every extra confirmation makes that holdout a little less honest,
    so the report shows the count instead of hiding it."""
    return sum(1 for r in read_leaderboard() if r.get("strategy") == strategy_name)


def read_trades(limit=10):
    if not CHAMPION_TRADES_CSV.exists():
        return []
    with open(CHAMPION_TRADES_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:] if limit else rows


def _f(row, key, default=None):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def format_search_telegram(status=None):
    """Telegram-ready summary of the latest daily strategy search."""
    status = status if status is not None else read_search()
    if status is None:
        return ("🔎 <b>Strategy search</b>\n"
                "No search run yet. It runs daily on GitHub Actions, or locally with\n"
                "<code>py alpha_ml/search.py --inst BTC-USDT-SWAP --bar 5m</code>.")

    when = status.get("ts_utc", "")
    try:
        when = datetime.fromisoformat(when).astimezone(WIB).strftime("%H:%M WIB, %d %b")
    except ValueError:
        pass

    v = status.get("champion_val", {}) or {}
    h = status.get("champion_holdout", {}) or {}
    b = status.get("baseline", {}) or {}

    def num(d, k, spec, default="n/a"):
        val = d.get(k)
        return spec.format(val) if isinstance(val, (int, float)) else default

    lines = [
        "🔎 <b>Strategy search</b>",
        f"{status.get('inst', 'n/a')} {status.get('bar', '')}   "
        f"{status.get('n_configs_tried', '?')} strategies tried   {when}",
        "",
        "<b>Champion</b> (picked on validation, confirmed on holdout):",
        f"<code>{status.get('champion', 'n/a')}</code>",
        "",
        "<b>Holdout — the number that counts:</b>",
        f"  PnL: {num(h, 'total_return', '{:+.2%}')}   "
        f"Sharpe: {num(h, 'sharpe', '{:+.2f}')}   "
        f"PF: {num(h, 'profit_factor', '{:.2f}')}",
        f"  Max DD: {num(h, 'max_drawdown', '{:.2%}')}   "
        f"Trades: {num(h, 'trade_count', '{:.0f}')}   "
        f"Win rate: {num(h, 'win_rate', '{:.1%}')}",
        f"  Avg hold: {num(h, 'avg_bars_held', '{:.1f}')} bars",
        "",
        f"Validation Sharpe: {num(v, 'sharpe', '{:+.2f}')}   "
        f"vs luck bar {num(status, 'luck_bar_sharpe', '{:+.2f}')}",
        f"Baseline (no ML, no stops): PnL {num(b, 'total_return', '{:+.2%}')}, "
        f"Sharpe {num(b, 'sharpe', '{:+.2f}')}",
    ]

    breakdown = status.get("champion_exit_breakdown") or {}
    if breakdown:
        lines.append("")
        lines.append("<b>How trades ended:</b>")
        for kind, s in sorted(breakdown.items(), key=lambda kv: -kv[1].get("n", 0)):
            lines.append(f"  {kind}: {s.get('n', 0)} trades, "
                         f"avg {s.get('avg_pnl', 0.0) * 100:+.3f}%")

    checks = status.get("target_checks") or {}
    if checks:
        lines.append("")
        lines.append("<b>Your bar:</b>")
        for k, c in checks.items():
            mark = "✅" if c.get("pass") else "❌"
            spec = "{:.0f}" if k == "trade_count" else ("{:.2%}" if k == "max_drawdown" else "{:.2f}")
            got, want = c.get("value"), c.get("target")
            got_s = spec.format(got) if isinstance(got, (int, float)) else str(got)
            want_s = spec.format(want) if isinstance(want, (int, float)) else str(want)
            label = "sharpe vs noise floor" if k == "above_noise" else k
            lines.append(f"  {mark} {label}: {got_s} (need {want_s})")

    lines.append("")
    if status.get("satisfied"):
        lines.append("🎯 <b>SATISFIED</b> — every target cleared on holdout.")
    else:
        lines.append("🔁 <b>Not there yet</b> — searching again tomorrow.")
    if not status.get("beats_luck_bar", True):
        lines.append("⚠️ Champion is below the luck bar: with this many tries, "
                     "noise alone would score that well. Treat it as nothing found.")
    uses = status.get("champion_holdout_uses")
    if isinstance(uses, int) and uses > 3:
        lines.append(f"⚠️ This strategy has now been holdout-tested {uses} times — "
                     f"that holdout is wearing out as an honest test.")
    lines.append("")
    lines.append("Use <code>/alpha_trades</code> to see the logic behind each trade.")
    return "\n".join(lines)


def format_trades_telegram(limit=8):
    """The per-trade audit trail: what was entered, why, how it ended, why."""
    trades = read_trades(limit=limit)
    if not trades:
        return ("🧾 <b>Trade log</b>\nNo trades logged yet — run the strategy search "
                "first (<code>/alpha_search</code>).")

    status = read_search() or {}
    lines = [
        "🧾 <b>Trade log</b> — champion strategy, holdout period",
        f"<code>{status.get('champion', 'n/a')}</code>",
        f"showing the last {len(trades)} trades",
        "",
    ]
    for t in trades:
        pnl = _f(t, "pnl", 0.0)
        mark = "🟢" if pnl > 0 else "🔴"
        entry_px, exit_px = _f(t, "entry_px"), _f(t, "exit_px")
        px = (f"{entry_px:,.2f} → {exit_px:,.2f}"
              if entry_px is not None and exit_px is not None else "n/a")
        when = str(t.get("entry_ts", ""))[:16]
        lines.append(f"{mark} <b>{t.get('side', '?').upper()}</b>  {px}   "
                     f"<b>{pnl * 100:+.2f}%</b>  ({t.get('bars_held', '?')} bars)")
        lines.append(f"   <i>{when}</i>  regime {t.get('regime', 'n/a')}")
        lines.append(f"   ▸ in : {t.get('entry_reason', 'n/a')}")
        lines.append(f"   ▸ out: {t.get('exit_reason', 'n/a')}")
        lines.append("")
    total = sum(_f(t, "pnl", 0.0) for t in trades)
    lines.append(f"Net over these {len(trades)}: <b>{total * 100:+.2f}%</b> "
                 f"(after fees, unit size)")
    return "\n".join(lines)


if __name__ == "__main__":
    import tempfile

    print("alpha_ml/status.py self-test\n")
    # Redirect the module's paths at a scratch directory FIRST. This self-test
    # writes a fake BTC-USDT-SWAP status with a healthy-looking Sharpe; run
    # against the live paths it would silently replace the real checkpoint with
    # plausible fiction, and unlike a SELFTEST row nothing downstream could
    # tell. No self-test in this package writes where the bot reads.
    _tmp = Path(tempfile.mkdtemp(prefix="alpha_status_selftest_"))
    STATUS_JSON = _tmp / "alpha_status.json"
    STATUS_HISTORY_CSV = _tmp / "alpha_status_history.csv"
    print(f"  writing to scratch dir {_tmp}\n")

    write_status({
        "inst": "BTC-USDT-SWAP", "bar": "5m", "search_frac": 0.7,
        "xgb_auc": 0.612, "xgb_acc": 0.548, "xgb_expectancy_bps": 4.2,
        "ddqn_episodes": 18400, "ddqn_avg_reward": 0.0091, "overfit_flag": False,
        "bt_search_total_return": 0.084, "bt_search_sharpe": 1.9, "bt_search_sortino": 2.4,
        "bt_search_max_drawdown": -0.084, "bt_search_profit_factor": 1.63,
        "bt_search_trade_count": 1284, "bt_search_win_rate": 0.547,
        "bt_search_turnover": 3120.0, "bt_search_fees_paid": 0.031, "bt_search_gated_entries": 92,
        "bt_holdout_total_return": 0.021, "bt_holdout_sharpe": 0.6, "bt_holdout_sortino": 0.8,
        "bt_holdout_max_drawdown": -0.061, "bt_holdout_profit_factor": 1.08,
        "bt_holdout_trade_count": 341, "bt_holdout_win_rate": 0.502,
        "bt_holdout_turnover": 890.0, "bt_holdout_fees_paid": 0.009, "bt_holdout_gated_entries": 28,
        "by_regime_search": {"trending": {"total_return": 0.05, "sharpe": 2.1},
                             "ranging": {"total_return": 0.02, "sharpe": 1.1}},
        "by_regime_holdout": {"trending": {"total_return": 0.015, "sharpe": 0.9},
                              "ranging": {"total_return": 0.003, "sharpe": 0.2}},
    })
    loaded = read_status()
    assert loaded is not None and loaded["inst"] == "BTC-USDT-SWAP"
    hist = read_history()
    assert len(hist) >= 1
    msg = format_telegram()
    print(msg)
    assert "XGBoost AUC: 0.612" in msg
    assert "Trades: 341" in msg
    assert "Holdout (never seen in training" in msg
    print("\n  PASS")
