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
WIB = ZoneInfo("Asia/Jakarta")

HISTORY_FIELDS = [
    "ts_utc", "inst", "bar",
    "xgb_auc", "xgb_acc", "xgb_expectancy_bps",
    "ddqn_episodes", "ddqn_avg_reward",
    "bt_total_return", "bt_sharpe", "bt_sortino", "bt_max_drawdown",
    "bt_profit_factor", "bt_trade_count", "bt_win_rate",
    "bt_turnover", "bt_fees_paid", "bt_gated_entries",
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
        return ("🤖 <b>Alpha ML status</b>\n"
                "No training run yet. Kick one off with "
                "<code>py alpha_ml/train.py --inst BTC-USDT-SWAP --bar 5m</code>.")

    when = "n/a"
    ts_utc = status.get("ts_utc")
    if ts_utc:
        try:
            when = datetime.fromisoformat(ts_utc).astimezone(WIB).strftime("%H:%M WIB, %d %b")
        except ValueError:
            when = ts_utc

    lines = [
        "🤖 <b>Alpha ML status</b>",
        f"{status.get('inst', 'n/a')} {status.get('bar', '')}",
        "",
        f"XGBoost AUC: {_fmt(status, 'xgb_auc', '{:.3f}')}   "
        f"acc: {_fmt(status, 'xgb_acc', '{:.3f}')}",
        f"DDQN avg reward: {_fmt(status, 'ddqn_avg_reward', '{:+.4f}')}   "
        f"episode: {_fmt(status, 'ddqn_episodes')}",
        "",
        f"OOS PnL: {_fmt(status, 'bt_total_return', '{:+.2%}')}   "
        f"Sharpe: {_fmt(status, 'bt_sharpe', '{:+.2f}')}   "
        f"Sortino: {_fmt(status, 'bt_sortino', '{:+.2f}')}",
        f"Max DD: {_fmt(status, 'bt_max_drawdown', '{:.2%}')}   "
        f"Profit factor: {_fmt(status, 'bt_profit_factor', '{:.2f}')}",
        f"Trades: {_fmt(status, 'bt_trade_count')}   "
        f"Win rate: {_fmt(status, 'bt_win_rate', '{:.1%}')}",
        f"Turnover: {_fmt(status, 'bt_turnover', '{:.1f}')}   "
        f"Fees paid: {_fmt(status, 'bt_fees_paid', '{:.2%}')}   "
        f"Gated entries: {_fmt(status, 'bt_gated_entries')}",
        "",
        f"Last checkpoint: {when}",
    ]
    by_regime = status.get("by_regime")
    if by_regime:
        lines.append("")
        lines.append("<b>By regime:</b>")
        for reg, m in by_regime.items():
            ret = m.get("total_return")
            sharpe = m.get("sharpe")
            ret_s = f"{ret:+.2%}" if isinstance(ret, (int, float)) else "n/a"
            sharpe_s = f"{sharpe:+.2f}" if isinstance(sharpe, (int, float)) else "n/a"
            lines.append(f"  {reg}: return {ret_s}, sharpe {sharpe_s}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("alpha_ml/status.py self-test\n")
    write_status({
        "inst": "BTC-USDT-SWAP", "bar": "5m", "xgb_auc": 0.612, "xgb_acc": 0.548,
        "xgb_expectancy_bps": 4.2, "ddqn_episodes": 18400, "ddqn_avg_reward": 0.0091,
        "bt_total_return": 0.084, "bt_sharpe": 1.9, "bt_sortino": 2.4,
        "bt_max_drawdown": -0.084, "bt_profit_factor": 1.63, "bt_trade_count": 1284,
        "bt_win_rate": 0.547, "bt_turnover": 3120.0, "bt_fees_paid": 0.031,
        "bt_gated_entries": 92,
        "by_regime": {"trending": {"total_return": 0.05, "sharpe": 2.1},
                      "ranging": {"total_return": 0.02, "sharpe": 1.1}},
    })
    loaded = read_status()
    assert loaded is not None and loaded["inst"] == "BTC-USDT-SWAP"
    hist = read_history()
    assert len(hist) >= 1
    msg = format_telegram()
    print(msg)
    assert "XGBoost AUC: 0.612" in msg
    assert "Trades: 1284" in msg
    print("\n  PASS")
