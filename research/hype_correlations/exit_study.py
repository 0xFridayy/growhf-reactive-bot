#!/usr/bin/env python3
"""Trade-level entry/TP/SL study for the top-3 HYPE trigger strategy.

Signals (from backtest.py, evaluated at the 00:00 UTC daily close):
    1. dip:       HYPE down >15% over the past 7 days
    2. fund_flip: daily funding flipped positive -> negative
    3. vix_drop:  VIX down >10% over the past 5 days

Entry: open of the first 1h bar after the signal close (00:00 UTC).
One position at a time; signals while in a position are ignored.

For every entry this script:
  - tracks max favorable / adverse excursion (MFE/MAE) over 5 days
  - simulates a grid of exits: time stop x take-profit x stop-loss,
    intrabar on 1h highs/lows, SL assumed to fill first if TP and SL
    are touched in the same hour (conservative), 10 bp round-trip cost.

Grid results are ranked on the IS window (first 70% of days) only and
then shown with their OOS performance, to keep exit selection honest.

Outputs: results/exit_grid.csv, results/trade_list.csv,
         results/mfe_mae.csv, results/mfe_mae.png
"""
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")

COST_RT = 0.0010
HORIZON_H = 120  # excursion tracking window (5 days)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True,
    "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
    "font.size": 10,
})


def signals():
    d = pd.read_csv(os.path.join(DATA, "hype_daily_dataset.csv"),
                    parse_dates=["date"]).set_index("date")
    c = d.h_close
    t1 = c.pct_change(7) < -0.15
    t2 = (d.funding_sum < 0) & (d.funding_sum.shift(1) > 0)
    t3 = d.vix.ffill().pct_change(5) < -0.10
    sig = pd.DataFrame({"dip": t1, "fund_flip": t2, "vix_drop": t3})
    sig["any"] = sig.any(axis=1)
    return sig


def hourly():
    h = pd.read_csv(os.path.join(DATA, "hype_1h.csv"), parse_dates=["date"])
    return h.sort_values("date").set_index("date")


def build_trades(sig, h1):
    """Entry rows: first 1h bar at 00:00 UTC after each signal day, single
    position, no re-entry until the tracking horizon has passed."""
    trades = []
    busy_until = pd.Timestamp.min
    for day, row in sig[sig["any"]].iterrows():
        entry_ts = day + pd.Timedelta(days=1)  # 00:00 UTC bar after close
        if entry_ts <= busy_until:
            continue
        if entry_ts not in h1.index:
            continue
        win = h1.loc[entry_ts:entry_ts + pd.Timedelta(hours=HORIZON_H - 1)]
        if len(win) < 24:
            continue
        entry_px = win.iloc[0].open
        trig = "+".join([t for t in ("dip", "fund_flip", "vix_drop") if row[t]])
        trades.append({"entry_ts": entry_ts, "entry_px": entry_px,
                       "trigger": trig, "window": win})
        busy_until = entry_ts + pd.Timedelta(hours=HORIZON_H - 1)
    return trades


def excursions(trades):
    rows = []
    for t in trades:
        w, px = t["window"], t["entry_px"]
        path_hi = w.high.cummax() / px - 1
        path_lo = w.low.cummin() / px - 1
        rec = {"entry_ts": t["entry_ts"], "trigger": t["trigger"],
               "entry_px": px}
        for hh in (24, 48, 72, 120):
            sub = w.iloc[:hh]
            rec[f"mfe_{hh}h"] = sub.high.max() / px - 1
            rec[f"mae_{hh}h"] = sub.low.min() / px - 1
            rec[f"ret_{hh}h"] = (sub.close.iloc[-1] / px - 1
                                 if len(sub) >= hh * 0.9 else np.nan)
        # hours until the trade first reaches +5% / -5%
        up5 = np.argmax(path_hi.values >= 0.05) if (path_hi >= 0.05).any() else np.nan
        dn5 = np.argmax(path_lo.values <= -0.05) if (path_lo <= -0.05).any() else np.nan
        rec["h_to_+5%"], rec["h_to_-5%"] = up5, dn5
        rows.append(rec)
    return pd.DataFrame(rows).set_index("entry_ts")


def simulate(trades, hold_h, tp, sl):
    """Return per-trade net returns for one exit rule."""
    out = []
    for t in trades:
        w, px = t["window"].iloc[:hold_h], t["entry_px"]
        ret, hrs, how = None, hold_h, "time"
        for i, (_, bar) in enumerate(w.iterrows()):
            hit_sl = sl is not None and bar.low <= px * (1 + sl)
            hit_tp = tp is not None and bar.high >= px * (1 + tp)
            if hit_sl:                      # conservative: SL fills first
                ret, hrs, how = sl, i + 1, "sl"
                break
            if hit_tp:
                ret, hrs, how = tp, i + 1, "tp"
                break
        if ret is None:
            ret = w.close.iloc[-1] / px - 1
        out.append({"entry_ts": t["entry_ts"], "trigger": t["trigger"],
                    "ret": ret - COST_RT, "hours": hrs, "exit": how})
    return pd.DataFrame(out)


def grid_search(trades, split_ts):
    rows = []
    grid_hold = (24, 48, 72, 120)
    grid_tp = (None, 0.05, 0.08, 0.12)
    grid_sl = (None, -0.04, -0.06, -0.08, -0.10)
    for hold in grid_hold:
        for tp in grid_tp:
            for sl in grid_sl:
                tr = simulate(trades, hold, tp, sl)
                isw = tr[tr.entry_ts < split_ts]
                oos = tr[tr.entry_ts >= split_ts]
                if len(isw) < 10:
                    continue

                def stats(x, pre):
                    if len(x) == 0:
                        return {}
                    eq = (1 + x.ret).prod()
                    days = max((x.entry_ts.max() - x.entry_ts.min()).days, 1)
                    shp = (x.ret.mean() / x.ret.std()
                           * np.sqrt(365 * 24 / x.hours.mean())
                           if x.ret.std() > 0 else np.nan)
                    return {f"{pre}_n": len(x),
                            f"{pre}_avg": x.ret.mean(),
                            f"{pre}_win": (x.ret > 0).mean(),
                            f"{pre}_total": eq - 1,
                            f"{pre}_sharpe": shp,
                            f"{pre}_worst": x.ret.min()}
                rec = {"hold_h": hold,
                       "tp": tp if tp is not None else np.nan,
                       "sl": sl if sl is not None else np.nan}
                rec.update(stats(isw, "is"))
                rec.update(stats(oos, "oos"))
                rows.append(rec)
    g = pd.DataFrame(rows)
    g = g.sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    return g


def chart_mfe_mae(exc):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {"dip": ORANGE, "fund_flip": BLUE, "vix_drop": AQUA}
    for trig in ("dip", "fund_flip", "vix_drop"):
        m = exc.trigger.str.contains(trig)
        ax.scatter(exc.loc[m, "mae_72h"] * 100, exc.loc[m, "mfe_72h"] * 100,
                   s=42, color=colors[trig], label=trig.replace("_", " "),
                   edgecolors=SURFACE, linewidths=1.2, alpha=0.9)
    xlo = exc.mae_72h.min() * 100 * 1.15
    ax.plot([0, xlo], [0, -xlo], color=BASE, lw=1, ls="--")
    ax.set_xlim(xlo, 0.5)
    ax.axvline(-6, color=MUTED, lw=1, ls=":")
    ax.text(-6, exc.mfe_72h.max() * 100, " SL -6%", color=MUTED,
            fontsize=8.5, ha="left", va="top")
    ax.axhline(5, color=MUTED, lw=1, ls=":")
    ax.text(xlo, 5, " TP +5%", color=MUTED, fontsize=8.5, va="bottom",
            ha="left")
    ax.set_xlabel("max adverse excursion in 72h after entry, %")
    ax.set_ylabel("max favorable excursion in 72h, %")
    ax.set_title("How far each trade ran for vs against you (72h window)",
                 loc="left", fontsize=12, color=INK, pad=14)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "mfe_mae.png"), dpi=150)
    plt.close(fig)


def main():
    sig = signals()
    h1 = hourly()
    trades = build_trades(sig, h1)
    n = len(sig)
    split_ts = sig.index[int(n * 0.7)]
    print(f"{len(trades)} non-overlapping entries "
          f"({sig['any'].sum()} signal days), IS/OOS split {split_ts.date()}")

    exc = excursions(trades)
    exc.round(4).to_csv(os.path.join(RES, "mfe_mae.csv"))
    agg = exc[["mfe_24h", "mae_24h", "mfe_72h", "mae_72h", "ret_72h"]]
    print("\n=== excursion summary (all trades) ===")
    print(agg.describe(percentiles=[.1, .25, .5, .75, .9]).round(4).to_string())

    print("\n=== excursions by trigger (medians) ===")
    for trig in ("dip", "fund_flip", "vix_drop"):
        m = exc.trigger.str.contains(trig)
        s = exc[m]
        print(f"{trig:10s} n={m.sum():3d}  mfe72 {s.mfe_72h.median():+.3f}  "
              f"mae72 {s.mae_72h.median():+.3f}  ret72 {s.ret_72h.median():+.3f}")

    g = grid_search(trades, split_ts)
    g.round(4).to_csv(os.path.join(RES, "exit_grid.csv"), index=False)
    print("\n=== top 12 exit rules by IS Sharpe (with OOS check) ===")
    cols = ["hold_h", "tp", "sl", "is_n", "is_avg", "is_win", "is_total",
            "is_sharpe", "oos_n", "oos_avg", "oos_win", "oos_total",
            "oos_sharpe", "oos_worst"]
    print(g[cols].head(12).round(3).to_string(index=False))

    print("\n=== baseline: pure 24h time exit, no TP/SL ===")
    base = g[(g.hold_h == 24) & g.tp.isna() & g.sl.isna()]
    print(base[cols].round(3).to_string(index=False))

    # trade list under the headline rule (filled in after inspection):
    tr = simulate(trades, 72, 0.08, -0.06)
    tr.round(4).to_csv(os.path.join(RES, "trade_list.csv"), index=False)
    chart_mfe_mae(exc)


if __name__ == "__main__":
    main()
