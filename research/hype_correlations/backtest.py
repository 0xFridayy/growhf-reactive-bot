#!/usr/bin/env python3
"""HYPE trigger study: what precedes HYPE price up-moves?

Reads data/hype_daily_dataset.csv (+ hype_1h.csv), engineers candidate
trigger features, and runs:

  1. Information-coefficient table  - Spearman corr of every feature vs
     forward 1/3/7-day returns, with Benjamini-Hochberg FDR flags.
  2. Lead-lag cross-correlation     - do BTC/ETH/SOL/VIX moves lead HYPE?
  3. Event studies                  - ~20 binary triggers; conditional
     forward returns vs the unconditional baseline, t-stats, bootstrap CIs.
  4. Trigger strategy backtest      - top in-sample triggers evaluated
     out-of-sample (70/30 chronological split), with trading costs.
  5. Hourly spike-continuation      - does a >=3% 1h move on >=3x volume
     (this repo's bot signal) continue or fade on HYPE?

All features on day t use only information available by that day's close
(00:00 UTC). Forward returns start at the same close. Outputs land in
results/ (CSVs + PNG charts).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

RNG = np.random.default_rng(7)
COST_RT = 0.0010  # round-trip cost assumption for the strategy test (10 bp)

# --- chart chrome (light mode) ---
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10,
})


def bh_fdr(pvals, q=0.10):
    """Benjamini-Hochberg: returns boolean 'significant' mask."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    sig = np.zeros(len(p), dtype=bool)
    if ok.sum() == 0:
        return sig
    idx = np.argsort(p[ok])
    m = ok.sum()
    thresh = q * (np.arange(1, m + 1)) / m
    passed = np.where(np.sort(p[ok]) <= thresh)[0]
    if len(passed):
        cut = np.sort(p[ok])[passed.max()]
        sig[ok] = p[ok] <= cut
    return sig


def build_features():
    d = pd.read_csv(os.path.join(DATA, "hype_daily_dataset.csv"),
                    parse_dates=["date"]).set_index("date")
    f = pd.DataFrame(index=d.index)
    c = d.h_close

    # targets (forward, close-to-close)
    f["fwd1"] = c.shift(-1) / c - 1
    f["fwd3"] = c.shift(-3) / c - 1
    f["fwd7"] = c.shift(-7) / c - 1

    # own price action
    f["ret1"] = c.pct_change()
    f["ret3"] = c.pct_change(3)
    f["ret7"] = c.pct_change(7)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + up / dn)
    f["dist_20d_high"] = c / c.rolling(20).max() - 1
    f["drawdown_ath"] = c / c.cummax() - 1
    rng = (d.h_high - d.h_low) / c
    f["range_pct"] = rng
    f["vol_spike"] = d.h_qvol / d.h_qvol.rolling(14).mean()
    f["spot_vol_spike"] = d.h_spot_vol_all / d.h_spot_vol_all.rolling(14).mean()
    f["taker_buy_ratio"] = d.h_taker_buy_qvol / d.h_qvol

    # derivatives positioning
    f["funding_sum"] = d.funding_sum
    f["funding_chg"] = d.funding_sum.diff()
    f["oi_chg1"] = d.oi.pct_change()
    f["oi_chg3"] = d.oi.pct_change(3)
    f["top_ls_pos"] = d.top_ls_pos
    f["top_ls_pos_chg"] = d.top_ls_pos.diff()
    f["global_ls"] = d.global_ls
    f["taker_ls_vol"] = d.taker_ls_vol

    # cross-asset
    for tag in ("btc", "eth", "sol"):
        f[f"{tag}_ret1"] = d[f"{tag}_close"].pct_change()
        f[f"{tag}_ret3"] = d[f"{tag}_close"].pct_change(3)
    f["ethbtc_ret3"] = (d.eth_close / d.btc_close).pct_change(3)
    f["hype_vs_btc_7d"] = f.ret7 - f.btc_ret3 * 0  # placeholder replaced below
    f["hype_vs_btc_7d"] = c.pct_change(7) - d.btc_close.pct_change(7)

    # macro
    f["vix"] = d.vix
    f["vix_chg1"] = d.vix.pct_change()
    f["vix_chg5"] = d.vix.pct_change(5)

    # on-chain (BTC/ETH as market-wide proxies)
    f["btc_netflow_z"] = ((d.btc_ex_netflow_usd - d.btc_ex_netflow_usd.rolling(30).mean())
                          / d.btc_ex_netflow_usd.rolling(30).std())
    f["eth_netflow_z"] = ((d.eth_ex_netflow_usd - d.eth_ex_netflow_usd.rolling(30).mean())
                          / d.eth_ex_netflow_usd.rolling(30).std())
    f["btc_mvrv"] = d.btc_mvrv
    f["btc_adract_chg7"] = d.btc_adract.pct_change(7)

    f["dow"] = f.index.dayofweek
    return d, f


def ic_table(f):
    feats = [c for c in f.columns if c not in ("fwd1", "fwd3", "fwd7", "dow")]
    rows = []
    for feat in feats:
        row = {"feature": feat}
        for h in ("fwd1", "fwd3", "fwd7"):
            sub = f[[feat, h]].dropna()
            if len(sub) < 40:
                row[f"ic_{h}"], row[f"p_{h}"], row[f"n_{h}"] = np.nan, np.nan, len(sub)
                continue
            ic, p = stats.spearmanr(sub[feat], sub[h])
            row[f"ic_{h}"], row[f"p_{h}"], row[f"n_{h}"] = ic, p, len(sub)
        rows.append(row)
    t = pd.DataFrame(rows).set_index("feature")
    for h in ("fwd1", "fwd3", "fwd7"):
        t[f"fdr10_{h}"] = bh_fdr(t[f"p_{h}"].values)
    t.round(4).to_csv(os.path.join(RES, "ic_table.csv"))
    return t


def leadlag(f):
    rows = []
    for drv in ("btc_ret1", "eth_ret1", "sol_ret1", "vix_chg1"):
        for k in range(0, 6):
            sub = pd.concat([f[drv].shift(k), f.ret1], axis=1).dropna()
            r, p = stats.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
            rows.append({"driver": drv, "lag_days": k, "corr": r, "p": p, "n": len(sub)})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(RES, "leadlag.csv"), index=False)
    return t


def triggers(f):
    """Binary conditions evaluated at day t's close."""
    T = {
        # volume / flow
        "vol_spike>2x": f.vol_spike > 2,
        "vol_spike>3x": f.vol_spike > 3,
        "taker_buy_ratio>0.52": f.taker_buy_ratio > 0.52,
        "taker_buy_ratio<0.48": f.taker_buy_ratio < 0.48,
        # derivatives
        "funding_negative": f.funding_sum < 0,
        "funding_flip_neg->pos": (f.funding_sum > 0) & (f.funding_sum.shift(1) < 0),
        "funding_flip_pos->neg": (f.funding_sum < 0) & (f.funding_sum.shift(1) > 0),
        "oi_up>5%_px_up": (f.oi_chg1 > 0.05) & (f.ret1 > 0),
        "oi_up>5%_px_down": (f.oi_chg1 > 0.05) & (f.ret1 < 0),
        "oi_down>5%": f.oi_chg1 < -0.05,
        "top_ls_pos_rising": f.top_ls_pos_chg > 0.05,
        # price action
        "20d_breakout": f.dist_20d_high >= -0.001,
        "rsi<30": f.rsi14 < 30,
        "rsi>70": f.rsi14 > 70,
        "down>15%_in_7d": f.ret7 < -0.15,
        "up>15%_in_7d": f.ret7 > 0.15,
        "big_down_day<-8%": f.ret1 < -0.08,
        "big_up_day>8%": f.ret1 > 0.08,
        # cross-asset / macro
        "btc_up>2%": f.btc_ret1 > 0.02,
        "btc_down>2%": f.btc_ret1 < -0.02,
        "hype_lag_btc_7d<-10%": f.hype_vs_btc_7d < -0.10,
        "vix_drop>10%_5d": f.vix_chg5 < -0.10,
        "vix_spike>15%_5d": f.vix_chg5 > 0.15,
        "vix>25": f.vix > 25,
        "btc_exflow_z<-1": f.btc_netflow_z < -1,
        "btc_exflow_z>1": f.btc_netflow_z > 1,
    }
    return T


def event_study(f, T):
    rows = []
    base = {h: f[h].dropna() for h in ("fwd1", "fwd3", "fwd7")}
    for name, cond in T.items():
        r = {"trigger": name}
        for h in ("fwd1", "fwd3", "fwd7"):
            sel = f.loc[cond.fillna(False), h].dropna()
            r[f"n_{h}"] = len(sel)
            if len(sel) < 8:
                r[f"mean_{h}"] = r[f"win_{h}"] = r[f"t_{h}"] = r[f"p_{h}"] = np.nan
                continue
            rest = f.loc[~cond.fillna(False), h].dropna()
            tt = stats.ttest_ind(sel, rest, equal_var=False)
            r[f"mean_{h}"] = sel.mean()
            r[f"med_{h}"] = sel.median()
            r[f"win_{h}"] = (sel > 0).mean()
            r[f"t_{h}"] = tt.statistic
            r[f"p_{h}"] = tt.pvalue
            if h == "fwd1":
                boot = [RNG.choice(sel.values, len(sel), replace=True).mean()
                        for _ in range(4000)]
                r["ci_lo"], r["ci_hi"] = np.percentile(boot, [2.5, 97.5])
        rows.append(r)
    t = pd.DataFrame(rows).set_index("trigger")
    t["fdr10_fwd1"] = bh_fdr(t["p_fwd1"].values)
    t["fdr10_fwd3"] = bh_fdr(t["p_fwd3"].values)
    for h, b in base.items():
        t.attrs[f"baseline_{h}"] = b.mean()
    t.round(5).to_csv(os.path.join(RES, "event_study.csv"))
    return t


def strategy_test(f, T):
    """Rank triggers in-sample, evaluate out-of-sample. Long next day when
    trigger is true at today's close; flat otherwise; 10 bp round-trip."""
    n = len(f)
    split = int(n * 0.7)
    is_idx, oos_idx = f.index[:split], f.index[split:]
    ranking = []
    for name, cond in T.items():
        sel = f.loc[cond.fillna(False) & f.index.isin(is_idx), "fwd1"].dropna()
        if len(sel) >= 12:
            ranking.append((name, sel.mean(), len(sel)))
    ranking.sort(key=lambda x: -x[1])
    top = [r[0] for r in ranking[:3]]

    rows = []
    curves = {}
    for name in top + ["UNION_top3", "buy_hold"]:
        if name == "buy_hold":
            pos = pd.Series(True, index=f.index)
            cost = 0.0
        elif name == "UNION_top3":
            pos = pd.concat([T[t_].fillna(False) for t_ in top], axis=1).any(axis=1)
            cost = COST_RT
        else:
            pos = T[name].fillna(False)
            cost = COST_RT
        daily = f.fwd1.where(pos, 0.0).copy()
        entries = pos & ~pos.shift(1, fill_value=False)
        daily[entries] -= cost
        for label, idx in (("IS", is_idx), ("OOS", oos_idx)):
            sub = daily.loc[idx].dropna()
            act = pos.loc[idx]
            if sub.std() == 0:
                continue
            eq = (1 + sub).cumprod()
            sharpe = sub.mean() / sub.std() * np.sqrt(365)
            mdd = (eq / eq.cummax() - 1).min()
            rows.append({"rule": name, "window": label,
                         "days_in_mkt": int(act.sum()), "days": len(sub),
                         "total_ret": eq.iloc[-1] - 1, "sharpe": sharpe,
                         "max_dd": mdd})
        curves[name] = (1 + daily.fillna(0)).cumprod()
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(RES, "strategy_summary.csv"), index=False)
    return t, curves, top, f.index[split]


def spike_study():
    h = pd.read_csv(os.path.join(DATA, "hype_1h.csv"), parse_dates=["date"])
    h = h.sort_values("date").reset_index(drop=True)
    r1 = h.close.pct_change()
    volx = h.quote_volume / h.quote_volume.rolling(10).mean().shift(1)
    fwd = {k: h.close.shift(-k) / h.close - 1 for k in (1, 4, 24)}
    rows = []
    for name, cond in {
        "up_spike>=3%_vol>=3x": (r1 >= 0.03) & (volx >= 3),
        "down_spike<=-3%_vol>=3x": (r1 <= -0.03) & (volx >= 3),
        "up_spike>=3%_lowvol": (r1 >= 0.03) & (volx < 3),
        "down_spike<=-3%_lowvol": (r1 <= -0.03) & (volx < 3),
        "up_spike>=5%_vol>=3x": (r1 >= 0.05) & (volx >= 3),
        "down_spike<=-5%_vol>=3x": (r1 <= -0.05) & (volx >= 3),
    }.items():
        rec = {"event": name, "n": int(cond.sum())}
        for k in (1, 4, 24):
            sel = fwd[k][cond].dropna()
            rec[f"mean_fwd{k}h"] = sel.mean()
            rec[f"win_fwd{k}h"] = (sel > 0).mean()
            rec[f"t_fwd{k}h"] = (stats.ttest_1samp(sel, 0).statistic
                                 if len(sel) > 5 else np.nan)
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("event")
    t.round(5).to_csv(os.path.join(RES, "spike_study.csv"))
    return t


# ---------------- charts ----------------

def chart_ic(ic):
    t = ic.dropna(subset=["ic_fwd1"]).sort_values("ic_fwd1")
    fig, ax = plt.subplots(figsize=(8, 8.5))
    colors = [BLUE if v > 0 else RED for v in t.ic_fwd1]
    alpha = [1.0 if s else 0.35 for s in t.fdr10_fwd1]
    bars = ax.barh(t.index, t.ic_fwd1, color=colors, height=0.62)
    for b, a in zip(bars, alpha):
        b.set_alpha(a)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_xlabel("Spearman IC vs next-day HYPE return")
    ax.set_title("Which daily features correlate with HYPE's next-day return?",
                 loc="left", fontsize=12, color=INK, pad=26)
    ax.text(0, 1.008, "solid = significant at 10% FDR; faded = not significant",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "ic_fwd1.png"), dpi=150)
    plt.close(fig)


def chart_event(ev):
    t = ev.dropna(subset=["mean_fwd1"]).sort_values("mean_fwd1")
    base = t.attrs.get("baseline_fwd1", 0)
    fig, ax = plt.subplots(figsize=(8, 8.5))
    y = np.arange(len(t))
    for i, (_, row) in enumerate(t.iterrows()):
        col = BLUE if row.mean_fwd1 > base else RED
        a = 1.0 if row.p_fwd1 < 0.05 else 0.4
        ax.plot([row.ci_lo * 100, row.ci_hi * 100], [i, i], color=col, lw=2, alpha=a,
                solid_capstyle="round")
        ax.plot(row.mean_fwd1 * 100, i, "o", ms=6, color=col, alpha=a)
    for i, (_, row) in enumerate(t.iterrows()):
        ax.text(1.005, i, f"n={int(row.n_fwd1)}", va="center", fontsize=7.5,
                color=MUTED, transform=ax.get_yaxis_transform())
    ax.axvline(base * 100, color=BASE, lw=1, ls="--")
    ax.text(base * 100, -1.6, f"baseline {base*100:.2f}%/d", fontsize=8.5,
            color=INK2, ha="center")
    ax.set_yticks(y, t.index)
    ax.set_ylim(-2.2, len(t) - 0.5)
    ax.set_xlabel("mean next-day HYPE return after trigger, % (dot) with 95% bootstrap CI")
    ax.set_title("Event study: next-day HYPE return conditioned on each trigger",
                 loc="left", fontsize=12, color=INK, pad=26)
    ax.text(0, 1.008, "solid = p<0.05 vs other days (none survive 10% FDR); faded = not",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "event_study.png"), dpi=150)
    plt.close(fig)


def chart_equity(curves, top, split_date):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(curves["buy_hold"].index, curves["buy_hold"], color=ORANGE, lw=2,
            label="buy & hold HYPE")
    ax.plot(curves["UNION_top3"].index, curves["UNION_top3"], color=BLUE, lw=2,
            label="long next day after any top-3 trigger (10 bp cost)")
    ax.axvline(split_date, color=BASE, lw=1, ls="--")
    ax.text(split_date, ax.get_ylim()[1], " OOS →", fontsize=9, color=INK2,
            va="top")
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Trigger strategy vs buy & hold  (top-3 triggers chosen in-sample only)",
                 loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "equity.png"), dpi=150)
    plt.close(fig)


def chart_leadlag(ll):
    drvs = [("btc_ret1", "BTC daily return"), ("eth_ret1", "ETH daily return"),
            ("sol_ret1", "SOL daily return"), ("vix_chg1", "VIX daily change")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, (drv, label) in zip(axes, drvs):
        sub = ll[ll.driver == drv]
        for _, row in sub.iterrows():
            col = BLUE if row["corr"] > 0 else RED
            a = 1.0 if row.p < 0.05 else 0.35
            ax.plot([row.lag_days, row.lag_days], [0, row["corr"]], color=col,
                    lw=2.5, alpha=a, solid_capstyle="round")
            ax.plot(row.lag_days, row["corr"], "o", color=col, ms=5, alpha=a)
        ax.axhline(0, color=BASE, lw=1)
        ax.set_title(label, fontsize=9.5, color=INK2)
        ax.set_xlabel("lead (days)")
    axes[0].set_ylabel("corr with HYPE return")
    fig.suptitle("Does anything lead HYPE? correlation of lagged driver vs same-day HYPE return",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(RES, "leadlag.png"), dpi=150)
    plt.close(fig)


def main():
    d, f = build_features()
    print(f"dataset: {len(f)} days  {f.index.min().date()} -> {f.index.max().date()}")
    print(f"baseline daily ret: {f.fwd1.mean()*100:.3f}%  (ann. vol {f.ret1.std()*np.sqrt(365)*100:.0f}%)")

    ic = ic_table(f)
    ll = leadlag(f)
    T = triggers(f)
    ev = event_study(f, T)
    strat, curves, top, split_date = strategy_test(f, T)
    sp = spike_study()

    chart_ic(ic)
    chart_event(ev)
    chart_equity(curves, top, split_date)
    chart_leadlag(ll)

    print("\n=== top IC (fwd 1d, FDR-significant) ===")
    sig = ic[ic.fdr10_fwd1].sort_values("ic_fwd1", key=abs, ascending=False)
    print(sig[["ic_fwd1", "p_fwd1", "n_fwd1"]].round(4).to_string())
    print("\n=== event study (sorted by mean fwd1) ===")
    print(ev.sort_values("mean_fwd1", ascending=False)
          [["n_fwd1", "mean_fwd1", "win_fwd1", "t_fwd1", "p_fwd1", "fdr10_fwd1"]]
          .round(4).to_string())
    print(f"\nbaseline fwd1 mean: {ev.attrs['baseline_fwd1']*100:.3f}%")
    print("\n=== in-sample top-3 triggers ===", top)
    print(strat.round(3).to_string(index=False))
    print("\n=== hourly spike study ===")
    print(sp.round(4).to_string())


if __name__ == "__main__":
    main()
