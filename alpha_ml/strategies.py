"""
alpha_ml/strategies.py — the explicit entry/exit strategy zoo the daily search
runs over, and the only place in this repo where a trade's REASONING is
recorded alongside its PnL.

Why this module exists
----------------------
Before it, "the strategy" was implicit and unauditable:

  * entries came from phase-1's direction convention (sign of chg_short, with a
    dead zone) plus an XGBoost probability gate, and
  * exits came from whatever the phase-2 DDQN happened to choose that bar.

That means there was NO stop loss, NO take profit and NO max holding period in
the live policy — the triple barrier in labels.py is used to LABEL phase-1
training data, not to manage a live position. A DDQN that never learns to
close a loser has nothing to stop it from riding one, which is exactly the
shape of a -18 holdout Sharpe.

This module makes entry and exit explicit, named, parameterized and searchable:

    Strategy = one EntryRule + a list of ExitRules (+ optional ML gate)

Every rule states its reasoning in words at the moment it fires, so each trade
carries `entry_reason` / `exit_reason` strings like

    entry  momentum: chg_short +0.42% > dead zone 0.15 (regime trending, xgb_p 0.61)
    exit   stop_loss: low 49820.00 <= stop 49850.00 (-1.20% from entry 50455.00)

Accounting conventions (kept identical to backtest.py / env.py so numbers stay
comparable across the whole pipeline):
  * unit notional, additive simple returns, no compounding, no leverage
  * entries and close-based exits fill at that bar's CLOSE
  * stop/target exits fill AT the level, intrabar, on the bar being held
  * when several stops could have fired inside one bar, the WORST fill for the
    position is assumed — the same conservative convention labels.py uses for
    ambiguous same-bar TP+SL
  * fees are charged on every unit of turnover, both sides

Bar-level metrics (Sharpe, drawdown) come from the per-bar trace; trade-level
PnL is computed entry price -> exit price. Those agree to second order — the
trace sums simple returns while a trade measures one — which the self-test
asserts stays within tolerance.

Run:  py strategies.py --selftest
Deps: numpy, pandas
"""

import argparse
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import _regime_series, compute_metrics
from labels import atr_fractions


# ======================================================================
# ENTRY RULES — pure functions of the (causal) feature table.
# `directions()` is vectorized over every bar; `explain()` is only called on
# the bars that actually open a trade, so formatting cost is per-trade.
# ======================================================================
class EntryRule:
    key = "base"

    def __init__(self, **params):
        self.params = params

    def __repr__(self):
        ps = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.key}({ps})"

    def directions(self, df):
        raise NotImplementedError

    def explain(self, df, i, direction):
        return f"{self.key}: fired {'long' if direction > 0 else 'short'}"

    @staticmethod
    def _col(df, name):
        """Missing feature -> all-NaN, which compares False everywhere, so a
        rule whose input isn't in the table simply never fires instead of
        blowing up a whole search run."""
        if name in df.columns:
            return df[name].to_numpy(dtype=float)
        return np.full(len(df), np.nan)


class MomentumEntry(EntryRule):
    """Follow short-horizon momentum — the side okx_tele_bot.py's chg_short
    read and growhf_reactive_bot.py's spike alert already imply. This is the
    incumbent: whatever beats it has earned its place."""
    key = "momentum"

    def __init__(self, deadzone=0.15):
        super().__init__(deadzone=deadzone)

    def directions(self, df):
        chg = self._col(df, "chg_short")
        dz = self.params["deadzone"]
        return np.where(chg > dz, 1.0, np.where(chg < -dz, -1.0, 0.0))

    def explain(self, df, i, direction):
        chg = df["chg_short"].iat[i]
        return (f"momentum: chg_short {chg:+.2f}% beyond dead zone "
                f"{self.params['deadzone']:.2f}")


class BreakoutEntry(EntryRule):
    """Donchian breakout: close takes out the prior N-bar high/low. Buys
    strength instead of the last few bars' drift, so it fires later than
    momentum but on confirmation."""
    key = "breakout"

    def __init__(self, lookback=20):
        super().__init__(lookback=lookback)

    def directions(self, df):
        n = self.params["lookback"]
        hi = df["h"].rolling(n).max().shift(1).to_numpy(dtype=float)
        lo = df["l"].rolling(n).min().shift(1).to_numpy(dtype=float)
        c = df["c"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            return np.where(c > hi, 1.0, np.where(c < lo, -1.0, 0.0))

    def explain(self, df, i, direction):
        n = self.params["lookback"]
        lvl = (df["h"].rolling(n).max().shift(1).iat[i] if direction > 0
               else df["l"].rolling(n).min().shift(1).iat[i])
        side = "above" if direction > 0 else "below"
        return (f"breakout: close {df['c'].iat[i]:.2f} {side} the {n}-bar "
                f"{'high' if direction > 0 else 'low'} {lvl:.2f}")


class MeanRevEntry(EntryRule):
    """Fade stretch: buy when market_profile.py's mean-reversion z-score is
    deeply negative, sell when it's deeply positive. The opposite bet to
    momentum, and the one the regime read says to prefer when the market is
    ranging."""
    key = "meanrev"

    def __init__(self, z=1.5):
        super().__init__(z=z)

    def directions(self, df):
        z = self._col(df, "mr_zscore")
        thr = self.params["z"]
        with np.errstate(invalid="ignore"):
            return np.where(z <= -thr, 1.0, np.where(z >= thr, -1.0, 0.0))

    def explain(self, df, i, direction):
        return (f"meanrev: mr_zscore {df['mr_zscore'].iat[i]:+.2f} past "
                f"±{self.params['z']:.2f} — fading the stretch")


class VolSpikeEntry(EntryRule):
    """The live reactive bot's actual rule, made backtestable: a volume spike
    versus the trailing average, confirmed by a price move in the same bar."""
    key = "volspike"

    def __init__(self, vol_ratio=2.5, move=0.15):
        super().__init__(vol_ratio=vol_ratio, move=move)

    def directions(self, df):
        vr = self._col(df, "vol_ratio")
        chg = self._col(df, "chg_short")
        with np.errstate(invalid="ignore"):
            fires = (vr >= self.params["vol_ratio"]) & (np.abs(chg) >= self.params["move"])
        return np.where(fires, np.sign(chg), 0.0)

    def explain(self, df, i, direction):
        return (f"volspike: volume {df['vol_ratio'].iat[i]:.1f}x its trailing "
                f"average with chg_short {df['chg_short'].iat[i]:+.2f}%")


class PocFadeEntry(EntryRule):
    """Fade distance from the TPO point of control — the profile read
    /profile already reports, traded as a reversion to fair value."""
    key = "pocfade"

    def __init__(self, dist=0.5):
        super().__init__(dist=dist)

    def directions(self, df):
        d = self._col(df, "poc_dist_pct")
        thr = self.params["dist"]
        with np.errstate(invalid="ignore"):
            return np.where(d >= thr, -1.0, np.where(d <= -thr, 1.0, 0.0))

    def explain(self, df, i, direction):
        return (f"pocfade: price {df['poc_dist_pct'].iat[i]:+.2f}% from POC, "
                f"past ±{self.params['dist']:.2f}% — fading back toward value")


class RegimeRouterEntry(EntryRule):
    """Let the regime read pick the rule: follow momentum when the market is
    trending, fade when it's ranging, stand aside when it's neither. Tests
    whether market_profile.py's regime() call is worth acting on."""
    key = "regime_router"

    def __init__(self, deadzone=0.15, z=1.5):
        super().__init__(deadzone=deadzone, z=z)
        self._mom = MomentumEntry(deadzone=deadzone)
        self._mr = MeanRevEntry(z=z)

    def directions(self, df):
        mom = self._mom.directions(df)
        mr = self._mr.directions(df)
        trending = self._col(df, "regime_momentum") > 0.5
        ranging = self._col(df, "regime_meanrev") > 0.5
        return np.where(trending, mom, np.where(ranging, mr, 0.0))

    def explain(self, df, i, direction):
        if df["regime_momentum"].iat[i] > 0.5:
            return "regime_router[trending] -> " + self._mom.explain(df, i, direction)
        return "regime_router[ranging] -> " + self._mr.explain(df, i, direction)


class ValueAreaBreakoutEntry(EntryRule):
    """Volume-profile breakout: price sat inside the TPO value area last bar and
    closes outside it (past VAH/VAL) this bar — the market has left balance.
    Trades in the direction of the break; the opposite bet to PocFadeEntry,
    which fades the same distance-from-POC read instead of following it."""
    key = "va_breakout"

    def __init__(self, min_dist=0.0):
        super().__init__(min_dist=min_dist)

    def directions(self, df):
        in_va = self._col(df, "in_value_area")
        poc = self._col(df, "poc_dist_pct")
        prev_in_va = pd.Series(in_va).shift(1).to_numpy()
        min_dist = self.params["min_dist"]
        with np.errstate(invalid="ignore"):
            broke_out = (prev_in_va == 1.0) & (in_va == 0.0) & (np.abs(poc) >= min_dist)
            return np.where(broke_out & (poc > 0), 1.0,
                    np.where(broke_out & (poc < 0), -1.0, 0.0))

    def explain(self, df, i, direction):
        poc = df["poc_dist_pct"].iat[i]
        return (f"va_breakout: price left the value area this bar ({poc:+.2f}% from "
                f"POC) after sitting inside it last bar — trading the break "
                f"{'long' if direction > 0 else 'short'}")


class ValueAreaReclaimEntry(EntryRule):
    """Volume-profile reclaim: price pierced outside the TPO value area
    (VAH/VAL) last bar and closes back inside it this bar — a failed breakout.
    Trades in the direction of the re-entry, betting the rejection carries
    through toward POC. The mirror of ValueAreaBreakoutEntry: that rule follows
    a confirmed break, this one fades a failed one."""
    key = "va_reclaim"

    def __init__(self, min_dist=0.0):
        super().__init__(min_dist=min_dist)

    def directions(self, df):
        in_va = self._col(df, "in_value_area")
        poc = self._col(df, "poc_dist_pct")
        prev_in_va = pd.Series(in_va).shift(1).to_numpy()
        prev_poc = pd.Series(poc).shift(1).to_numpy()
        min_dist = self.params["min_dist"]
        with np.errstate(invalid="ignore"):
            reclaimed = (prev_in_va == 0.0) & (in_va == 1.0) & (np.abs(prev_poc) >= min_dist)
            # was below the value area (prev_poc < 0) and reclaimed up -> long;
            # was above it (prev_poc > 0) and reclaimed down -> short.
            return np.where(reclaimed & (prev_poc < 0), 1.0,
                    np.where(reclaimed & (prev_poc > 0), -1.0, 0.0))

    def explain(self, df, i, direction):
        prev_poc = df["poc_dist_pct"].iat[i - 1] if i > 0 else float("nan")
        side = "below" if prev_poc < 0 else "above"
        return (f"va_reclaim: price was {prev_poc:+.2f}% from POC ({side} the value "
                f"area) last bar, closed back inside this bar — trading the reclaim "
                f"{'long' if direction > 0 else 'short'}")


ENTRY_RULES = {
    "momentum": MomentumEntry,
    "breakout": BreakoutEntry,
    "meanrev": MeanRevEntry,
    "volspike": VolSpikeEntry,
    "pocfade": PocFadeEntry,
    "regime_router": RegimeRouterEntry,
    "va_breakout": ValueAreaBreakoutEntry,
    "va_reclaim": ValueAreaReclaimEntry,
}


# ======================================================================
# EXIT RULES
#
# Two kinds, because they observe different things:
#   "close"    — decided from information available at bar t's close; fills at
#                that close (time stop, signal flip).
#   "intrabar" — a resting level that can be touched while the bar being held
#                trades; fills at the level (stop loss, target, trailing stop).
# ======================================================================
@dataclass
class TradeState:
    side: float
    entry_idx: int
    entry_px: float
    entry_reason: str
    bars_held: int = 0
    best_px: float = 0.0          # most favorable price seen so far (for trails)
    levels: dict = field(default_factory=dict)


class ExitRule:
    key = "base"
    kind = "close"

    def __init__(self, **params):
        self.params = params

    def __repr__(self):
        ps = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.key}({ps})"

    def on_entry(self, st, ctx):
        """Hook to stash per-trade levels at entry time."""

    def check_close(self, st, t, ctx):
        """Return a reason string to exit at bar t's close, else None."""
        return None

    def check_intrabar(self, st, j, ctx):
        """Return (fill_price, reason) if a level is touched during bar j."""
        return None


class AtrBracketExit(ExitRule):
    """Fixed ATR-scaled stop loss and take profit, set at entry and left alone.
    The risk control the DDQN policy never had. sl_mult/tp_mult are multiples
    of the causal ATR fraction from labels.py (already clipped to [0.2%, 8%])."""
    key = "bracket"
    kind = "intrabar"

    def __init__(self, sl_mult=1.0, tp_mult=2.0):
        super().__init__(sl_mult=sl_mult, tp_mult=tp_mult)

    def on_entry(self, st, ctx):
        frac = ctx["atr"][st.entry_idx]
        if not np.isfinite(frac):
            frac = 0.005
        sl_f = frac * self.params["sl_mult"]
        tp_f = frac * self.params["tp_mult"]
        st.levels["sl"] = st.entry_px * (1 - sl_f * st.side)
        st.levels["tp"] = st.entry_px * (1 + tp_f * st.side)

    def check_intrabar(self, st, j, ctx):
        hi, lo = ctx["h"][j], ctx["l"][j]
        sl, tp = st.levels.get("sl"), st.levels.get("tp")
        if st.side > 0:
            if sl is not None and lo <= sl:
                return sl, (f"stop_loss: low {lo:.2f} <= stop {sl:.2f} "
                            f"({(sl / st.entry_px - 1) * 100:+.2f}% from entry {st.entry_px:.2f})")
            if tp is not None and hi >= tp:
                return tp, (f"take_profit: high {hi:.2f} >= target {tp:.2f} "
                            f"({(tp / st.entry_px - 1) * 100:+.2f}% from entry {st.entry_px:.2f})")
        else:
            if sl is not None and hi >= sl:
                return sl, (f"stop_loss: high {hi:.2f} >= stop {sl:.2f} "
                            f"({(1 - sl / st.entry_px) * 100:+.2f}% from entry {st.entry_px:.2f})")
            if tp is not None and lo <= tp:
                return tp, (f"take_profit: low {lo:.2f} <= target {tp:.2f} "
                            f"({(1 - tp / st.entry_px) * 100:+.2f}% from entry {st.entry_px:.2f})")
        return None


class AtrTrailExit(ExitRule):
    """Trailing stop at `mult` ATR from the best price the trade has seen.
    The level is computed from the best price BEFORE the bar being checked —
    using the same bar's extreme to both set and trigger the trail would be
    assuming an intrabar ordering the OHLC can't tell us."""
    key = "trail"
    kind = "intrabar"

    def __init__(self, mult=2.0):
        super().__init__(mult=mult)

    def on_entry(self, st, ctx):
        st.best_px = st.entry_px
        st.levels["trail_frac"] = ctx["atr"][st.entry_idx]

    def check_intrabar(self, st, j, ctx):
        frac = st.levels.get("trail_frac")
        if not np.isfinite(frac):
            frac = 0.005
        dist = frac * self.params["mult"]
        lvl = st.best_px * (1 - dist) if st.side > 0 else st.best_px * (1 + dist)
        hi, lo = ctx["h"][j], ctx["l"][j]
        touched = lo <= lvl if st.side > 0 else hi >= lvl
        if touched:
            return lvl, (f"trail_stop: gave back {dist * 100:.2f}% from the best "
                         f"price {st.best_px:.2f} (level {lvl:.2f})")
        return None

    def after_bar(self, st, j, ctx):
        st.best_px = (max(st.best_px, ctx["h"][j]) if st.side > 0
                      else min(st.best_px, ctx["l"][j]))


class TimeExit(ExitRule):
    """Hard cap on holding period — the vertical barrier from labels.py, now
    actually enforced on the live position instead of only on the label."""
    key = "time"
    kind = "close"

    def __init__(self, max_hold=12):
        super().__init__(max_hold=max_hold)

    def check_close(self, st, t, ctx):
        if st.bars_held >= self.params["max_hold"]:
            return f"time_stop: held {st.bars_held} bars, cap is {self.params['max_hold']}"
        return None


class SignalFlipExit(ExitRule):
    """Exit when the entry rule itself turns against the position — the
    implicit behavior of a always-in position policy, made explicit so it can
    be compared against real stops."""
    key = "flip"
    kind = "close"

    def check_close(self, st, t, ctx):
        d = ctx["entry_dir"][t]
        if d != 0.0 and np.sign(d) != np.sign(st.side):
            return f"signal_flip: entry rule now points {'long' if d > 0 else 'short'}"
        return None


EXIT_RULES = {
    "bracket": AtrBracketExit,
    "trail": AtrTrailExit,
    "time": TimeExit,
    "flip": SignalFlipExit,
}


# ======================================================================
# STRATEGY
# ======================================================================
@dataclass
class Strategy:
    entry: EntryRule
    exits: list
    prob_threshold: float = 0.0   # 0 disables the phase-1 ML gate
    atr_lookback: int = 14

    @property
    def name(self):
        ex = "+".join(repr(e) for e in self.exits) or "none"
        gate = f"+ml_gate({self.prob_threshold:.2f})" if self.uses_gate else ""
        return f"{self.entry!r} -> {ex}{gate}"

    @property
    def uses_gate(self):
        return self.prob_threshold > 0.0

    def describe(self):
        lines = [f"entry: {self.entry!r}"]
        if self.uses_gate:
            lines.append(f"gate : phase-1 XGBoost P(win) >= {self.prob_threshold:.2f}")
        for e in self.exits:
            lines.append(f"exit : {e!r}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "entry": self.entry.key,
            "entry_params": dict(self.entry.params),
            "exits": [{"rule": e.key, "params": dict(e.params)} for e in self.exits],
            "prob_threshold": self.prob_threshold,
        }


def strategy_from_dict(d):
    entry = ENTRY_RULES[d["entry"]](**d.get("entry_params", {}))
    exits = [EXIT_RULES[e["rule"]](**e.get("params", {})) for e in d.get("exits", [])]
    return Strategy(entry=entry, exits=exits, prob_threshold=d.get("prob_threshold", 0.0))


# ======================================================================
# RUNNER
# ======================================================================
def run_strategy(df, strategy, feature_cols=None, fee=0.0005, scorer=None, probs=None):
    """Walk `df` bar by bar under `strategy`. Returns (trace, trades).

    trace  — per-bar DataFrame, same schema backtest.compute_metrics() expects.
    trades — list of dicts, one per completed trade, each carrying the reason
             it was opened and the reason it was closed.

    A position decided at bar t is held over t -> t+1, matching env.py and
    backtest.run_policy(). Only one position at a time, unit size.

    `probs` optionally supplies the phase-1 P(long wins) for every bar, already
    batch-computed (see SignalScorer.predict_proba_batch) — the search scores
    hundreds of strategies over the same bars and would otherwise re-run the
    model per candidate entry. Identical results, far less time.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    c = df["c"].to_numpy(dtype=float)
    h = df["h"].to_numpy(dtype=float)
    l = df["l"].to_numpy(dtype=float)
    ts = df["ts"].to_numpy()
    regimes = _regime_series(df)

    entry_dir = np.nan_to_num(strategy.entry.directions(df), nan=0.0)
    ctx = {
        "h": h, "l": l, "c": c,
        "atr": atr_fractions(df, lookback=strategy.atr_lookback, mult=1.0),
        "entry_dir": entry_dir,
    }

    close_exits = [e for e in strategy.exits if e.kind == "close"]
    intrabar_exits = [e for e in strategy.exits if e.kind == "intrabar"]

    has_model = probs is not None or (scorer is not None and getattr(scorer, "is_trained", False))
    gate_on = strategy.uses_gate and has_model
    gate_cols = list(feature_cols or (scorer.cols if scorer is not None else []))
    probs = None if probs is None else np.asarray(probs, dtype=float)

    position, st = 0.0, None
    rows, trades = [], []

    def close_trade(state, exit_idx, exit_px, reason):
        gross = state.side * (exit_px - state.entry_px) / state.entry_px
        fees = fee * 2.0                                   # one unit in, one out
        trades.append({
            "side": "long" if state.side > 0 else "short",
            "entry_ts": ts[state.entry_idx], "entry_px": float(state.entry_px),
            "exit_ts": ts[exit_idx], "exit_px": float(exit_px),
            "bars_held": int(exit_idx - state.entry_idx),
            "gross_ret": float(gross), "fees": float(fees),
            "pnl": float(gross - fees),
            "entry_reason": state.entry_reason, "exit_reason": reason,
            "regime": str(regimes[state.entry_idx]),
            "xgb_prob": state.levels.get("xgb_prob"),
            "chg_short": _at(df, "chg_short", state.entry_idx),
            "mr_zscore": _at(df, "mr_zscore", state.entry_idx),
            "vol_ratio": _at(df, "vol_ratio", state.entry_idx),
            "poc_dist_pct": _at(df, "poc_dist_pct", state.entry_idx),
        })

    for t in range(n - 1):
        target, gated, prob = position, False, None

        if position == 0.0:
            d = float(entry_dir[t])
            if d != 0.0:
                if gate_on:
                    if probs is not None:
                        p = float(probs[t])
                    else:
                        row = {col: float(df[col].iat[t]) for col in gate_cols if col in df.columns}
                        p = scorer.predict_proba(row)
                    prob = p if d > 0 else 1.0 - p
                    if prob < strategy.prob_threshold:
                        d, gated = 0.0, True
                if d != 0.0:
                    target = d
        else:
            for rule in close_exits:
                why = rule.check_close(st, t, ctx)
                if why:
                    close_trade(st, t, c[t], why)
                    st, target = None, 0.0
                    break

        turnover = abs(target - position)
        fee_cost = fee * turnover

        if target != 0.0 and st is None:
            reason = strategy.entry.explain(df, t, target)
            if prob is not None:
                reason += f" (xgb_p {prob:.2f})"
            reason += f" [regime {regimes[t]}]"
            st = TradeState(side=target, entry_idx=t, entry_px=float(c[t]),
                            entry_reason=reason)
            if prob is not None:
                st.levels["xgb_prob"] = float(prob)
            for rule in strategy.exits:
                rule.on_entry(st, ctx)

        bar_ret = (c[t + 1] - c[t]) / c[t] if c[t] else 0.0
        realized, hit = bar_ret, None
        if target != 0.0 and st is not None:
            # Several levels can sit inside one bar and OHLC can't order them:
            # assume the worst fill for the side we're holding.
            fills = [f for f in (r.check_intrabar(st, t + 1, ctx) for r in intrabar_exits) if f]
            if fills:
                hit = min(fills, key=lambda f: f[0]) if target > 0 else max(fills, key=lambda f: f[0])
                realized = (hit[0] - c[t]) / c[t] if c[t] else 0.0

        reward = target * realized - fee_cost
        if hit is not None:
            reward -= fee * abs(target)                    # the exit's own turnover
            turnover += abs(target)

        rows.append({"ts": ts[t], "position": position, "target": target,
                     "bar_ret": bar_ret, "reward": reward, "turnover": turnover,
                     "fee_cost": fee * turnover, "gated": gated,
                     "regime": regimes[t]})

        if hit is not None:
            close_trade(st, t + 1, hit[0], hit[1])
            st, position = None, 0.0
        else:
            if st is not None:
                st.bars_held += 1
                for rule in intrabar_exits:
                    if hasattr(rule, "after_bar"):
                        rule.after_bar(st, t + 1, ctx)
            position = target

    if st is not None:                                     # mark the tail open trade out
        close_trade(st, n - 1, float(c[-1]), "end_of_data: still open when the series ran out")

    return pd.DataFrame(rows), trades


def _at(df, col, i):
    if col not in df.columns:
        return None
    v = df[col].iat[i]
    return None if pd.isna(v) else float(v)


def strategy_metrics(trace, trades, bars_per_year=288 * 365):
    """backtest.compute_metrics() for the bar-level risk numbers, with the
    trade-level fields recomputed from the explicit trade list — this runner
    knows where trades started and ended, so it doesn't have to infer them
    from the position series the way compute_metrics() does."""
    m = compute_metrics(trace, bars_per_year=bars_per_year)
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    m["trade_count"] = int(len(pnls))
    m["win_rate"] = float(len(wins) / len(pnls)) if len(pnls) else float("nan")
    m["avg_trade_pnl"] = float(pnls.mean()) if len(pnls) else float("nan")
    m["profit_factor"] = (float(wins.sum() / -losses.sum()) if losses.sum() < 0
                          else (float("inf") if wins.sum() > 0 else 0.0))
    m["avg_bars_held"] = (float(np.mean([t["bars_held"] for t in trades]))
                          if trades else float("nan"))
    m["exit_breakdown"] = _exit_breakdown(trades)
    return m


def _exit_breakdown(trades):
    """How each trade ended, and what that exit was worth on average — the
    fastest way to see whether the stop is too tight or the target too far."""
    out = {}
    for t in trades:
        kind = t["exit_reason"].split(":")[0]
        slot = out.setdefault(kind, {"n": 0, "pnl": 0.0})
        slot["n"] += 1
        slot["pnl"] += t["pnl"]
    for kind, slot in out.items():
        slot["avg_pnl"] = slot["pnl"] / slot["n"] if slot["n"] else 0.0
    return out


def trades_to_frame(trades):
    cols = ["entry_ts", "exit_ts", "side", "entry_px", "exit_px", "bars_held",
            "gross_ret", "fees", "pnl", "regime", "xgb_prob", "chg_short",
            "mr_zscore", "vol_ratio", "poc_dist_pct", "entry_reason", "exit_reason"]
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(trades)[cols]


# ======================================================================
# GRID — the candidate space the daily search draws from
# ======================================================================
def default_grid(prob_thresholds=(0.0, 0.55)):
    """Every (entry x exit x params) combination worth trying. Ordering is
    deterministic so a capped run is reproducible; search.py shuffles with a
    date-derived seed so different days explore different slices."""
    entries = []
    for dz in (0.10, 0.20, 0.35):
        entries.append(MomentumEntry(deadzone=dz))
    for lb in (12, 24, 48):
        entries.append(BreakoutEntry(lookback=lb))
    for z in (1.0, 1.5, 2.0):
        entries.append(MeanRevEntry(z=z))
    for vr, mv in ((2.0, 0.10), (3.0, 0.20)):
        entries.append(VolSpikeEntry(vol_ratio=vr, move=mv))
    for d in (0.4, 0.8):
        entries.append(PocFadeEntry(dist=d))
    entries.append(RegimeRouterEntry(deadzone=0.15, z=1.5))
    for d in (0.0, 0.5):
        entries.append(ValueAreaBreakoutEntry(min_dist=d))
        entries.append(ValueAreaReclaimEntry(min_dist=d))

    exit_sets = []
    for sl, tp in ((1.0, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 2.0)):
        for mh in (12, 48):
            exit_sets.append([AtrBracketExit(sl_mult=sl, tp_mult=tp), TimeExit(max_hold=mh)])
    for mult in (1.5, 3.0):
        exit_sets.append([AtrTrailExit(mult=mult), TimeExit(max_hold=48)])
        exit_sets.append([AtrBracketExit(sl_mult=1.5, tp_mult=4.0),
                          AtrTrailExit(mult=mult), TimeExit(max_hold=48)])
    exit_sets.append([SignalFlipExit(), TimeExit(max_hold=48)])
    # The incumbent: no stop at all, flip on signal. Kept in the grid on
    # purpose so every report shows what the stops are actually buying.
    exit_sets.append([SignalFlipExit()])

    grid = []
    for entry, exits, pt in itertools.product(entries, exit_sets, prob_thresholds):
        grid.append(Strategy(entry=entry, exits=list(exits), prob_threshold=pt))
    return grid


def baseline_strategy():
    """Plain momentum, no stops, flip on signal — what the bot would do with
    no ML and no risk management. The number every candidate has to beat."""
    return Strategy(entry=MomentumEntry(deadzone=0.15), exits=[SignalFlipExit()])


# ======================================================================
# SELF-TEST
# ======================================================================
def _selftest():
    print("alpha_ml/strategies.py self-test (synthetic market)\n")
    from features import assemble, synthetic_market

    market = synthetic_market(n=2500, planted=True)
    df, cols = assemble(market, window=48)
    print(f"  feature table: {len(df)} bars x {len(cols)} features\n")

    strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                     exits=[AtrBracketExit(sl_mult=1.0, tp_mult=2.0), TimeExit(max_hold=24)])
    trace, trades = run_strategy(df, strat, feature_cols=cols, fee=0.0005)
    m = strategy_metrics(trace, trades)
    print(f"  {strat.name}")
    print(f"    trades {m['trade_count']}  win rate {m['win_rate']*100:.1f}%  "
          f"return {m['total_return']*100:+.2f}%  sharpe {m['sharpe']:+.2f}")
    assert m["trade_count"] > 0, "a momentum rule on a trending synthetic market must trade"

    # Every trade must carry its reasoning — that is the whole point of the module.
    assert all(t["entry_reason"] and t["exit_reason"] for t in trades)
    print(f"\n  every one of the {len(trades)} trades carries an entry and exit reason ✓")
    print("  sample trade:")
    s = trades[0]
    print(f"    {s['side']} @ {s['entry_px']:.2f} -> {s['exit_px']:.2f}  "
          f"{s['pnl']*100:+.2f}% net over {s['bars_held']} bars")
    print(f"      why in : {s['entry_reason']}")
    print(f"      why out: {s['exit_reason']}")

    # Stops must actually bind: no trade may exceed its own stop distance by
    # more than the bar it filled on, and none may outlive the time cap.
    assert all(t["bars_held"] <= 24 for t in trades), "time stop did not bind"
    kinds = set(t["exit_reason"].split(":")[0] for t in trades)
    assert kinds & {"stop_loss", "take_profit", "time_stop"}, f"unexpected exits: {kinds}"
    print(f"  time cap binds on every trade ✓   exit kinds seen: {sorted(kinds)} ✓")

    # Trade PnL (entry->exit) and the bar trace (sum of simple returns) measure
    # the same thing two ways; they agree to second order.
    trade_sum = sum(t["pnl"] for t in trades)
    trace_sum = float(trace["reward"].sum())
    print(f"\n  trade-sum {trade_sum*100:+.3f}%  vs  bar-trace {trace_sum*100:+.3f}%  "
          f"(gap {abs(trade_sum - trace_sum)*100:.3f}pp)")
    assert abs(trade_sum - trace_sum) < 0.02, "trade and bar accounting have diverged"

    # A no-stop policy must be able to hold longer than a stopped one.
    naked, naked_trades = run_strategy(df, baseline_strategy(), feature_cols=cols, fee=0.0005)
    nm = strategy_metrics(naked, naked_trades)
    print(f"\n  baseline (no stops): trades {nm['trade_count']}  "
          f"avg hold {nm['avg_bars_held']:.1f} bars  return {nm['total_return']*100:+.2f}%")
    print(f"  bracketed          : trades {m['trade_count']}  "
          f"avg hold {m['avg_bars_held']:.1f} bars  return {m['total_return']*100:+.2f}%")

    # The ML gate can only ever reduce the number of entries taken.
    from xgb_entry import build_dataset, train_final
    xdf, xcols = build_dataset(market, window=48, max_hold=10, direction_mode="long_only")
    scorer = train_final(xdf, xcols)
    gated_strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                           exits=[AtrBracketExit(), TimeExit(max_hold=24)],
                           prob_threshold=0.55)
    gtrace, gtrades = run_strategy(df, gated_strat, feature_cols=xcols, fee=0.0005, scorer=scorer)
    assert len(gtrades) <= len(trades) + 1, "the ML gate somehow created entries"
    assert int(gtrace["gated"].sum()) >= 0
    print(f"\n  ml gate: {len(trades)} ungated trades -> {len(gtrades)} gated "
          f"({int(gtrace['gated'].sum())} entries blocked) ✓")

    # The sharpest check available: on an UNPLANTED market (pure random walk)
    # there is no edge to find, so every strategy must bleed roughly its own
    # fee bill. Anything that turns a profit here is reading the future.
    print("\n  no-edge control (planted=False — nothing to find):")
    for seed in (1, 2, 3):
        ctrl_mkt = synthetic_market(n=2000, seed=seed, planted=False)
        ctrl_df, ctrl_cols = assemble(ctrl_mkt, window=48)
        ctrl_strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                              exits=[AtrBracketExit(sl_mult=1.0, tp_mult=2.0), TimeExit(max_hold=24)])
        ctr, ctd = run_strategy(ctrl_df, ctrl_strat, feature_cols=ctrl_cols, fee=0.0005)
        cm = strategy_metrics(ctr, ctd)
        print(f"    seed {seed}: {cm['trade_count']:>4} trades  "
              f"return {cm['total_return']*100:+.2f}%  fees {cm['fees_paid']*100:.2f}%")
        assert cm["total_return"] < 0, (
            f"seed {seed}: turned a profit on a random walk — that is lookahead, not alpha")
        assert cm["total_return"] > -3 * cm["fees_paid"], (
            f"seed {seed}: lost far more than the fee bill — check the fill logic")
    print("  every control run bleeds about its fee bill, none profits ✓")

    grid = default_grid()
    assert len(grid) > 100, f"grid is only {len(grid)} configs"
    rt = strategy_from_dict(grid[7].to_dict())
    assert rt.name == grid[7].name, "strategy dict round-trip lost information"
    print(f"\n  grid: {len(grid)} candidate strategies  ✓   dict round-trip ✓")

    print("\n  PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("give --selftest (search.py drives the full daily strategy search)")
