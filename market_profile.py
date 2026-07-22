"""
market_profile.py — TPO / Market-Profile analytics for the OKX Telegram bot.

Pure-Python (stdlib only, consistent with the rest of the repo). Given OKX
candles (newest-first), it computes:

  - TPO profile      → POC, Value Area High/Low (default 70% value area),
                       profile range, and whether price sits inside value.
  - Mean-reversion   → z-score of price vs its rolling mean + Bollinger-style
                       bands, and a fade signal when price is stretched.
  - Regime status    → trend vs range using Kaufman's efficiency ratio, and
                       which playbook that favours (momentum vs mean-reversion).

Candle row format (OKX): [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
Candles come newest-first from the OKX API; helpers reverse to oldest-first.
"""

from dataclasses import dataclass


def _closes(candles):
    """Closes, oldest-first, as floats."""
    return [float(c[4]) for c in reversed(candles)]


# --------------------------------------------------------------------------- #
# TPO profile: POC / VAH / VAL
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    poc: float               # point of control (most-traded price)
    vah: float               # value area high
    val: float               # value area low
    value_area_pct: float    # fraction of TPOs captured (~ value_area_frac)
    price_high: float        # profile range top
    price_low: float         # profile range bottom
    last: float              # current price
    in_value_area: bool      # is price inside [VAL, VAH]?
    poc_dist_pct: float      # signed % distance of price above/below POC


def build_tpo_profile(candles, num_buckets=50, value_area_frac=0.70):
    """
    Build a TPO (time-price-opportunity) profile. Each candle contributes one
    TPO to every price bucket its [low, high] range overlaps. POC is the busiest
    bucket; the value area expands out from POC (larger neighbour first) until it
    covers value_area_frac of all TPOs.

    Returns a Profile, or None if there isn't enough range/data to bin.
    """
    if len(candles) < 5:
        return None
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    last = float(candles[0][4])
    price_high = max(highs)
    price_low = min(lows)
    span = price_high - price_low
    if span <= 0:
        return None

    tick = span / num_buckets
    counts = [0] * num_buckets
    for hi, lo in zip(highs, lows):
        lo_b = max(0, min(num_buckets - 1, int((lo - price_low) / tick)))
        hi_b = max(0, min(num_buckets - 1, int((hi - price_low) / tick)))
        for b in range(lo_b, hi_b + 1):
            counts[b] += 1

    total = sum(counts)
    if total <= 0:
        return None

    poc_idx = max(range(num_buckets), key=lambda i: counts[i])
    target = total * value_area_frac
    running = counts[poc_idx]
    low_i = high_i = poc_idx
    while running < target and (low_i > 0 or high_i < num_buckets - 1):
        up = counts[high_i + 1] if high_i < num_buckets - 1 else -1
        dn = counts[low_i - 1] if low_i > 0 else -1
        if up == -1 and dn == -1:
            break
        if up >= dn:
            high_i += 1
            running += counts[high_i]
        else:
            low_i -= 1
            running += counts[low_i]

    poc = price_low + (poc_idx + 0.5) * tick
    vah = price_low + (high_i + 1) * tick
    val = price_low + low_i * tick
    return Profile(
        poc=poc,
        vah=vah,
        val=val,
        value_area_pct=running / total * 100.0,
        price_high=price_high,
        price_low=price_low,
        last=last,
        in_value_area=val <= last <= vah,
        poc_dist_pct=(last - poc) / poc * 100.0 if poc else 0.0,
    )


# --------------------------------------------------------------------------- #
# Mean reversion
# --------------------------------------------------------------------------- #
@dataclass
class MeanReversion:
    mean: float
    std: float
    zscore: float
    upper: float             # mean + 2*std
    lower: float             # mean - 2*std
    last: float
    signal: str              # "fade-short" | "fade-long" | "neutral"
    note: str


def mean_reversion(candles, lookback=20, z_trigger=2.0):
    """Rolling-mean deviation read. Stretched price is a fade signal."""
    closes = _closes(candles)
    if len(closes) < 5:
        return None
    window = closes[-lookback:] if len(closes) >= lookback else closes
    n = len(window)
    mean = sum(window) / n
    std = (sum((x - mean) ** 2 for x in window) / n) ** 0.5
    last = closes[-1]
    z = (last - mean) / std if std else 0.0

    if z >= z_trigger:
        signal, note = "fade-short", f"stretched {z:+.2f}σ above mean → expect reversion down"
    elif z <= -z_trigger:
        signal, note = "fade-long", f"stretched {z:+.2f}σ below mean → expect reversion up"
    elif abs(z) < 0.5:
        signal, note = "neutral", f"near fair value ({z:+.2f}σ)"
    else:
        signal, note = "neutral", f"mild deviation ({z:+.2f}σ)"

    return MeanReversion(
        mean=mean,
        std=std,
        zscore=z,
        upper=mean + 2 * std,
        lower=mean - 2 * std,
        last=last,
        signal=signal,
        note=note,
    )


# --------------------------------------------------------------------------- #
# Regime status
# --------------------------------------------------------------------------- #
@dataclass
class Regime:
    label: str               # "Trending up/down" | "Ranging" | "Transitional"
    efficiency_ratio: float  # 0..1 (1 = pure trend, 0 = pure chop)
    direction: str           # "up" | "down" | "flat"
    favored: str             # "momentum" | "mean-reversion" | "none"


def efficiency_ratio(closes, n):
    """Kaufman efficiency ratio over the last n moves: |net move| / path length."""
    n = min(n, len(closes) - 1)
    if n < 1:
        return 0.0
    seg = closes[-(n + 1):]
    net = abs(seg[-1] - seg[0])
    path = sum(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg)))
    return net / path if path else 0.0


def regime(candles, lookback=20, trend_er=0.5, range_er=0.3):
    """Classify trend vs range and name the playbook it favours."""
    closes = _closes(candles)
    if len(closes) < 5:
        return None
    er = efficiency_ratio(closes, lookback)
    ref = closes[-min(lookback + 1, len(closes))]
    net_pct = (closes[-1] - ref) / ref * 100.0 if ref else 0.0
    direction = "up" if net_pct > 0.2 else "down" if net_pct < -0.2 else "flat"

    if er >= trend_er and direction != "flat":
        label = f"\U0001F3C3 Trending {direction} (ER {er:.2f})"
        favored = "momentum"
    elif er <= range_er:
        label = f"\U0001F501 Ranging / balanced (ER {er:.2f})"
        favored = "mean-reversion"
    else:
        label = f"⚖️ Transitional / choppy (ER {er:.2f})"
        favored = "none"
    return Regime(label=label, efficiency_ratio=er, direction=direction, favored=favored)
