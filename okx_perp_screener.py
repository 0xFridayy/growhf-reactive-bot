"""
OKX perpetual-futures screener: alerts to Telegram on unusual price spikes
confirmed by a volume surge. Uses only OKX public market-data endpoints
(no API key/secret needed) and Python's standard library (no pip installs).

Run:
    python okx_perp_screener.py
    python okx_perp_screener.py --list-suspects   # regenerate the equity denylist

Config lives in config.json next to this file.

WHAT CHANGED (2026-08-04) AND WHY
---------------------------------
1. IT WAS LONG-ONLY BY CONSTRUCTION. The old sizing block ran only when
   `pct_change < 0`, and build_dip_trade_plan always put the stop BELOW entry.
   So every plan it had ever emitted was a dip-buy, and UP spikes got no plan at
   all — not a bug in the sizing, a hard-coded direction. It now takes both
   sides: buy dips in a bullish regime, sell rips in a bearish one.

2. IT HAD NO IDEA WHAT THE MARKET WAS DOING. The screener never consulted
   /bias, so it dip-bought into any tape, including a falling one. Direction is
   now chosen BY the bias score (bias_telegram.current_bias): a mean-reversion
   entry is only taken WITH the regime, and in chop no plan is emitted at all.
   The bias is cached — it is an hourly number, not a per-tick one.

3. EVERY SPIKE GOT POSTED. There was one gate (move % + volume multiple) and it
   fired on anything that cleared it. Alerts now carry a 0-100 CONFIDENCE built
   from move size, volume expansion, regime agreement and liquidity, and only
   those at or above `min_confidence` are sent.

4. SIZING IS GONE FROM THE MESSAGE. Leverage / notional / margin / contracts /
   risk-$ were a $280-account's arithmetic restated on every alert, and they
   made the signal harder to read. Risk is now shown as a PERCENTAGE (stop
   distance), which is account-size independent. kelly_sizing.py is untouched
   and still used by okx_tele_bot.py.

5. TOKENIZED EQUITIES. Half the alerts were US stocks (TSEM, IREN, CRWV, APLD,
   FLNC, TTMI, SNXX...), which spike on the US cash open and have nothing to do
   with a crypto regime. OKX's instrument metadata cannot distinguish them —
   every USDT swap reports category "1", ctType "linear", ruleType "normal",
   and max leverage does not separate them either (ADBE and 1INCH are both 20x).
   Nor are they closed at weekends — measured 2026-08-04, IREN and CRWV trade
   through Saturday and Sunday at 45-62% of weekday turnover, so a "no weekend
   bars" test finds nothing. They are therefore excluded by a hand-maintained
   config denylist, `exclude_bases`. `--list-suspects` RANKS candidates by that
   weekend/weekday turnover ratio to make the list easier to maintain, but it is
   a hint for a human to review, not a classifier — crypto sits near 0.88-1.00
   and these equities near 0.45-0.62, which overlap enough to misfile things.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the emoji used in
# alert text — without this a print() of an alert raises UnicodeEncodeError and
# kills the poll loop.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CONFIG_PATH = Path(__file__).with_name("config.json")
LOCK_PATH = Path(__file__).with_name(".okx_perp_screener.lock")
OKX_BASE = "https://www.okx.com"
USER_AGENT = "okx-perp-screener/1.0"

# Preferred source: the cache written by asset_classifier.py in the Perps repo,
# which DERIVES the classification from CoinGecko's tokenized-asset categories
# plus OKX spot listings instead of anyone maintaining a list by hand. Falls
# back to config.json's exclude_bases, then to the literal below. Set
# ASSET_CLASSES to relocate the cache.
ASSET_CLASSES_PATH = os.environ.get(
    "ASSET_CLASSES",
    r"C:\Users\jason\OneDrive\Desktop\VsCode\Perps\asset_classes.json")


def classified_equities():
    """Bases the classifier calls equity/unknown, or None if unavailable."""
    try:
        with open(ASSET_CLASSES_PATH, encoding="utf-8") as f:
            d = json.load(f)
        assets = d.get("assets") or {}
        bad = {b for b, v in assets.items() if v.get("cls") in ("equity", "unknown")}
        age_h = (time.time() - float(d.get("ts", 0))) / 3600.0
        if bad:
            print(f"[classifier] {len(bad)} non-crypto bases from cache "
                  f"({age_h:.0f}h old)")
            return bad
    except (OSError, ValueError, KeyError) as e:
        print(f"[classifier] cache unavailable ({e}) — using config exclude_bases")
    return None


# Seeded from the symbols observed alerting plus the obvious US large caps. Not
# exhaustive and cannot be — see note 5 above. Override/extend in config.json.
DEFAULT_EXCLUDE_BASES = [
    "TSEM", "SNXX", "FLNC", "IREN", "CRWV", "TTMI", "APLD", "AAOI", "ADBE",
    "GOOGL", "AAPL", "MSFT", "META", "AMZN", "NVDA", "TSLA", "INTC", "AMD",
    "MU", "SOXL", "SPCX", "COIN", "MSTR", "HOOD", "CRCL", "AVGO", "NFLX",
    "BABA", "SMCI", "ARM", "PLTR", "TSM", "SKHYNIX", "SAMSUNG", "HYUNDAI",
    "QQQ", "SPY", "SPX", "NDX", "GLD", "XAU", "XAG", "CL", "A",
]


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("telegram_bot_token", "").startswith("PUT_YOUR"):
        raise SystemExit(
            f"Edit {CONFIG_PATH} first: fill in telegram_bot_token and telegram_chat_id."
        )
    return cfg


def http_get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_tickers(quote_filter):
    data = http_get_json(f"{OKX_BASE}/api/v5/market/tickers?instType=SWAP")
    rows = data.get("data", [])
    if quote_filter:
        rows = [r for r in rows if r["instId"].split("-")[1:2] == [quote_filter]]
    return rows


def fetch_recent_candles(inst_id, bar="1m", limit=20):
    url = f"{OKX_BASE}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    data = http_get_json(url)
    return data.get("data", [])  # newest first: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.URLError as e:
        print(f"[telegram] send failed: {e}")


def regime_side(bias_score, cfg):
    """Which way this regime lets us fade a spike.

    A spike is an overreaction; fading it is only worth doing WITH the tape.
    Buying a dip in a bearish regime is catching a knife, and shorting a rip in
    a bull market is standing in front of one. In chop neither edge exists, so
    nothing is emitted rather than guessing.

    Returns "long" (fade DOWN spikes), "short" (fade UP spikes), or None.
    """
    band = float(cfg.get("bias_neutral_band", 12.0))
    if bias_score >= band:
        return "long"
    if bias_score <= -band:
        return "short"
    return None


def confidence(pct_change, threshold, vol_ratio, vol_multiplier,
               bias_score, turnover_usd, cfg):
    """0-100 conviction. Four independent inputs, none of which can carry the
    score alone — a huge move on thin volume in a flat regime should not read as
    high conviction just because the move was huge.

      move     how far past its own threshold the spike went
      volume   how far past the volume multiple it was confirmed
      regime   how strongly /bias agrees with the direction we would take
      depth    24h turnover, as a liquidity sanity check
    """
    move = min(abs(pct_change) / (threshold * 2.0), 1.0)
    vol = min(vol_ratio / (vol_multiplier * 3.0), 1.0)
    regime = min(abs(bias_score) / 50.0, 1.0)
    depth = min(turnover_usd / float(cfg.get("full_confidence_turnover_usd", 5e7)), 1.0)
    w = cfg.get("confidence_weights", {})
    return 100.0 * (
        move * float(w.get("move", 0.30))
        + vol * float(w.get("volume", 0.30))
        + regime * float(w.get("regime", 0.25))
        + depth * float(w.get("depth", 0.15))
    )


def build_reversion_plan(candles, entry_price, side, scfg):
    """Stop from the spike's own structure: beyond the recent swing extreme,
    clamped to [min_stop_frac, max_stop_frac]. Target is 1R back toward the mean.
    Returns (stop_price, take_profit, stop_pct).

    No position sizing — that is the trader's call, and it depends on an account
    balance this screener has no business restating on every alert.
    """
    lookback = int(scfg.get("swing_lookback_candles", 10))
    window = candles[:lookback]
    if not window:
        return None
    buffer = float(scfg.get("stop_buffer_frac", 0.001))
    min_sf = float(scfg.get("min_stop_frac", 0.004))
    max_sf = float(scfg.get("max_stop_frac", 0.05))

    if side == "long":
        extreme = min(float(c[3]) for c in window)          # swing low
        stop_dist = (entry_price - extreme * (1.0 - buffer)) / entry_price
    else:
        extreme = max(float(c[2]) for c in window)          # swing high
        stop_dist = (extreme * (1.0 + buffer) - entry_price) / entry_price
    if stop_dist <= 0:                                       # extreme on the wrong
        stop_dist = min_sf                                   # side -> use the floor
    stop_dist = max(min_sf, min(max_sf, stop_dist))

    if side == "long":
        stop_price = entry_price * (1.0 - stop_dist)
        take_profit = entry_price + (entry_price - stop_price)
    else:
        stop_price = entry_price * (1.0 + stop_dist)
        take_profit = entry_price - (stop_price - entry_price)
    return stop_price, take_profit, stop_dist * 100.0


def format_plan(side, entry_price, stop_price, take_profit, stop_pct, conf,
                bias_score, bias_label):
    verb = "BUY THE DIP" if side == "long" else "SELL THE RIP"
    icon = "\U0001F4C8" if side == "long" else "\U0001F4C9"
    filled = int(round(conf / 10.0))
    bar = "█" * filled + "░" * (10 - filled)
    return (
        f"\n{icon} <b>{verb} ({'LONG' if side == 'long' else 'SHORT'})</b>\n"
        f"Confidence <b>{conf:.0f}/100</b> <code>{bar}</code>\n"
        f"Entry ~{entry_price:.6g} | Stop {stop_price:.6g} | TP {take_profit:.6g} (+1R)\n"
        f"Risk {stop_pct:.2f}% to stop · reward {stop_pct:.2f}% to TP\n"
        f"Regime: {bias_label} ({bias_score:+.0f}/100) — trading with it"
    )


def spike_has_edge(inst_id, candles, spike_side):
    """Require order-flow/structure support before emitting the signal."""
    try:
        import chart_command as cc
    except Exception:
        return True

    if not candles:
        return False

    chart_candles = [
        {
            "ts": int(c[0]),
            "o": float(c[1]),
            "h": float(c[2]),
            "l": float(c[3]),
            "c": float(c[4]),
            "v": float(c[5]),
        }
        for c in reversed(candles)
        if len(c) >= 6
    ]
    if not chart_candles:
        return False

    ccy = inst_id.split("-")[0]
    cvd_series = cc.fetch_cvd(ccy)
    if not cvd_series:
        return False

    cvd_values = cc.resample_to_candles(cvd_series, chart_candles, "cvd")
    if not cvd_values or all(v is None for v in cvd_values):
        return False

    cvd_tag, _ = cc.cvd_read(chart_candles, cvd_values)
    if cvd_tag is None:
        return False

    highs, lows = cc.causal_pivots(chart_candles, n=3)
    st = cc.structure_state(chart_candles, highs, lows)

    if spike_side == "long":
        return cvd_tag == "confirmed up" and st.state != "downtrend"
    return cvd_tag == "confirmed down" and st.state != "uptrend"


def acquire_lock():
    """Single instance only. Every bot in this repo was running TWICE (two
    identical PIDs from the scheduler), which doubles every alert — the same
    class of bug the vbs launcher fix addressed. A lock older than 120s is
    assumed stale (killed process) and reclaimed."""
    now = time.time()
    try:
        if LOCK_PATH.exists():
            age = now - LOCK_PATH.stat().st_mtime
            if age < 120:
                print(f"[lock] another screener is running (lock {age:.0f}s old) — exiting")
                return False
            print(f"[lock] reclaiming stale lock ({age:.0f}s old)")
        LOCK_PATH.write_text(str(now), encoding="utf-8")
        return True
    except OSError as e:
        print(f"[lock] could not take lock ({e}) — continuing anyway")
        return True


def list_suspects(quote_filter, ratio_max=0.75):
    """Rank perps by weekend/weekday turnover ratio, lowest first.

    A REVIEW AID, NOT A CLASSIFIER. Tokenized equities trade 24/7 on OKX, so
    there is no clean structural tell (see note 5 in the module docstring); they
    merely go quieter at weekends when their underlying cash market is shut.
    Measured: IREN 0.62, CRWV 0.45, BTC 0.88, SOL 1.00. The bands overlap, so
    read the output and decide — do not paste it in blind.
    """
    rows = fetch_tickers(quote_filter)
    print(f"Scoring {len(rows)} perps by weekend/weekday turnover "
          f"(takes a few minutes)...\n")
    scored = []
    for r in rows:
        inst = r["instId"]
        try:
            data = http_get_json(
                f"{OKX_BASE}/api/v5/market/candles?instId={inst}&bar=1D&limit=21"
            ).get("data", [])
        except Exception:  # noqa: BLE001
            continue
        if len(data) < 14:
            continue
        we, wd = [], []
        for c in data:
            dt = datetime.fromtimestamp(int(c[0]) / 1000, timezone.utc)
            try:
                q = float(c[7] or 0)
            except (TypeError, ValueError):
                continue
            (we if dt.weekday() >= 5 else wd).append(q)
        if not we or not wd:
            continue
        aw, ad = sum(we) / len(we), sum(wd) / len(wd)
        if ad <= 0:
            continue
        scored.append((aw / ad, inst.split("-")[0]))
        time.sleep(0.05)

    scored.sort()
    flagged = [s for ratio, s in scored if ratio <= ratio_max]
    print(f"{'symbol':<12}{'weekend/weekday turnover':>26}")
    for ratio, sym in scored[:40]:
        mark = "  <-- review" if ratio <= ratio_max else ""
        print(f"  {sym:<10}{ratio:>24.3f}{mark}")
    print(f"\n{len(flagged)} below {ratio_max}. Review these, then merge the ones "
          f"that really are equities into config.json exclude_bases:\n")
    print(f'  "exclude_bases": {json.dumps(sorted(set(flagged)))}')


def price_n_seconds_ago(history, now, window):
    past = None
    for ts, price in history:
        if now - ts >= window:
            past = price
        else:
            break
    return past


def main():
    cfg = load_config()
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    quote_filter = cfg.get("quote_filter") or None
    poll_interval = cfg.get("poll_interval_seconds", 15)
    price_thresholds = {int(k): float(v) for k, v in cfg.get("price_thresholds", {}).items()}
    max_window = max(price_thresholds) if price_thresholds else 300
    vol_multiplier = cfg.get("volume_spike_multiplier", 3.0)
    cooldown = cfg.get("cooldown_seconds", 900)

    scfg = cfg.get("sizing", {})
    min_conf = float(cfg.get("min_confidence", 60.0))
    exclude = classified_equities() or set(cfg.get("exclude_bases",
                                                   DEFAULT_EXCLUDE_BASES))
    bias_refresh = float(cfg.get("bias_refresh_seconds", 1800))
    alert_when_no_plan = bool(cfg.get("alert_when_no_plan", False))

    history = defaultdict(deque)  # instId -> deque[(ts, last_price)]
    last_alert = {}  # instId -> ts of last alert
    turnover = {}    # instId -> 24h turnover USD, refreshed from the ticker rows

    # /bias is an hourly regime read, not a per-tick one, so it is fetched on its
    # own slow timer and reused. A failure leaves score 0 -> regime_side() None
    # -> no plans, which is the correct fail-safe: without a regime we have no
    # basis for choosing a side.
    bias = {"score": 0.0, "label": "UNKNOWN", "fresh": False}
    bias_checked = 0.0

    def refresh_bias(now):
        nonlocal bias, bias_checked
        if now - bias_checked < bias_refresh:
            return
        bias_checked = now
        try:
            from bias_telegram import current_bias
            bias = current_bias()
            print(f"[bias] {bias['label']} {bias['score']:+.0f}/100 "
                  f"(age {bias['age_min']:.0f}m)")
        except Exception as e:  # noqa: BLE001
            print(f"[bias] unavailable ({e}) — no directional plans until it returns")
            bias = {"score": 0.0, "label": "UNKNOWN", "fresh": False}

    print(
        f"Starting OKX perp screener. poll={poll_interval}s "
        f"quote_filter={quote_filter or 'ALL'} thresholds={price_thresholds} "
        f"vol_multiplier={vol_multiplier}x cooldown={cooldown}s "
        f"min_confidence={min_conf} excluded_bases={len(exclude)}"
    )
    refresh_bias(time.time())
    send_telegram(
        token, chat_id,
        f"\U0001F7E2 OKX perp screener started — confidence ≥ {min_conf:.0f}/100, "
        f"direction set by /bias (now {bias['label']} {bias['score']:+.0f})."
    )

    try:
        while True:
            loop_start = time.time()
            try:
                tickers = fetch_tickers(quote_filter)
            except Exception as e:
                print(f"[poll] fetch_tickers failed: {e}")
                time.sleep(poll_interval)
                continue

            now = time.time()
            refresh_bias(now)
            side_allowed = regime_side(bias["score"], cfg)

            for row in tickers:
                inst_id = row["instId"]
                if inst_id.split("-")[0] in exclude:
                    continue          # tokenized equity — not a crypto signal
                try:
                    last_price = float(row["last"])
                except (KeyError, ValueError, TypeError):
                    continue
                try:
                    turnover[inst_id] = float(row.get("volCcy24h") or 0) * last_price
                except (ValueError, TypeError):
                    turnover[inst_id] = 0.0

                dq = history[inst_id]
                dq.append((now, last_price))
                while dq and now - dq[0][0] > max_window + poll_interval:
                    dq.popleft()

                if inst_id in last_alert and now - last_alert[inst_id] < cooldown:
                    continue

                for window, threshold in price_thresholds.items():
                    past = price_n_seconds_ago(dq, now, window)
                    if past is None or past == 0:
                        continue
                    pct_change = (last_price - past) / past * 100
                    if abs(pct_change) < threshold:
                        continue

                    try:
                        candles = fetch_recent_candles(inst_id, bar="1m", limit=20)
                    except Exception as e:
                        print(f"[confirm] candle fetch failed for {inst_id}: {e}")
                        continue
                    if len(candles) < 6:
                        continue

                    current_vol = float(candles[0][5])
                    prior_vols = [float(c[5]) for c in candles[1:11]]
                    avg_prior_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 0
                    if avg_prior_vol <= 0:
                        continue
                    vol_ratio = current_vol / avg_prior_vol
                    if vol_ratio < vol_multiplier:
                        continue

                    # A spike is only tradeable if we would FADE it in the
                    # direction the regime already favours: DOWN spike + bullish
                    # regime = buy the dip; UP spike + bearish regime = sell the
                    # rip. Anything else (wrong side, or chop) is just news.
                    spike_side = "long" if pct_change < 0 else "short"
                    tradeable = side_allowed is not None and spike_side == side_allowed

                    conf = confidence(pct_change, threshold, vol_ratio,
                                      vol_multiplier, bias["score"],
                                      turnover.get(inst_id, 0.0), cfg)

                    if not tradeable:
                        if alert_when_no_plan:
                            direction = "\U0001F680 UP" if pct_change > 0 else "\U0001F53B DOWN"
                            send_telegram(token, chat_id, (
                                f"{direction} spike: <b>{inst_id}</b>\n"
                                f"{pct_change:+.2f}% over {window}s · "
                                f"vol {vol_ratio:.1f}x · no trade "
                                f"({bias['label']} regime)"
                            ))
                            last_alert[inst_id] = now
                        print(f"[skip] {inst_id} {pct_change:+.2f}% — "
                              f"{spike_side} spike, regime allows "
                              f"{side_allowed or 'nothing (chop)'}")
                        break

                    if conf < min_conf:
                        print(f"[skip] {inst_id} {pct_change:+.2f}% — "
                              f"confidence {conf:.0f} < {min_conf:.0f}")
                        break

                    if not spike_has_edge(inst_id, candles, spike_side):
                        print(f"[skip] {inst_id} {pct_change:+.2f}% — no edge from CVD/structure")
                        break

                    plan = build_reversion_plan(candles, last_price, side_allowed, scfg)
                    if plan is None:
                        break
                    stop_price, take_profit, stop_pct = plan

                    direction = "\U0001F680 UP" if pct_change > 0 else "\U0001F53B DOWN"
                    msg = (
                        f"{direction} spike: <b>{inst_id}</b>\n"
                        f"{pct_change:+.2f}% over {window}s\n"
                        f"Last: {last_price}\n"
                        f"1m volume: {vol_ratio:.1f}x avg (last {len(prior_vols)} candles)"
                    )
                    msg += format_plan(side_allowed, last_price, stop_price,
                                       take_profit, stop_pct, conf,
                                       bias["score"], bias["label"])

                    send_telegram(token, chat_id, msg)
                    last_alert[inst_id] = now
                    print(f"[alert] {inst_id} {pct_change:+.2f}% / {window}s, "
                          f"vol {vol_ratio:.1f}x, conf {conf:.0f}, {side_allowed}")
                    time.sleep(0.1)  # small throttle before next candle-confirm fetch
                    break

            # Keep the lock's mtime current, or a second instance would decide
            # after 120s that this one had died and start alerting alongside it.
            try:
                LOCK_PATH.write_text(str(time.time()), encoding="utf-8")
            except OSError:
                pass

            elapsed = time.time() - loop_start
            time.sleep(max(0.5, poll_interval - elapsed))
    except KeyboardInterrupt:
        print("Stopping screener.")
        send_telegram(token, chat_id, "\U0001F534 OKX perp screener stopped.")


if __name__ == "__main__":
    if "--list-suspects" in sys.argv:
        _cfg = load_config()
        list_suspects(_cfg.get("quote_filter") or None)
    elif acquire_lock():
        try:
            main()
        finally:
            try:
                LOCK_PATH.unlink()
            except OSError:
                pass
