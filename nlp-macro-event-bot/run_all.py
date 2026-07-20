"""
RUN_ALL — one-process supervisor for the whole stack
====================================================
Launches every long-running module, restarts any that crash (capped backoff),
and runs the two periodic snapshot jobs in-process. One terminal, one Ctrl+C.

  Subprocesses (each its own script, auto-restarted):
    macro_event_bot.py      needs MACRO_BOT_TOKEN + MACRO_BOT_CHAT_ID
    news_nlp_bot.py         needs those + ANTHROPIC_API_KEY
    feeds_ws.py             always (Tree free tier; Phoenix if PHOENIX_API_KEY)
    telegram_listener.py    needs TELEGRAM_API_ID + TELEGRAM_API_HASH

  In-process periodic jobs (stdlib threads, no extra deps):
    sentiment_engine.recompute_and_store()   every 60s
    data_sources.snapshot_regime_features()  every 3600s

A service whose required credentials are missing is SKIPPED with a clear note,
so a partial config still runs everything it can.

  py run_all.py --dry-run     # print the launch plan + preflight, then exit
  py run_all.py               # run everything enabled
  py run_all.py --only feeds_ws,sentiment   # run a subset
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

from shared import (have, BOT_TOKEN, CHAT_ID, ANTHROPIC_KEY, PHOENIX_KEY,
                    TREE_KEY, WIB)

API_ID   = os.environ.get("TELEGRAM_API_ID", "")
API_HASH = os.environ.get("TELEGRAM_API_HASH", "")

RESTART_BACKOFF_MAX = 60     # seconds
TRAINING_INTERVAL = int(os.environ.get("TRAINING_INTERVAL", "300"))  # 5m bars

# name -> (script, required-creds-ok predicate, human note)
SERVICES = {
    "macro_event_bot": (
        "macro_event_bot.py",
        lambda: have(BOT_TOKEN) and have(CHAT_ID),
        "Telegram token + chat id"),
    "news_nlp_bot": (
        "news_nlp_bot.py",
        lambda: have(BOT_TOKEN) and have(CHAT_ID) and have(ANTHROPIC_KEY),
        "Telegram + ANTHROPIC_API_KEY"),
    "feeds_ws": (
        "feeds_ws.py",
        lambda: True,
        "always on (Tree free" + (", Phoenix keyed" if have(PHOENIX_KEY) else "") + ")"),
    "telegram_listener": (
        "telegram_listener.py",
        lambda: bool(API_ID and API_HASH),
        "TELEGRAM_API_ID + TELEGRAM_API_HASH"),
}

_stop = threading.Event()


# ----------------------------------------------------------------------
# PERIODIC IN-PROCESS JOBS
# ----------------------------------------------------------------------
def _periodic(name, fn, interval, jitter_first=1):
    """Run fn() every `interval` seconds until stop is set."""
    time.sleep(jitter_first)
    while not _stop.is_set():
        try:
            fn()
        except Exception as e:
            print(f"[{name}] error: {e}")
        _stop.wait(interval)


def start_periodic_jobs(enabled):
    threads = []
    if "sentiment" in enabled:
        from sentiment_engine import recompute_and_store

        def _sent():
            snap = recompute_and_store()
            sw = snap.get("swing", {})
            print(f"[sentiment] swing net_tone={sw.get('net_tone')} "
                  f"decayed_impact={sw.get('decayed_impact')} n={sw.get('n_events')}")
        t = threading.Thread(target=_periodic, args=("sentiment", _sent, 60),
                             daemon=True, name="sentiment")
        t.start(); threads.append(t)

    if "regime" in enabled:
        from data_sources import snapshot_regime_features

        def _regime():
            feat = snapshot_regime_features()
            print(f"[regime] fear&greed={feat.get('fng')} ({feat.get('fng_label')})")
        t = threading.Thread(target=_periodic, args=("regime", _regime, 3600),
                             daemon=True, name="regime")
        t.start(); threads.append(t)

    if "training" in enabled:
        from training_snapshot import snapshot

        def _training():
            snapshot(symbol=os.environ.get("PERP_SYMBOL", "BTCUSDT"),
                     interval=os.environ.get("PERP_INTERVAL", "5m"),
                     label_horizon=int(os.environ.get("LABEL_HORIZON", "1")))
        # Cadence matches the bar interval (default 5m). Snapshot is idempotent.
        t = threading.Thread(target=_periodic,
                             args=("training", _training, TRAINING_INTERVAL),
                             daemon=True, name="training")
        t.start(); threads.append(t)
    return threads


# ----------------------------------------------------------------------
# SUBPROCESS SUPERVISION
# ----------------------------------------------------------------------
def _supervise(name, script):
    """Keep `script` alive: relaunch on exit with capped exponential backoff."""
    backoff = 1
    while not _stop.is_set():
        print(f"[{name}] starting ({script})")
        try:
            proc = subprocess.Popen([sys.executable, script])
        except Exception as e:
            print(f"[{name}] failed to launch: {e}")
            _stop.wait(backoff); backoff = min(backoff * 2, RESTART_BACKOFF_MAX)
            continue
        _CHILDREN[name] = proc
        start = time.monotonic()
        while proc.poll() is None and not _stop.is_set():
            time.sleep(0.5)
        if _stop.is_set():
            _terminate(proc)
            return
        ran = time.monotonic() - start
        backoff = 1 if ran > 30 else min(backoff * 2, RESTART_BACKOFF_MAX)
        print(f"[{name}] exited (code {proc.returncode}) after {ran:.0f}s — "
              f"restart in {backoff}s")
        _stop.wait(backoff)


_CHILDREN = {}


def _terminate(proc):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


# ----------------------------------------------------------------------
# PREFLIGHT
# ----------------------------------------------------------------------
def preflight(only):
    print("=" * 62)
    print(f"  RUN_ALL preflight — {datetime.now(tz=WIB):%Y-%m-%d %H:%M} WIB")
    print("=" * 62)
    enabled_procs, skipped = [], []
    for name, (script, ok, note) in SERVICES.items():
        if only and name not in only:
            continue
        if ok():
            enabled_procs.append(name)
            print(f"  [ON ] {name:<18} {note}")
        else:
            skipped.append(name)
            print(f"  [skip] {name:<18} missing: {note}")

    periodic = []
    _psrc = {"sentiment": "sentiment_engine every 60s",
             "regime": "Fear&Greed every 1h",
             "training": f"training-table snapshot every {TRAINING_INTERVAL}s"}
    for p in ("sentiment", "regime", "training"):
        if not only or p in only:
            periodic.append(p)
            print(f"  [ON ] {p:<18} (in-process, {_psrc[p]})")
    print("=" * 62)
    return enabled_procs, periodic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print launch plan + preflight, then exit")
    ap.add_argument("--only", default="",
                    help="comma list: macro_event_bot,news_nlp_bot,feeds_ws,"
                         "telegram_listener,sentiment,regime")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None

    enabled_procs, periodic = preflight(only)

    if args.dry_run:
        print("dry-run: nothing launched.")
        return

    def _sig(_signum, _frame):
        print("\nshutting down…")
        _stop.set()
    signal.signal(signal.SIGINT, _sig)
    try:
        signal.signal(signal.SIGTERM, _sig)
    except (ValueError, AttributeError):
        pass

    start_periodic_jobs(periodic)
    sup_threads = []
    for name in enabled_procs:
        script = SERVICES[name][0]
        t = threading.Thread(target=_supervise, args=(name, script),
                             daemon=True, name=name)
        t.start(); sup_threads.append(t)

    print(f"\nrun_all up: {len(enabled_procs)} service(s) + {len(periodic)} "
          f"periodic job(s). Ctrl+C to stop.\n")
    try:
        while not _stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        _stop.set()
    finally:
        _stop.set()
        for proc in _CHILDREN.values():
            _terminate(proc)
        print("stopped.")


if __name__ == "__main__":
    main()
