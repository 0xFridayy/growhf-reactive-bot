"""
alpha_ml/data.py — historical OHLCV fetcher for the alpha-research pipeline.

Pulls OKX public candles (no API key needed — same endpoints okx_tele_bot.py
and growhf_reactive_bot.py already use), paginating backward with
/market/history-candles to build a deep-enough series for training/backtesting,
and caches the result to CSV so repeated runs don't re-hit the API.

Run:
    py data.py --inst BTC-USDT-SWAP --bar 5m --bars 5000
Deps: requests, pandas
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

OKX_BASE = "https://www.okx.com"
USER_AGENT = "growhf-alpha-ml/1.0"
DATA_DIR = Path(__file__).with_name("data")

CANDLE_COLS = ["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"]


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


SESSION = _session()


def _get(path, params, timeout=15):
    r = SESSION.get(f"{OKX_BASE}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("data", [])


def _rows_to_df(rows):
    if not rows:
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(rows, columns=CANDLE_COLS[: len(rows[0])])
    df = df[["ts", "o", "h", "l", "c", "vol"]].rename(columns={"vol": "v"})
    for col in ("o", "h", "l", "c", "v"):
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def cache_path(inst_id, bar):
    return DATA_DIR / f"{inst_id}_{bar}.csv"


def load_cache(inst_id, bar):
    p = cache_path(inst_id, bar)
    if not p.exists():
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])
    df = pd.read_csv(p, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def save_cache(df, inst_id, bar):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values("ts").drop_duplicates("ts").to_csv(cache_path(inst_id, bar), index=False)


def fetch_ohlcv(inst_id, bar="5m", total_bars=2000, use_cache=True, sleep=0.15):
    """Historical OHLCV for one instrument, newest bars included, oldest-first.

    Combines /market/candles (recent, incl. in-progress bar) with paginated
    /market/history-candles (older, confirmed bars) up to `total_bars` rows.
    Cached to CSV so re-runs only fetch what's missing.
    """
    cached = load_cache(inst_id, bar) if use_cache else pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])

    recent = _rows_to_df(_get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": 300}))
    df = pd.concat([cached, recent], ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    oldest_ts_ms = None
    while len(df) < total_bars:
        before_ts = oldest_ts_ms if oldest_ts_ms is not None else int(df["ts"].min().value // 10**6)
        rows = _get(
            "/api/v5/market/history-candles",
            {"instId": inst_id, "bar": bar, "limit": 100, "after": before_ts},
        )
        if not rows:
            break
        batch = _rows_to_df(rows)
        df = pd.concat([batch, df], ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        oldest_ts_ms = int(batch["ts"].min().value // 10**6)
        time.sleep(sleep)

    if use_cache:
        save_cache(df, inst_id, bar)

    if len(df) > total_bars:
        df = df.iloc[-total_bars:].reset_index(drop=True)
    return df


def fetch_funding_history(inst_id, total=200, sleep=0.15):
    """Historical funding rates -> DataFrame[ts, funding], oldest-first."""
    out, before_ts = [], None
    while len(out) < total:
        params = {"instId": inst_id, "limit": 100}
        if before_ts is not None:
            params["after"] = before_ts
        rows = _get("/api/v5/public/funding-rate-history", params)
        if not rows:
            break
        out.extend(rows)
        before_ts = rows[-1]["fundingTime"]
        time.sleep(sleep)
    if not out:
        return pd.DataFrame(columns=["ts", "funding"])
    df = pd.DataFrame(out)[["fundingTime", "fundingRate"]].rename(
        columns={"fundingTime": "ts", "fundingRate": "funding"}
    )
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df["funding"] = df["funding"].astype(float)
    return df.sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="5m")
    ap.add_argument("--bars", type=int, default=2000)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    df = fetch_ohlcv(args.inst, args.bar, args.bars, use_cache=not args.no_cache)
    print(f"{args.inst} {args.bar}: {len(df)} bars, {df['ts'].min()} -> {df['ts'].max()}")
    print(f"cached -> {cache_path(args.inst, args.bar)}")
