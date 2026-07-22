#!/usr/bin/env python3
"""Fetch all raw data needed for the HYPE trigger/correlation study.

Sources (all publicly accessible):
  1. Binance public market data - S3 bucket `data.binance.vision`
     (accessed path-style via s3.ap-northeast-1.amazonaws.com because the
      CDN hostname may be blocked in restricted environments):
       - USD-M futures klines for HYPEUSDT (1d + 1h), BTC/ETH/SOL (1d)
       - HYPEUSDT funding rates (8h)
       - HYPEUSDT daily "metrics" files: open interest, top-trader
         long/short ratios, global long/short ratio, taker buy/sell ratio
  2. CoinMetrics community data (GitHub): BTC/ETH on-chain daily metrics
     (exchange flows, MVRV, active addresses) and HYPE spot volume.
  3. datahub `finance-vix` (GitHub): daily VIX (macro risk / inverse-SPX proxy).

Everything lands in research/hype_correlations/data/raw/.
Run build_dataset.py afterwards to produce the merged daily dataset.
"""
import concurrent.futures as cf
import io
import os
import sys
import zipfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
S3 = "https://s3.ap-northeast-1.amazonaws.com/data.binance.vision"
RAWGH = "https://raw.githubusercontent.com"

FUT_MONTHS_HYPE = [f"2025-{m:02d}" for m in range(5, 13)] + [f"2026-{m:02d}" for m in range(1, 7)]
JULY_2026_DAYS = [f"2026-07-{d:02d}" for d in range(1, 22)]
METRIC_DAYS = None  # filled in main()


def _get(url, dest, ok404=False):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "cached"
    r = requests.get(url, timeout=60)
    if r.status_code == 404 and ok404:
        return "404"
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    return "ok"


def fetch_zip_csv(url, dest_csv, ok404=False):
    """Download a Binance .zip and store the contained CSV."""
    if os.path.exists(dest_csv) and os.path.getsize(dest_csv) > 0:
        return "cached"
    r = requests.get(url, timeout=120)
    if r.status_code == 404 and ok404:
        return "404"
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    name = zf.namelist()[0]
    with open(dest_csv, "wb") as f:
        f.write(zf.read(name))
    return "ok"


def binance_jobs():
    jobs = []  # (url, dest, ok404)
    os.makedirs(os.path.join(RAW, "klines"), exist_ok=True)
    os.makedirs(os.path.join(RAW, "funding"), exist_ok=True)
    os.makedirs(os.path.join(RAW, "metrics"), exist_ok=True)

    def monthly_kline(sym, iv, month):
        url = f"{S3}/data/futures/um/monthly/klines/{sym}/{iv}/{sym}-{iv}-{month}.zip"
        dest = os.path.join(RAW, "klines", f"{sym}-{iv}-{month}.csv")
        return (url, dest, True)

    def daily_kline(sym, iv, day):
        url = f"{S3}/data/futures/um/daily/klines/{sym}/{iv}/{sym}-{iv}-{day}.zip"
        dest = os.path.join(RAW, "klines", f"{sym}-{iv}-{day}.csv")
        return (url, dest, True)

    for m in FUT_MONTHS_HYPE:
        jobs.append(monthly_kline("HYPEUSDT", "1d", m))
        jobs.append(monthly_kline("HYPEUSDT", "1h", m))
        url = f"{S3}/data/futures/um/monthly/fundingRate/HYPEUSDT/HYPEUSDT-fundingRate-{m}.zip"
        jobs.append((url, os.path.join(RAW, "funding", f"HYPEUSDT-fundingRate-{m}.csv"), True))
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            jobs.append(monthly_kline(sym, "1d", m))
    for d in JULY_2026_DAYS:
        jobs.append(daily_kline("HYPEUSDT", "1d", d))
        jobs.append(daily_kline("HYPEUSDT", "1h", d))
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            jobs.append(daily_kline(sym, "1d", d))
    for d in METRIC_DAYS:
        url = f"{S3}/data/futures/um/daily/metrics/HYPEUSDT/HYPEUSDT-metrics-{d}.zip"
        jobs.append((url, os.path.join(RAW, "metrics", f"HYPEUSDT-metrics-{d}.csv"), True))
    return jobs


def main():
    global METRIC_DAYS
    import pandas as pd

    os.makedirs(RAW, exist_ok=True)
    METRIC_DAYS = [d.strftime("%Y-%m-%d")
                   for d in pd.date_range("2025-05-30", "2026-07-21", freq="D")]

    # --- GitHub-hosted CSVs ---
    gh = {
        "coinmetrics_btc.csv": f"{RAWGH}/coinmetrics/data/master/csv/btc.csv",
        "coinmetrics_eth.csv": f"{RAWGH}/coinmetrics/data/master/csv/eth.csv",
        "coinmetrics_hype.csv": f"{RAWGH}/coinmetrics/data/master/csv/hype.csv",
        "vix_daily.csv": f"{RAWGH}/datasets/finance-vix/main/data/vix-daily.csv",
    }
    for name, url in gh.items():
        print(name, _get(url, os.path.join(RAW, name)))

    # --- Binance S3 ---
    jobs = binance_jobs()
    print(f"{len(jobs)} binance files to fetch...")
    done = {"ok": 0, "cached": 0, "404": 0}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch_zip_csv, u, d, k): (u, d) for u, d, k in jobs}
        for f in cf.as_completed(futs):
            try:
                done[f.result()] += 1
            except Exception as e:
                print("FAIL", futs[f][0], e, file=sys.stderr)
    print("binance:", done)


if __name__ == "__main__":
    main()
