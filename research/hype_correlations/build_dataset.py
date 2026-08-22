#!/usr/bin/env python3
"""Merge raw files into one daily feature dataset for the HYPE trigger study.

Output: data/hype_daily_dataset.csv  (one row per UTC day)
        data/hype_1h.csv             (hourly klines, for the spike study)
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data")

KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def load_klines(pattern):
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "klines", pattern))):
        d = pd.read_csv(f, header=None, names=KCOLS, skiprows=1)
        frames.append(d)
    k = pd.concat(frames, ignore_index=True)
    k = k[pd.to_numeric(k.open_time, errors="coerce").notna()].astype({"open_time": "int64"})
    k["date"] = pd.to_datetime(k.open_time, unit="ms")
    k = k.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    num = [c for c in KCOLS if c not in ("open_time", "close_time", "ignore")]
    k[num] = k[num].astype(float)
    return k


def main():
    # ---- HYPE daily klines ----
    h = load_klines("HYPEUSDT-1d-*.csv")
    h["date"] = h.date.dt.normalize()
    hype = h.set_index("date")[["open", "high", "low", "close", "volume",
                                "quote_volume", "count", "taker_buy_quote_volume"]]
    hype.columns = ["h_open", "h_high", "h_low", "h_close", "h_vol",
                    "h_qvol", "h_trades", "h_taker_buy_qvol"]

    # ---- cross-asset daily closes ----
    for sym, tag in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol")):
        k = load_klines(f"{sym}-1d-*.csv")
        k["date"] = k.date.dt.normalize()
        hype[f"{tag}_close"] = k.set_index("date").close
        hype[f"{tag}_qvol"] = k.set_index("date").quote_volume

    # ---- funding (4h/8h events -> daily) ----
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, "funding", "*.csv")))]
    fu = pd.concat(frames, ignore_index=True)
    fu["date"] = pd.to_datetime(fu.calc_time, unit="ms").dt.normalize()
    g = fu.groupby("date").last_funding_rate
    hype["funding_sum"] = g.sum()          # total funding paid that day
    hype["funding_last"] = g.last()        # last print of the day

    # ---- metrics: OI, long/short ratios, taker vol ratio (5m -> daily) ----
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "metrics", "*.csv"))):
        d = pd.read_csv(f)
        frames.append(d)
    me = pd.concat(frames, ignore_index=True)
    me["ts"] = pd.to_datetime(me.create_time)
    me["date"] = me.ts.dt.normalize()
    gm = me.groupby("date")
    hype["oi"] = gm.sum_open_interest.last()                 # HYPE contracts
    hype["oi_usd"] = gm.sum_open_interest_value.last()
    hype["top_ls_acct"] = gm.count_toptrader_long_short_ratio.mean()
    hype["top_ls_pos"] = gm.sum_toptrader_long_short_ratio.mean()
    hype["global_ls"] = gm.count_long_short_ratio.mean()
    hype["taker_ls_vol"] = gm.sum_taker_long_short_vol_ratio.mean()

    # ---- VIX ----
    vix = pd.read_csv(os.path.join(RAW, "vix_daily.csv"), parse_dates=["DATE"])
    vix = vix.set_index("DATE").rename(columns={"CLOSE": "vix"})[["vix"]]
    hype = hype.join(vix, how="left")
    hype["vix"] = hype.vix.ffill()  # weekends/holidays carry last close

    # ---- CoinMetrics BTC/ETH on-chain ----
    for asset in ("btc", "eth"):
        cm = pd.read_csv(os.path.join(RAW, f"coinmetrics_{asset}.csv"),
                         parse_dates=["time"], low_memory=False)
        cm = cm.set_index("time")
        cols = {}
        cols[f"{asset}_ex_netflow_usd"] = cm.FlowInExUSD - cm.FlowOutExUSD
        cols[f"{asset}_mvrv"] = cm.CapMVRVCur
        cols[f"{asset}_adract"] = cm.AdrActCnt
        hype = hype.join(pd.DataFrame(cols), how="left")

    # ---- CoinMetrics HYPE spot volume (all exchanges, not just Binance) ----
    cmh = pd.read_csv(os.path.join(RAW, "coinmetrics_hype.csv"), parse_dates=["time"])
    hype = hype.join(cmh.set_index("time")[["volume_reported_spot_usd_1d"]]
                     .rename(columns={"volume_reported_spot_usd_1d": "h_spot_vol_all"}),
                     how="left")

    hype.index.name = "date"
    hype.to_csv(os.path.join(OUT, "hype_daily_dataset.csv"))
    print("daily dataset:", hype.shape, hype.index.min().date(), "->", hype.index.max().date())
    print("null counts (top):")
    print(hype.isna().sum().sort_values(ascending=False).head(8).to_string())

    # ---- hourly klines for spike study ----
    h1 = load_klines("HYPEUSDT-1h-*.csv")
    h1 = h1[["date", "open", "high", "low", "close", "volume", "quote_volume",
             "taker_buy_quote_volume"]]
    h1.to_csv(os.path.join(OUT, "hype_1h.csv"), index=False)
    print("hourly dataset:", h1.shape)


if __name__ == "__main__":
    main()
