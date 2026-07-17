"""Quick test: verify OKX API is working and show sample data."""

import json
import urllib.request

OKX_BASE = "https://www.okx.com"
USER_AGENT = "growhf-bot/1.0"

def http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

print("Testing OKX API...")

# Fetch tickers
data = http_get_json(f"{OKX_BASE}/api/v5/market/tickers?instType=SWAP")
tickers = data.get("data", [])

# Check format
if tickers:
    print(f"Sample instId formats:")
    for t in tickers[:3]:
        print(f"  {t['instId']}")

usdt_tickers = [t for t in tickers if "-USDT-SWAP" in t["instId"]]

print(f"\n[OK] Fetched {len(tickers)} SWAP instruments")
print(f"[OK] {len(usdt_tickers)} USDT-quoted pairs\n")

# Show sample data
if usdt_tickers:
    sample = usdt_tickers[0]
    print(f"Sample ticker: {sample['instId']}")
    print(f"  Last price: {sample['last']}")
    print(f"  24h volume: {sample['volCcy24h']}")
    print(f"  24h high: {sample['high24h']}")
    print(f"  24h low: {sample['low24h']}")

    # Fetch candles for this pair
    inst_id = sample["instId"]
    candles = http_get_json(f"{OKX_BASE}/api/v5/market/candles?instId={inst_id}&bar=1m&limit=5")
    data = candles.get("data", [])

    if data:
        print(f"\n[OK] Fetched candles for {inst_id}")
        print(f"  Latest 1m candle (most recent):")
        ts, o, h, l, c, vol = data[0][:6]
        print(f"    Time: {ts}, Close: {c}, Volume: {vol}")

print("\n[OK] OKX API is working!")
print("\nThresholds in config.json:")
print("  60s:  3% move required")
print("  300s: 5% move required")
print("  Volume: 3x average\n")
print("To detect more signals:")
print("  - Lower thresholds: 60s to 2%, 300s to 3.5%")
print("  - Reduce volume multiplier: 3x to 2.5x")
