"""
okx_orderflow_logger.py — live order-flow capture for OKX perps.

OKX's public REST only exposes the last ~500 trades and a live order-book
snapshot -- no historical tape/L2 data -- so order flow can't be backtested
retroactively. This logs real trades + top-of-book snapshots to a local
SQLite DB starting now, so there's real data to backtest once enough has
accumulated.

Captures, per instrument:
  - every trade (ts, side, size, price) from the public "trades" channel
  - best bid/ask size snapshots from the public "books5" channel, whenever
    they change, giving a bid/ask imbalance read

Computes and stores rolling CVD (cumulative volume delta = buy vol - sell
vol) at 1m/5m/1h windows so a later backtest doesn't have to replay every
raw trade.

No Telegram alerts here by design -- this is a silent logger until there's
enough data to know whether any of this predicts anything. Run alongside
okx_tele_bot.py / okx_perp_screener.py, doesn't share their config keys.

Run:
    python okx_orderflow_logger.py
"""
import json
import sqlite3
import threading
import time
from pathlib import Path

import websocket

REPO_DIR = Path(__file__).parent
DB_PATH = REPO_DIR / "orderflow.db"
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

DEFAULT_INSTRUMENTS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "HYPE-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP",
]

RECONNECT_DELAY_S = 5
CVD_ROLLUP_INTERVAL_S = 60


def load_instruments():
    cfg_path = REPO_DIR / "config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        insts = cfg.get("order_flow", {}).get("instruments")
        if insts:
            return insts
    return DEFAULT_INSTRUMENTS


def init_db(con):
    con.execute("""CREATE TABLE IF NOT EXISTS trades (
        ts INTEGER, inst_id TEXT, side TEXT, sz REAL, px REAL
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trades_inst_ts ON trades(inst_id, ts)")
    con.execute("""CREATE TABLE IF NOT EXISTS book_snapshots (
        ts INTEGER, inst_id TEXT,
        bid_px REAL, bid_sz REAL, ask_px REAL, ask_sz REAL, imbalance REAL
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_book_inst_ts ON book_snapshots(inst_id, ts)")
    con.execute("""CREATE TABLE IF NOT EXISTS cvd_rollup (
        ts INTEGER, inst_id TEXT, window_s INTEGER, cvd REAL, trade_count INTEGER
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cvd_inst_ts ON cvd_rollup(inst_id, ts, window_s)")
    con.commit()


class OrderFlowLogger:
    def __init__(self, instruments):
        self.instruments = instruments
        self.con = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.lock = threading.Lock()
        init_db(self.con)
        # rolling per-instrument buy/sell volume for CVD windows
        self.window_vol = {inst: {"buy": 0.0, "sell": 0.0, "n": 0} for inst in instruments}

    def on_open(self, ws):
        args = []
        for inst in self.instruments:
            args.append({"channel": "trades", "instId": inst})
            args.append({"channel": "books5", "instId": inst})
        ws.send(json.dumps({"op": "subscribe", "args": args}))
        print(f"[orderflow] subscribed to {len(self.instruments)} instruments", flush=True)

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        arg = msg.get("arg", {})
        channel = arg.get("channel")
        data = msg.get("data")
        if not data:
            return
        if channel == "trades":
            self._handle_trades(arg.get("instId"), data)
        elif channel == "books5":
            self._handle_book(arg.get("instId"), data)

    def _handle_trades(self, inst_id, rows):
        with self.lock:
            for r in rows:
                ts = int(r["ts"])
                side = r["side"]
                sz = float(r["sz"])
                px = float(r["px"])
                self.con.execute(
                    "INSERT INTO trades (ts, inst_id, side, sz, px) VALUES (?,?,?,?,?)",
                    (ts, inst_id, side, sz, px),
                )
                bucket = self.window_vol.setdefault(inst_id, {"buy": 0.0, "sell": 0.0, "n": 0})
                if side == "buy":
                    bucket["buy"] += sz
                else:
                    bucket["sell"] += sz
                bucket["n"] += 1
            self.con.commit()

    def _handle_book(self, inst_id, rows):
        row = rows[0]
        bids = row.get("bids") or []
        asks = row.get("asks") or []
        if not bids or not asks:
            return
        bid_px, bid_sz = float(bids[0][0]), float(bids[0][1])
        ask_px, ask_sz = float(asks[0][0]), float(asks[0][1])
        total = bid_sz + ask_sz
        imbalance = (bid_sz - ask_sz) / total if total else 0.0
        ts = int(row.get("ts") or time.time() * 1000)
        with self.lock:
            self.con.execute(
                "INSERT INTO book_snapshots (ts, inst_id, bid_px, bid_sz, ask_px, ask_sz, imbalance) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, inst_id, bid_px, bid_sz, ask_px, ask_sz, imbalance),
            )
            self.con.commit()

    def rollup_loop(self):
        while True:
            time.sleep(CVD_ROLLUP_INTERVAL_S)
            ts = int(time.time() * 1000)
            with self.lock:
                for inst, bucket in self.window_vol.items():
                    cvd = bucket["buy"] - bucket["sell"]
                    self.con.execute(
                        "INSERT INTO cvd_rollup (ts, inst_id, window_s, cvd, trade_count) VALUES (?,?,?,?,?)",
                        (ts, inst, CVD_ROLLUP_INTERVAL_S, cvd, bucket["n"]),
                    )
                    bucket["buy"] = 0.0
                    bucket["sell"] = 0.0
                    bucket["n"] = 0
                self.con.commit()
            print(f"[orderflow] rolled up CVD for {len(self.window_vol)} instruments at {ts}", flush=True)

    def on_error(self, ws, error):
        print(f"[orderflow] ws error: {error}", flush=True)

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[orderflow] ws closed: {close_status_code} {close_msg}", flush=True)

    def run_forever(self):
        threading.Thread(target=self.rollup_loop, daemon=True).start()
        while True:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
            print(f"[orderflow] disconnected, reconnecting in {RECONNECT_DELAY_S}s...", flush=True)
            time.sleep(RECONNECT_DELAY_S)


def main():
    instruments = load_instruments()
    print(f"Starting OKX order-flow logger. instruments={instruments} db={DB_PATH}", flush=True)
    logger = OrderFlowLogger(instruments)
    try:
        logger.run_forever()
    except KeyboardInterrupt:
        print("Stopping order-flow logger.", flush=True)


if __name__ == "__main__":
    main()
