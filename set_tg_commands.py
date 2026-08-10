"""Register the bot's command list with Telegram so typing "/" autocompletes.

Run once after adding or renaming a command:
    python set_tg_commands.py
"""

import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    TOKEN = json.load(f)["telegram_bot_token"]

COMMANDS = [
    {"command": "bias",
     "description": "Market bias now — TOTAL3, BTC.D, USDT.D + macro"},
    {"command": "analyze",
     "description": "Full read + verdict, e.g. /analyze BTC"},
    {"command": "profile",
     "description": "TPO profile — POC / VAH / VAL, e.g. /profile BTC"},
    {"command": "regime",
     "description": "Regime + mean-reversion read, e.g. /regime BTC"},
    {"command": "chart",
     "description": "Render a 4H OKX chart for a symbol, e.g. /chart BTC"},
    {"command": "status", "description": "What I'm watching"},
    {"command": "alpha_status",
     "description": "XGBoost + DDQN training/backtest progress"},
    {"command": "alpha_search",
     "description": "Today's entry/exit strategy search — what won"},
    {"command": "alpha_trades",
     "description": "Recent trades + why each was entered and exited"},
    {"command": "help", "description": "Command list"},
]

body = urllib.parse.urlencode({"commands": json.dumps(COMMANDS)}).encode()
req = urllib.request.Request(
    f"https://api.telegram.org/bot{TOKEN}/setMyCommands",
    data=body,
    headers={"User-Agent": "okx-tele-bot/1.0"},
)
with urllib.request.urlopen(req, timeout=30) as r:
    resp = json.loads(r.read().decode("utf-8"))

print(f"setMyCommands ok={resp.get('ok')} ({len(COMMANDS)} commands)")
if not resp.get("ok"):
    print(resp)
