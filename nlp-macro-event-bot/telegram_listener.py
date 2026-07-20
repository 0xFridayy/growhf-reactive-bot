"""
TELEGRAM LISTENER — free real-time firehose via Telethon (build-plan step 2)
============================================================================
Listens to public channels as a USER account (MTProto) and fires on every new
message in real time, funneling text through the same Stage-1 -> Stage-2 ->
alert pipeline (shared.ingest_batch). This is the highest-leverage FREE source:
Tree of Alpha's channel, Wu Blockchain, BWE/BlockBeats, Watcher.Guru and
exchange announcement channels all post to Telegram instantly.

⚠️  ToS: automating a user account is a gray area. Use a SECONDARY/BURNER
account, READ-ONLY (this script never sends to channels or joins en masse),
and keep behavior human-like. Telethon's repo went read-only (maintenance
mode) 2026-02-21 — pin the version and don't expect new features.

Setup:
  1. Get api_id/api_hash from https://my.telegram.org (burner account).
  2. set TELEGRAM_API_ID / TELEGRAM_API_HASH env vars.
  3. pip install telethon
  4. py telegram_listener.py     # first run prompts for phone + login code,
                                 # then caches a .session file for reuse.

Run:  py telegram_listener.py            # live
      py telegram_listener.py --selftest # message-handler logic, no network
Deps: pip install telethon
"""

import argparse
import asyncio
import os

from shared import classify, ingest_batch, send

# Highest-signal public channels. Usernames (without @) or t.me links.
# Tune this list — start with 5-10, dedup happens downstream by content hash.
CHANNELS = [
    "treeofalpha_official",   # Tree of Alpha channel
    "WuBlockchain",           # Wu Blockchain
    "BWEnews",                # BWE / BlockBeats english
    "WatcherGuru",            # Watcher.Guru
    "binance_announcements",  # exchange listings
]

# Batch window: collect messages briefly, then score together (cost guard).
BATCH_SECONDS = 3.0

API_ID   = os.environ.get("TELEGRAM_API_ID", "")
API_HASH = os.environ.get("TELEGRAM_API_HASH", "")
SESSION  = os.environ.get("TELEGRAM_SESSION", "tg_listener")


def _channel_label(chat):
    name = getattr(chat, "username", None) or getattr(chat, "title", None) or "?"
    return f"telegram:{name}"


async def run():
    if not (API_ID and API_HASH):
        print("[telegram] TELEGRAM_API_ID / TELEGRAM_API_HASH not set — "
              "get them from https://my.telegram.org. Aborting.")
        return
    from telethon import TelegramClient, events   # imported here; selftest is dep-free

    client = TelegramClient(SESSION, int(API_ID), API_HASH)
    queue = asyncio.Queue()

    @client.on(events.NewMessage(chats=CHANNELS))
    async def handler(event):
        text = (event.message.message or "").strip()
        if not text:
            return
        source = _channel_label(await event.get_chat())
        # First line is the headline; keep it short for the classifier.
        headline = text.splitlines()[0][:280]
        link = ""
        await queue.put((source, headline, link))

    async def scorer():
        while True:
            await asyncio.sleep(BATCH_SECONDS)
            batch = []
            while not queue.empty():
                batch.append(queue.get_nowait())
            if batch:
                try:
                    alerts = await asyncio.to_thread(ingest_batch, batch)
                    if alerts:
                        print(f"[telegram] {len(alerts)} alert(s) sent")
                except Exception as e:
                    print(f"[telegram scorer error] {e}")

    await client.start()
    send("📨 Telegram listener online (read-only firehose).")
    print(f"[telegram] listening on {len(CHANNELS)} channels. Ctrl+C to stop.")
    asyncio.create_task(scorer())
    await client.run_until_disconnected()


# ----------------------------------------------------------------------
# SELF-TEST — exercise the message->classify path with synthetic messages.
# ----------------------------------------------------------------------
def _selftest():
    print("telegram_listener self-test (synthetic messages)\n")
    msgs = [
        ("telegram:binance_announcements", "Binance Will List Aster (ASTER) with USDT Perpetual"),
        ("telegram:WuBlockchain", "A whale deposited 5,000 BTC to Binance"),
        ("telegram:WatcherGuru", "JUST IN: US CPI comes in hotter than expected at 3.4%"),
        ("telegram:someone", "gm frens wagmi lfg 🚀"),
    ]
    candidates = 0
    for source, text in msgs:
        headline = text.splitlines()[0][:280]
        cat, base = classify(headline)
        keep = bool(cat and base >= 2)
        candidates += keep
        print(f"  [{source.split(':')[1]:<22}] {headline[:50]:<50} "
              f"-> {cat+' (base '+str(base)+')' if keep else 'filtered'}")
    print(f"\n  {candidates} of {len(msgs)} messages would escalate to Haiku. "
          f"Expect the listing, whale, and CPI to pass; the 'gm' chatter filtered.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            print("\nstopped.")
