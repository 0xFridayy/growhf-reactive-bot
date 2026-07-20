"""
FEEDS_WS — real-time news websockets: Tree of Alpha + Phoenix News (step 2)
===========================================================================
A persistent asyncio consumer that keeps one connection per source, normalizes
every headline into (source, title, link), and drains them through the shared
Stage-1 -> Stage-2 -> alert pipeline (shared.ingest_batch).

Sources (both have a usable FREE tier — verified 2026-07-10):
  Tree of Alpha   wss://news.treeofalpha.com/ws   free delayed; `login {KEY}` to authenticate
  Phoenix News    wss://wss.phoenixnews.io        free base; `login {KEY}` or x-api-key header

Latency design (per build plan): the Stage-1 keyword tier fires immediately and
cheaply; only survivors go to Haiku. For the very fastest listing/hack trades,
consume `raw_queue` directly off the keyword tier and let the LLM score refine
after — don't put the LLM on the critical path.

Run:
  py feeds_ws.py --selftest     # parse synthetic frames, no network/keys
  py feeds_ws.py                # live (needs `websockets`; keys optional for Tree)

Deps: pip install websockets
"""

import argparse
import asyncio
import json
import time

from shared import (TREE_KEY, PHOENIX_KEY, have, ingest_batch, record_raw,
                    classify, send)

TREE_WS    = "wss://news.treeofalpha.com/ws"
PHOENIX_WS = "wss://wss.phoenixnews.io"

SCORE_INTERVAL = 2.0     # seconds between Haiku batch drains
BACKOFF_MAX    = 60      # seconds, reconnect cap


# ----------------------------------------------------------------------
# FRAME NORMALIZATION — defensive: these feeds vary their JSON shape.
# ----------------------------------------------------------------------
_TITLE_KEYS = ("title", "text", "body", "content", "headline", "en")
_LINK_KEYS  = ("url", "link", "sourceUrl", "source_url", "href")


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):  # e.g. {"title": {"en": "..."}}
            inner = _first(v, keys)
            if inner:
                return inner
    return None


def normalize(source, raw):
    """Parse one websocket frame -> (source, title, link) or None if not news.
    `raw` may be a JSON string or an already-decoded object."""
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    # Skip obvious control frames (heartbeats, auth acks, likes stream).
    if data.get("type") in ("heartbeat", "ping", "pong", "auth", "connected"):
        return None
    title = _first(data, _TITLE_KEYS)
    if not title:
        return None
    link = _first(data, _LINK_KEYS) or ""
    # Some Tree frames nest the real link under suggestions/twitter.
    if not link and isinstance(data.get("link"), dict):
        link = _first(data["link"], _LINK_KEYS) or ""
    return (source, title, link)


# ----------------------------------------------------------------------
# CONSUMERS
# ----------------------------------------------------------------------
async def _consume(name, url, api_key, raw_queue, ai_suppress=False):
    """Generic resilient consumer: connect, auth, stream frames into queue."""
    import websockets  # imported here so --selftest needs no dep
    connect_url = url + ("?ai=false" if ai_suppress else "")
    backoff = 1
    while True:
        try:
            async with websockets.connect(connect_url, ping_interval=20,
                                          max_size=2**21) as ws:
                if have(api_key):
                    await ws.send(f"login {api_key}")
                print(f"[{name}] connected"
                      + (" (authenticated)" if have(api_key) else " (free tier)"))
                backoff = 1
                async for msg in ws:
                    item = normalize(name, msg)
                    if item:
                        await raw_queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[{name}] disconnected: {e} — reconnect in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)


async def _scorer(raw_queue):
    """Drain the queue on a fixed cadence and run the shared pipeline.
    Blocking Haiku HTTP call is pushed to a thread so the event loop keeps
    ingesting websocket frames while scoring is in flight."""
    while True:
        await asyncio.sleep(SCORE_INTERVAL)
        batch = []
        while not raw_queue.empty():
            batch.append(raw_queue.get_nowait())
        if not batch:
            continue
        # Fast keyword-tier log (no LLM) so you SEE flow even before scoring.
        for source, title, _ in batch:
            cat, base = classify(title)
            if cat and base >= 4:
                print(f"[fast:{source}] {cat} <= {title[:70]}")
        try:
            alerts = await asyncio.to_thread(ingest_batch, batch)
            if alerts:
                print(f"[scored] {len(alerts)} alert(s) sent")
        except Exception as e:
            print(f"[scorer error] {e}")


async def run():
    raw_queue = asyncio.Queue()
    tasks = [asyncio.create_task(_scorer(raw_queue))]
    tasks.append(asyncio.create_task(
        _consume("tree", TREE_WS, TREE_KEY, raw_queue)))
    tasks.append(asyncio.create_task(
        _consume("phoenix", PHOENIX_WS, PHOENIX_KEY, raw_queue, ai_suppress=True)))
    send("📡 Websocket feeds online — Tree of Alpha + Phoenix News.")
    print("feeds_ws running (Tree + Phoenix). Ctrl+C to stop.")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass


# ----------------------------------------------------------------------
# SELF-TEST — feed synthetic frames through the parser + Stage-1 filter.
# No network, no keys, no LLM.
# ----------------------------------------------------------------------
def _selftest():
    print("feeds_ws parser self-test (synthetic frames)\n")
    frames = [
        ("tree", '{"title":"Binance will list ASTER perpetual futures","source":"Binance","url":"https://x.com/a/1"}'),
        ("tree", '{"type":"heartbeat"}'),
        ("phoenix", '{"title":"BREAKING: SEC approves spot XRP ETF","sourceUrl":"https://sec.gov/x"}'),
        ("phoenix", '{"body":"Fed\'s Warsh signals openness to a rate cut in September","url":"https://ff.com/y"}'),
        ("tree", '{"title":{"en":"Top 10 altcoins to buy this week"},"url":"https://spam/z"}'),
        ("tree", 'not-json-garbage'),
        ("phoenix", '{"type":"pong"}'),
    ]
    parsed, candidates = 0, 0
    for src, raw in frames:
        item = normalize(src, raw)
        if item is None:
            print(f"  [{src}] dropped (control/garbage)")
            continue
        parsed += 1
        _, title, link = item
        cat, base = classify(title)
        verdict = f"-> {cat} (base {base})" if cat and base >= 2 else "-> filtered as noise/low"
        if cat and base >= 2:
            candidates += 1
        print(f"  [{src}] {title[:60]:<60} {verdict}")
    print(f"\n  parsed {parsed}/{len(frames)} frames, {candidates} would escalate "
          f"to Haiku. Expect 3 candidates (listing, ETF, Fed) and the 'Top 10' "
          f"spam filtered out.")


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
