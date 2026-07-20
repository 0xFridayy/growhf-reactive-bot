# Macro + Sentiment Intelligence Stack

Real-time crypto sentiment + macro/event layer for a solo Binance USDT-M perps
ML trader. Independent modules — run any subset in separate terminals. Every
scored headline (RSS, websocket, or Telegram) funnels through one pipeline and
lands in one SQLite store the regime engine reads.

```
 RSS  ─┐
 Tree ─┤                         ┌─ Telegram alert (impact ≥ 3)
 Phx  ─┼─► Stage 1 keyword filter ─► Stage 2 Claude Haiku ─┤
 TG   ─┘   (free, kills ~95%)      (batched, cost-guarded)  └─► signals.db
                                                                   │
   Fear&Greed / FRED / Finnhub / CoinMarketCal ──► regime_features │
                                                                   ▼
                              sentiment_engine  ──►  sentiment_state()  ──► your regime engine
                              (exp. time-decay + rolling windows)
```

## Modules

| File | What it does | Runs without keys? |
|---|---|---|
| `macro_event_bot.py` | Scheduled macro calendar, `is_blackout()`, T-60 alerts, `/events /today /blackout /btc` | needs Telegram token |
| `news_nlp_bot.py` | 8 RSS feeds → dedupe → classify → Haiku → alert, `market_risk_state()` | needs Telegram + Anthropic |
| `shared.py` | Canonical config, Telegram, taxonomy/classifier, Haiku scorer, `ingest_batch()`, `signals.db` store | ✅ self-test |
| `data_sources.py` | Fear & Greed (free), FRED, Finnhub calendar, CoinMarketCal → `snapshot_regime_features()` | ✅ Fear&Greed live; rest need free keys |
| `feeds_ws.py` | Tree of Alpha + Phoenix News websockets → `ingest_batch()` | ✅ `--selftest`; live needs network |
| `telegram_listener.py` | Telethon user-account firehose over N channels | ✅ `--selftest`; live needs api_id/hash |
| `sentiment_engine.py` | Discrete events → continuous ML feature via exp. time-decay + rolling windows | ✅ fully local |
| `run_all.py` | One-process supervisor: launches every service, auto-restarts crashes, runs periodic snapshot jobs | ✅ `--dry-run` + live |
| `features.py` | Leakage-safe training table: perp + point-in-time sentiment + regime + forward label | ✅ `--selftest` |
| `model_lab.py` | Embargoed walk-forward CV, XGBoost/LightGBM, honest OOS metrics | ✅ `--selftest` |

## Setup

```
cd "C:\Users\jason\Desktop\VsCode\NLP Macro Event Bot"
py -m pip install -r requirements.txt
```

Env vars (set what each feature needs):

```
TG_BOT_TOKEN      TG_CHAT_ID          ANTHROPIC_API_KEY     # core (same bot as okx_spike_screener.py)
FINNHUB_API_KEY   FRED_API_KEY        COINMARKETCAL_API_KEY # free-tier calendars
TREE_API_KEY      PHOENIX_API_KEY                            # optional WS auth (Tree free w/o)
TELEGRAM_API_ID   TELEGRAM_API_HASH                          # Telethon (burner account)
```

Verify everything compiles + logic works, no keys needed:

```
py shared.py                    # classifier + store smoke test
py data_sources.py              # Fear & Greed pulls live
py sentiment_engine.py --selftest
py feeds_ws.py --selftest
py telegram_listener.py --selftest
py features.py --selftest       # leakage-safe table on synthetic data
py model_lab.py --selftest      # walk-forward XGBoost recovers a planted signal
py run_all.py --dry-run         # preflight: what will/won't launch
```

## Run the whole stack — one terminal

```
py run_all.py                   # launches every service whose creds are set
py run_all.py --only feeds_ws,sentiment,regime   # a subset
```

`run_all` skips any service missing its credentials (a partial config still runs
everything it can), auto-restarts crashes with capped backoff, and runs the
`sentiment_engine` (60s) + Fear & Greed (1h) snapshot jobs in-process.

## The integration that matters — sentiment as a regime feature

`sentiment_engine` turns discrete scored headlines into a continuous feature via
`weight = impact · exp(−Δt/τ)`, tuned per horizon (`scalp` τ=5m, `swing` τ=1h,
`macro` τ=6h). It reads RSS rows from `news_nlp.db` and websocket/Telegram rows
from `signals.db`, so it sees **every** source with no extra wiring.

```python
from sentiment_engine import sentiment_state

feat = sentiment_state("swing")
# {'decayed_impact': .., 'decayed_sentiment': .., 'net_tone': -5..5,
#  'max_impact': 1..5, 'n_events': .., 'top_category': 'FED'}

if feat["net_tone"] < -2 and feat["decayed_impact"] > 8:
    ...  # heavy bearish newsflow → cut size / block longs
```

Plus the existing gates:

```python
from macro_event_bot import is_blackout          # scheduled-event blackout
from news_nlp_bot     import market_risk_state    # raw signed-impact sum
from data_sources     import fear_greed_feature   # slow daily regime tag
```

Call `sentiment_engine.recompute_and_store()` on your bar cadence (1m/5m close)
to persist a `sentiment_features` snapshot row per horizon for offline ML/backtest.

## Build order (from the fact-checked plan)

1. **Done — free macro layer.** Fear & Greed + Fed RSS in the RSS bot; Finnhub /
   CoinMarketCal / FRED clients in `data_sources.py` (drop in free keys).
2. **Done — real-time news layer.** `feeds_ws.py` (Tree free + Phoenix) and
   `telegram_listener.py` (burner) feed the shared pipeline with cross-source dedup.
3. **Done — the signal.** `sentiment_engine.py` decay + aggregation +
   `sentiment_state()`.
4. **Done (scaffold) — modeling.** `features.py` builds the leakage-safe table;
   `model_lab.py` trains XGBoost (LightGBM if installed) with embargoed
   walk-forward CV. Point it at real data:
   ```
   py features.py --symbol BTCUSDT --interval 5m --out training_table.csv
   py model_lab.py --csv training_table.csv
   ```
   Then judge honestly against your rigor checklist before trusting any number.
5. **Only if it pays for itself.** Tree of Alpha Sprout (~$500/mo) to cut latency —
   after step 4 shows a real, fees-included edge exceeding the cost.

### Why the modeling harness is trustworthy
Every feature is a backward as-of join or explicit lag (no look-ahead by
construction); the label is a forward return with the tail dropped; CV is
expanding-window walk-forward with an embargo equal to the label horizon, so a
train row's forward label can never overlap a test block. The `--selftest`
plants a decaying impulse-response signal and confirms the model recovers it
**out-of-sample** (AUC ≈ 0.60) — proof the pipeline finds real structure without
memorizing. That is a synthetic sanity check, **not** an edge claim: metrics
ignore fees/slippage, and your system is still validated only on synthetic data.

### Validation battery — `backtest.py`
```
py backtest.py            # full battery (~1 min) -> backtest_results.json
```
Six correctness checks on 4,000-bar synthetic data (all passed on last run):

| Check | Result | Meaning |
|---|---|---|
| Permutation (shuffle labels) | 0.589 → 0.501 | no leakage — shuffled collapses to chance |
| No false edge (0 signal) | AUC 0.504 | invents no edge from noise |
| SNR recovery (×0/0.5/1/2) | 0.50 → 0.74 | detects signal in proportion to strength |
| Embargo 0→12 bars | 0.584 → 0.600 | edge isn't label-overlap artifact |
| Per-fold stability | 0.573–0.615 | consistent across time, not one fold |
| Feature ablation | tech 0.642 · all 0.589 · sent 0.544 | **honest finding:** here sentiment was redundant to price — must carry orthogonal info on real data to help |

The point of the battery isn't the AUC — it's that the harness **can't be fooled**:
it stays at 0.50 when there's nothing to find and never sees the future. Judge
real-data results against your rigor checklist before trusting any Sharpe.

_Full results dashboard (mobile): the "Stack built & validated" artifact._

## Caveats (don't skip)

- **Telethon ToS is a gray area.** Burner account, read-only (this listener never
  posts or mass-joins). Telethon's repo went maintenance-mode 2026-02-21 — pinned
  to 1.44.0 in requirements.
- **Sentiment edge is unproven.** Validated on synthetic data only. Treat
  `sentiment_state()` as a hypothesis to backtest on *real* data before it sizes a
  single trade.
- **Free Tree feed is delayed** — fine for prototyping/swing, not latency races.
- **Fear & Greed is daily + lagging** — a slow regime tag, never a trigger.

_Full rationale, costs, and source verification: see the build-plan doc and the
mobile reference artifact._
