# alpha_ml — XGBoost + DDQN alpha research pipeline

Two-phase ML on top of the same signal vocabulary the Telegram bots
(`okx_tele_bot.py`, `growhf_reactive_bot.py`) already alert on — price
momentum, volume ratio, funding, TPO/mean-reversion/regime — so both phases
learn from exactly what the live bot sees, not a separate feature set.

```
 OKX candles (data.py)
    │
    ▼
 feature table (features.py) ── same indicators as the live Telegram alerts
    │                            (market_profile.py TPO/mean-reversion/regime,
    │                             funding, vol-ratio, momentum)
    ▼
 triple-barrier labels (labels.py) ── TP/SL/time-barrier outcome, no lookahead
    │
    ├──► PHASE 1 — xgb_entry.py ──► "was this entry profitable" classifier
    │      embargoed walk-forward CV, same scheme as
    │      nlp-macro-event-bot/model_lab.py
    │
    └──► PHASE 2 — env.py + ddqn.py ──► position policy (flat/long/short)
           Double DQN searches for alpha beyond the momentum-follow default,
           trained against env.PerpTradingEnv

 backtest.py ── phase-2 policy, phase-1-gated, walked over the full OOS
                series: PnL, Sharpe, Sortino, max DD, profit factor, trade
                count, turnover+fees, win rate, per-regime breakdown
    │
    ▼
 status.py ── writes alpha_status.json (latest) + alpha_status_history.csv
              (trend log, survives restarts) ──► /alpha_status in
              okx_tele_bot.py reads it on demand
```

Running alongside that, on its own daily schedule:

```
 strategies.py ── the entry/exit zoo: 6 entry rules x 4 exit rules, each
    │              parameterized, each stating IN WORDS why it fired
    ▼
 search.py ── tries every combination on real candles, ranks them, confirms
               the winner on data the ranking never saw, logs every trade
    │
    ▼
 search_status.json  ──► /alpha_search   (what won, does it clear your bar)
 champion_trades.csv ──► /alpha_trades   (why each trade was taken and closed)
 leaderboard.csv     ──► the running record across days
```

## Why two phases, not one

Phase 1 answers "is this a good entry" as a yes/no gate — cheap, well-
calibrated, easy to sanity-check with walk-forward AUC. Phase 2 answers
"given the market right now, what position should I hold" — a sequential
decision the classifier can't express (when to exit, when to flip, when to
sit out a chop) — and is where new alpha (behavior the momentum-follow
default wouldn't take) actually gets searched for. The backtest applies
both together: phase 2 picks the position, phase 1 gates new entries.

## Setup

```bash
cd alpha_ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The root bots never need this — `status.py` (read by `/alpha_status`) is
stdlib-only by design.

## Run

```bash
# Fast, synthetic, no network — proves the whole pipeline wires together
# and each stage recovers a planted signal out-of-sample.
python train.py --selftest

# Real data: fetches OKX historical candles, trains both phases, backtests,
# writes status.
python train.py --inst BTC-USDT-SWAP --bar 5m --bars 5000 \
    --ddqn-episodes 500 --episode-len 200
```

Every module also has its own `--selftest` (`features.py`, `labels.py`,
`xgb_entry.py`, `ddqn.py`, `backtest.py`, `strategies.py`, `search.py`) for
isolated debugging.

## The daily strategy search

`train.py` learns one policy. `search.py` asks a different question: **of all
the entry and exit rules we could be using, which one actually works?** It
runs daily and keeps running until one clears the bar.

```bash
# Synthetic, fast, no network
python search.py --selftest

# Real: every strategy in the grid, over the last ~40 days of 5m candles
python search.py --inst BTC-USDT-SWAP --bar 5m --bars 12000

# Say what "good enough" means to you
python search.py --inst BTC-USDT-SWAP --bar 5m --bars 20000 \
    --target-sharpe 1.5 --target-profit-factor 1.3 --target-trades 40
```

### What it searches over

| Entry rule | Fires when |
|---|---|
| `momentum` | `chg_short` clears a dead zone — the side the live spike alert already takes |
| `breakout` | close takes out the prior N-bar high/low (Donchian) |
| `meanrev` | `mr_zscore` is stretched past ±z — fade it |
| `volspike` | volume ≥ Nx its trailing average **and** the bar confirms with a price move |
| `pocfade` | price is more than N% from the TPO point of control — fade back to value |
| `regime_router` | follow momentum when trending, fade when ranging, sit out otherwise |

| Exit rule | Closes when |
|---|---|
| `bracket` | price hits an ATR-scaled stop loss or take profit set at entry |
| `trail` | price gives back N×ATR from the best level the trade reached |
| `time` | the holding-period cap is reached |
| `flip` | the entry rule itself points the other way |

The DDQN policy had **none** of these exits — it held whatever position it
chose, with no stop, no target and no time cap. That is the single biggest
difference between what the search tests and what phase 2 was doing.

### Why the split is three-way, not two

```
|------------ train ------------|---- validation ----|--- holdout ---|
 phase-1 XGBoost fits here        candidates ranked    only the top-k
                                  here                 are run here
```

Once you are choosing between hundreds of candidates, picking the best one on
the holdout makes the holdout in-sample — the overfitting just moves up into
the selection step. So the winner is chosen on validation, and holdout only
confirms a choice that has already been made. **The champion is never the best
holdout number**, and `search.py`'s self-test asserts exactly that.

### The luck bar

Trying hundreds of strategies a day is an excellent way to find something that
never worked. Draw N candidates from pure noise and the best still posts a
positive Sharpe of about `sqrt(bars_per_year / n_bars) * sqrt(2 ln N)`. Every
run prints that threshold beside the winner, and clearing it is part of being
"satisfied" — a Sharpe target of 1.0 means nothing if noise alone scores 10.

Holdout wears out too: the leaderboard counts how many times each strategy has
been confirmed, and warns once the same one has been re-tested too often. Each
daily run pulls candles that did not exist yesterday, which is what keeps the
tail genuinely unseen.

### Reading the trades

Every trade records why it was opened and why it was closed:

```
🔴 LONG  64,653.70 → 63,983.40   -1.14%  (38 bars)
   2026-08-10 13:30  regime transitional
   ▸ in : pocfade: price -0.80% from POC, past ±0.40% — fading back toward
          value (xgb_p 0.57) [regime transitional]
   ▸ out: stop_loss: low 63,980.10 <= stop 63,983.40 (-1.04% from entry)
```

`/alpha_trades 10` in Telegram shows the last ten; the full log is in
`champion_trades.csv` with the feature values at entry.

## Checking progress

Send `/alpha_status` to the Telegram bot any time — it reads the latest
`alpha_status.json` written by the last `train.py` run, no re-training
needed:

```
🤖 Alpha ML status
BTC-USDT-SWAP 5m

XGBoost AUC: 0.612   acc: 0.548
DDQN avg reward: +0.0091   episode: 18400

OOS PnL: +8.40%   Sharpe: +1.90   Sortino: +2.40
Max DD: -8.40%   Profit factor: 1.63
Trades: 1284   Win rate: 54.7%
Turnover: 3120.0   Fees paid: 3.10%   Gated entries: 92

Last checkpoint: 19:02 WIB, 09 Aug

By regime:
  trending: return +5.00%, sharpe +2.10
  ranging: return +2.00%, sharpe +1.10
```

## Caveats — read before trusting any number here

- **Not investment advice, no edge claim.** Every self-test caveats its own
  AUC/Sharpe as a synthetic sanity check, same convention as
  `nlp-macro-event-bot/backtest.py`. A real run's numbers need the same
  skepticism: check the permutation/embargo sensitivity a real validation
  pass would run before sizing anything against this.
- **No fees/slippage beyond the flat `--fee` turnover cost** modeled in
  `backtest.py` — real perp execution has funding carry, slippage on size,
  and partial fills that this does not simulate.
- **Cold start is safe.** `SignalScorer.predict_proba()` returns a neutral
  0.5 and `/alpha_status` reports "no training run yet" until `train.py` has
  actually been run once — nothing breaks if you wire phase-1 scoring into
  the live bots before a model exists.
- **A self-test can never publish itself.** `train.py --selftest` and
  `search.py --selftest` run on synthetic data and pass `persist=False`, so
  neither can overwrite the status files `/alpha_status` and `/alpha_search`
  serve. The CI steps that report to Telegram and commit results are gated on
  the real run succeeding, for the same reason: a failed run must leave the
  last real result standing, not broadcast the smoke test's numbers.
- **Big models are gitignored, small results are not.** `alpha_ml/models/`
  and `alpha_ml/data/*.csv` are regenerated, never committed. The status
  files, trade log and leaderboard are committed on purpose — training and
  searching run on GitHub Actions, and committing the output back is how the
  local bot ever sees it.
