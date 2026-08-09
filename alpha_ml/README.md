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
`xgb_entry.py`, `ddqn.py`, `backtest.py`) for isolated debugging.

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
- **Runtime artifacts are gitignored** (`alpha_ml/models/`,
  `alpha_ml/data/*.csv`, `alpha_ml/alpha_status.json`) — they're regenerated
  by `train.py`, not source. `alpha_status_history.csv` persists on disk
  across bot restarts but isn't committed either; back it up separately if
  you care about long-run training history.
