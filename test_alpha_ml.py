"""
test_alpha_ml.py — unit tests for the alpha_ml phase-1/phase-2 pipeline.

Requires alpha_ml/requirements.txt (numpy, pandas, scikit-learn, xgboost,
torch) — NOT part of root requirements.txt, since the live Telegram bots
never need these. Run with alpha_ml's deps installed:

    pip install -r alpha_ml/requirements.txt
    pytest test_alpha_ml.py -v

All tests are fast (small synthetic data, few DDQN episodes) and network-free.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).with_name("alpha_ml")))

from backtest import compute_metrics, run_policy, momentum_target  # noqa: E402
from env import PerpTradingEnv  # noqa: E402
from ddqn import DDQNAgent  # noqa: E402
from features import assemble, synthetic_market  # noqa: E402
from labels import atr_fractions, triple_barrier, triple_barrier_signed  # noqa: E402
from status import format_telegram, read_status, write_status  # noqa: E402
from xgb_entry import SignalScorer, build_dataset  # noqa: E402


# --------------------------------------------------------------------------- #
# features.py
# --------------------------------------------------------------------------- #
class TestFeatures:
    def test_assemble_shape_and_no_nans(self):
        market = synthetic_market(n=400, planted=True)
        df, cols = assemble(market, window=32)
        assert len(df) > 200
        assert len(cols) > 0
        assert df[cols].isna().sum().sum() == 0

    def test_no_future_leakage(self):
        """Perturbing the LAST bar must not change an earlier row's features —
        the profile/TPO/regime columns are built from a rolling window that
        ends at each row, never looks ahead."""
        market = synthetic_market(n=300, planted=True)
        df1, cols = assemble(market, window=24)
        market2 = market.copy()
        market2.loc[len(market2) - 1, ["o", "h", "l", "c"]] *= 2.0
        df2, _ = assemble(market2, window=24)
        row_a = df1.iloc[50][cols].to_numpy(dtype=float)
        row_b = df2.iloc[50][cols].to_numpy(dtype=float)
        assert np.allclose(row_a, row_b)


# --------------------------------------------------------------------------- #
# labels.py
# --------------------------------------------------------------------------- #
class TestLabels:
    def _ramp(self, n=20, direction=1):
        vals = np.linspace(100, 100 + direction * 20, n)
        return pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
            "o": vals, "h": vals + 0.5, "l": vals - 0.5, "c": vals, "v": np.ones(n),
        })

    def test_monotonic_uptrend_long_always_wins(self):
        ramp = self._ramp(direction=1)
        res = triple_barrier(ramp, tp_frac=0.01, sl_frac=0.01, max_hold=5, direction=1)
        labeled = res.dropna(subset=["y"])
        assert len(labeled) > 0
        assert (labeled["y"] == 1.0).all()

    def test_monotonic_uptrend_short_always_loses(self):
        ramp = self._ramp(direction=1)
        res = triple_barrier(ramp, tp_frac=0.01, sl_frac=0.01, max_hold=5, direction=-1)
        labeled = res.dropna(subset=["y"])
        assert len(labeled) > 0
        assert (labeled["y"] == 0.0).all()

    def test_tail_rows_unlabeled(self):
        ramp = self._ramp(n=20)
        res = triple_barrier(ramp, tp_frac=0.5, sl_frac=0.5, max_hold=5, direction=1)
        # tp/sl set unreachable (50%) so every row falls through to the time
        # barrier; only rows within max_hold of the end lack full forward data.
        assert res.iloc[:15]["y"].notna().all()

    def test_signed_direction_skips_flat_bars(self):
        ramp = self._ramp(n=20)
        direction = np.array([1, 0, -1] * 6 + [0, 0])
        res = triple_barrier_signed(ramp, direction, tp_frac=0.01, sl_frac=0.01, max_hold=3)
        flat_mask = direction == 0
        assert res.loc[flat_mask, "y"].isna().all()

    def test_atr_fractions_causal_and_clipped(self):
        market = synthetic_market(n=200, planted=False)
        frac = atr_fractions(market, lookback=14, mult=1.5, floor=0.002, ceil=0.08)
        assert np.isnan(frac[:14]).all()
        valid = frac[~np.isnan(frac)]
        assert (valid >= 0.002).all() and (valid <= 0.08).all()


# --------------------------------------------------------------------------- #
# env.py
# --------------------------------------------------------------------------- #
class TestPerpTradingEnv:
    def _env(self):
        market = synthetic_market(n=400, planted=True)
        df, cols = assemble(market, window=32)
        return PerpTradingEnv(df, cols, episode_len=50, fee=0.001, seed=0), cols

    def test_reset_returns_correct_shape(self):
        env, cols = self._env()
        obs = env.reset(start_idx=0)
        assert obs.shape == (len(cols) + 2,)
        assert env.position == 0.0

    def test_step_flat_action_zero_reward(self):
        env, _ = self._env()
        env.reset(start_idx=0)
        _, r, done, info = env.step(0)  # flat
        assert r == 0.0
        assert info["position"] == 0.0
        assert done is False

    def test_step_charges_fee_on_turnover(self):
        env, _ = self._env()
        env.reset(start_idx=0)
        c0, c1 = env._closes[0], env._closes[1]
        expected_bar_ret = (c1 - c0) / c0
        _, r, _, info = env.step(1)  # flat -> long: turnover 1.0
        assert r == pytest.approx(expected_bar_ret - env.fee * 1.0)
        assert info["turnover"] == 1.0

    def test_episode_terminates_after_episode_len(self):
        env, _ = self._env()
        env.reset(start_idx=0)
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
            assert steps <= env.episode_len + 1
        assert steps == env.episode_len


# --------------------------------------------------------------------------- #
# ddqn.py — tiny, deterministic, proves the wiring learns something
# --------------------------------------------------------------------------- #
class _SignEnv:
    ACTION_DIM = 3
    ACTIONS = {0: 0.0, 1: 1.0, 2: -1.0}

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.state_dim = 3

    def reset(self, start_idx=None):
        self.signal = float(self.rng.choice([-1.0, 1.0]))
        self.position = 0.0
        return np.array([self.signal, self.position, 0.0], dtype=np.float32)

    def step(self, action):
        target = self.ACTIONS[int(action)]
        reward = target * self.signal * 0.01 - 0.0005 * abs(target - self.position)
        self.position = target
        return np.array([self.signal, self.position, 1.0], dtype=np.float32), float(reward), True, {}


class TestDDQNAgent:
    def test_learns_trivial_sign_following_task(self):
        env = _SignEnv(seed=0)
        agent = DDQNAgent(env.state_dim, env.ACTION_DIM, hidden=16, lr=5e-3,
                          buffer_size=1000, batch_size=32, eps_decay_episodes=100,
                          target_update_every=20, seed=0)
        pre = np.mean(agent.evaluate(env, episodes=100))
        agent.train(env, episodes=200, warmup_steps=32, log_every=0)
        post = np.mean(agent.evaluate(env, episodes=100))
        assert post > pre
        assert post > 0.005  # optimal is ~0.01

    def test_save_load_roundtrip(self, tmp_path):
        env = _SignEnv(seed=1)
        agent = DDQNAgent(env.state_dim, env.ACTION_DIM, hidden=16, seed=1)
        ckpt = tmp_path / "agent.pt"
        agent.save(ckpt)
        loaded = DDQNAgent.load(ckpt, hidden=16)
        s = env.reset()
        assert agent.act(s, greedy=True) == loaded.act(s, greedy=True)


# --------------------------------------------------------------------------- #
# xgb_entry.py
# --------------------------------------------------------------------------- #
class TestSignalScorer:
    def test_cold_start_returns_neutral(self):
        scorer = SignalScorer()
        assert scorer.predict_proba({"any_feature": 1.0}) == 0.5
        assert not scorer.is_trained

    def test_save_load_roundtrip(self, tmp_path):
        market = synthetic_market(n=600, planted=True)
        df, cols = build_dataset(market, window=32, max_hold=8, direction_mode="long_only")
        from xgb_entry import train_final
        scorer = train_final(df, cols)
        path = tmp_path / "model.json"
        scorer.save(path)
        loaded = SignalScorer.load(path)
        assert loaded.is_trained
        row = df.iloc[-1][cols].to_dict()
        assert loaded.predict_proba(row) == pytest.approx(scorer.predict_proba(row))

    def test_load_missing_path_is_cold_start(self, tmp_path):
        scorer = SignalScorer.load(tmp_path / "does_not_exist.json")
        assert not scorer.is_trained
        assert scorer.predict_proba({}) == 0.5


# --------------------------------------------------------------------------- #
# backtest.py
# --------------------------------------------------------------------------- #
class TestBacktest:
    def test_momentum_baseline_metrics_well_formed(self):
        market = synthetic_market(n=600, planted=True)
        df, cols = assemble(market, window=32)
        trace = run_policy(df, cols, target_fn=momentum_target, fee=0.0005)
        m = compute_metrics(trace)
        assert m["n_bars"] == len(trace)
        assert -1.0 <= m["max_drawdown"] <= 0.0
        assert m["turnover"] >= 0.0
        assert set(m["by_regime"]).issubset({"trending", "ranging", "transitional"})

    def test_scorer_gate_reduces_or_holds_trade_count(self):
        market = synthetic_market(n=600, planted=True)
        df, cols = build_dataset(market, window=32, max_hold=8, direction_mode="long_only")
        from xgb_entry import train_final
        scorer = train_final(df, cols)
        full_df, full_cols = assemble(market, window=32)
        ungated = compute_metrics(run_policy(full_df, cols, target_fn=momentum_target, fee=0.0005))
        gated = compute_metrics(run_policy(full_df, cols, target_fn=momentum_target,
                                           scorer=scorer, prob_threshold=0.9, fee=0.0005))
        assert gated["trade_count"] <= ungated["trade_count"]
        assert gated["gated_entries"] >= 0


# --------------------------------------------------------------------------- #
# status.py
# --------------------------------------------------------------------------- #
class TestStatus:
    def test_write_read_roundtrip(self, tmp_path, monkeypatch):
        import status as status_mod
        monkeypatch.setattr(status_mod, "STATUS_JSON", tmp_path / "status.json")
        monkeypatch.setattr(status_mod, "STATUS_HISTORY_CSV", tmp_path / "history.csv")
        written = write_status({"inst": "TEST-SWAP", "bar": "5m", "xgb_auc": 0.6})
        assert "ts_utc" in written  # filled in by write_status, visible to the caller
        loaded = read_status()
        assert loaded["inst"] == "TEST-SWAP"
        assert loaded["ts_utc"] == written["ts_utc"]

    def test_format_telegram_no_status_is_cold_start_safe(self, tmp_path, monkeypatch):
        import status as status_mod
        monkeypatch.setattr(status_mod, "STATUS_JSON", tmp_path / "missing.json")
        msg = format_telegram()
        assert "No training run yet" in msg

    def test_format_telegram_includes_key_metrics(self):
        msg = format_telegram({
            "inst": "BTC-USDT-SWAP", "bar": "5m", "search_frac": 0.7,
            "xgb_auc": 0.61, "xgb_acc": 0.55,
            "ddqn_episodes": 100, "ddqn_avg_reward": 0.01, "overfit_flag": False,
            "bt_search_total_return": 0.08, "bt_search_sharpe": 2.0, "bt_search_sortino": 2.5,
            "bt_search_max_drawdown": -0.04, "bt_search_profit_factor": 1.6,
            "bt_search_trade_count": 80, "bt_search_win_rate": 0.55,
            "bt_search_turnover": 150.0, "bt_search_fees_paid": 0.02, "bt_search_gated_entries": 5,
            "bt_holdout_total_return": 0.05, "bt_holdout_sharpe": 1.5, "bt_holdout_sortino": 2.0,
            "bt_holdout_max_drawdown": -0.03, "bt_holdout_profit_factor": 1.4,
            "bt_holdout_trade_count": 50, "bt_holdout_win_rate": 0.5,
            "bt_holdout_turnover": 100.0, "bt_holdout_fees_paid": 0.01, "bt_holdout_gated_entries": 3,
            "ts_utc": "2026-01-01T00:00:00+00:00",
        })
        assert "BTC-USDT-SWAP" in msg
        assert "XGBoost AUC: 0.610" in msg
        assert "Trades: 50" in msg

    def test_format_telegram_flags_overfit(self):
        msg = format_telegram({
            "inst": "BTC-USDT-SWAP", "bar": "5m", "search_frac": 0.7,
            "xgb_auc": 0.61, "xgb_acc": 0.55, "ddqn_episodes": 100, "ddqn_avg_reward": 0.01,
            "overfit_flag": True,
            "bt_search_sharpe": 1.68, "bt_holdout_sharpe": -0.65,
            "ts_utc": "2026-01-01T00:00:00+00:00",
        })
        assert "Overfit warning" in msg

    def test_selftest_status_is_labeled_as_synthetic(self):
        """A SELFTEST status must never read like a real BTC result — that is
        exactly how synthetic numbers got broadcast to Telegram once."""
        msg = format_telegram({
            "inst": "SELFTEST", "bar": "5m", "search_frac": 0.7,
            "xgb_auc": 0.55, "bt_holdout_sharpe": -18.4,
            "ts_utc": "2026-01-01T00:00:00+00:00",
        })
        assert "synthetic self-test" in msg


# --------------------------------------------------------------------------- #
# train.py — the self-test must not be able to publish itself
# --------------------------------------------------------------------------- #
class TestTrainPersistence:
    def test_selftest_does_not_write_status(self, tmp_path, monkeypatch):
        import status as status_mod
        import train as train_mod
        target = tmp_path / "status.json"
        monkeypatch.setattr(status_mod, "STATUS_JSON", target)
        monkeypatch.setattr(status_mod, "STATUS_HISTORY_CSV", tmp_path / "history.csv")
        market = synthetic_market(n=900, planted=True)
        status, _ = train_mod.run("SELFTEST", "5m", market, window=32, max_hold=8,
                                  xgb_folds=2, ddqn_episodes=3, episode_len=40,
                                  seed=0, persist=False)
        assert status["inst"] == "SELFTEST"
        assert not target.exists(), "a synthetic self-test wrote the published status file"

    def test_real_run_does_write_status(self, tmp_path, monkeypatch):
        import status as status_mod
        import train as train_mod
        target = tmp_path / "status.json"
        monkeypatch.setattr(status_mod, "STATUS_JSON", target)
        monkeypatch.setattr(status_mod, "STATUS_HISTORY_CSV", tmp_path / "history.csv")
        market = synthetic_market(n=900, planted=True)
        train_mod.run("TEST-SWAP", "5m", market, window=32, max_hold=8, xgb_folds=2,
                      ddqn_episodes=3, episode_len=40, seed=0, persist=True)
        assert target.exists()

    def test_overfit_flag_needs_a_winning_search_period(self):
        """Two losing Sharpes are not overfitting, they are a broken policy —
        the old `gap > 1.0` rule called search -8.4 / holdout -18.5 an overfit."""
        import train as train_mod
        market = synthetic_market(n=900, planted=True)
        status, metrics = train_mod.run("TEST-SWAP", "5m", market, window=32, max_hold=8,
                                        xgb_folds=2, ddqn_episodes=3, episode_len=40,
                                        seed=0, persist=False)
        if metrics["search"]["sharpe"] <= 0:
            assert not status["overfit_flag"]


# --------------------------------------------------------------------------- #
# strategies.py
# --------------------------------------------------------------------------- #
class TestStrategies:
    @staticmethod
    def _table(n=1200, seed=7, planted=True):
        return assemble(synthetic_market(n=n, seed=seed, planted=planted), window=32)

    def test_every_trade_carries_its_reasoning(self):
        from strategies import (AtrBracketExit, MomentumEntry, Strategy, TimeExit,
                                run_strategy)
        df, cols = self._table()
        strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                         exits=[AtrBracketExit(sl_mult=1.0, tp_mult=2.0), TimeExit(max_hold=16)])
        _, trades = run_strategy(df, strat, feature_cols=cols, fee=0.0005)
        assert trades
        for t in trades:
            assert t["entry_reason"] and t["exit_reason"]
            assert t["exit_reason"].split(":")[0] in {
                "stop_loss", "take_profit", "time_stop", "trail_stop",
                "signal_flip", "end_of_data"}

    def test_stops_actually_bind(self):
        from strategies import (AtrBracketExit, MomentumEntry, Strategy, TimeExit,
                                run_strategy)
        df, cols = self._table()
        strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                         exits=[AtrBracketExit(sl_mult=1.0, tp_mult=2.0), TimeExit(max_hold=10)])
        _, trades = run_strategy(df, strat, feature_cols=cols, fee=0.0005)
        assert all(t["bars_held"] <= 10 for t in trades), "the time cap did not bind"

    def test_no_profit_on_a_random_walk(self):
        """The strongest lookahead check there is: with planted=False there is
        no edge to find, so a strategy must bleed roughly its fee bill."""
        from strategies import (AtrBracketExit, MomentumEntry, Strategy, TimeExit,
                                run_strategy, strategy_metrics)
        for seed in (1, 2, 3):
            df, cols = self._table(n=1500, seed=seed, planted=False)
            strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                             exits=[AtrBracketExit(sl_mult=1.0, tp_mult=2.0), TimeExit(max_hold=24)])
            trace, trades = run_strategy(df, strat, feature_cols=cols, fee=0.0005)
            m = strategy_metrics(trace, trades)
            assert m["total_return"] < 0, f"seed {seed} profited on a random walk"
            assert m["total_return"] > -3 * m["fees_paid"]

    def test_trade_and_bar_accounting_agree(self):
        from strategies import (AtrBracketExit, MomentumEntry, Strategy, TimeExit,
                                run_strategy)
        df, cols = self._table()
        strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                         exits=[AtrBracketExit(), TimeExit(max_hold=24)])
        trace, trades = run_strategy(df, strat, feature_cols=cols, fee=0.0005)
        assert abs(sum(t["pnl"] for t in trades) - float(trace["reward"].sum())) < 0.02

    def test_ml_gate_only_removes_entries(self):
        from strategies import (AtrBracketExit, MomentumEntry, Strategy, TimeExit,
                                run_strategy)
        from xgb_entry import train_final
        market = synthetic_market(n=1200, planted=True)
        df, cols = assemble(market, window=32)
        xdf, xcols = build_dataset(market, window=32, max_hold=8, direction_mode="long_only")
        scorer = train_final(xdf, xcols)
        exits = [AtrBracketExit(), TimeExit(max_hold=16)]
        plain = Strategy(entry=MomentumEntry(deadzone=0.15), exits=exits)
        gated = Strategy(entry=MomentumEntry(deadzone=0.15), exits=exits, prob_threshold=0.9)
        _, plain_trades = run_strategy(df, plain, feature_cols=cols, fee=0.0005)
        gtrace, gated_trades = run_strategy(df, gated, feature_cols=xcols, fee=0.0005,
                                            scorer=scorer)
        assert len(gated_trades) <= len(plain_trades)
        assert int(gtrace["gated"].sum()) >= 0

    def test_batch_probs_match_per_row_probs(self):
        """The search's speed-up must not change any decision."""
        from strategies import (AtrBracketExit, MomentumEntry, Strategy, TimeExit,
                                run_strategy)
        from xgb_entry import train_final
        market = synthetic_market(n=1200, planted=True)
        df, cols = assemble(market, window=32)
        xdf, xcols = build_dataset(market, window=32, max_hold=8, direction_mode="long_only")
        scorer = train_final(xdf, xcols)
        strat = Strategy(entry=MomentumEntry(deadzone=0.15),
                         exits=[AtrBracketExit(), TimeExit(max_hold=16)], prob_threshold=0.6)
        _, slow = run_strategy(df, strat, feature_cols=xcols, fee=0.0005, scorer=scorer)
        _, fast = run_strategy(df, strat, feature_cols=xcols, fee=0.0005,
                               probs=scorer.predict_proba_batch(df))
        assert [t["entry_ts"] for t in slow] == [t["entry_ts"] for t in fast]
        assert np.allclose([t["pnl"] for t in slow], [t["pnl"] for t in fast])

    def test_grid_round_trips_through_dicts(self):
        from strategies import default_grid, strategy_from_dict
        grid = default_grid()
        assert len(grid) > 100
        for s in (grid[0], grid[len(grid) // 2], grid[-1]):
            assert strategy_from_dict(s.to_dict()).name == s.name


# --------------------------------------------------------------------------- #
# search.py
# --------------------------------------------------------------------------- #
class TestSearch:
    def test_champion_is_picked_on_validation_not_holdout(self):
        import search as search_mod
        market = synthetic_market(n=2400, planted=True)
        status, finalists = search_mod.run_search("TEST-SWAP", "5m", market, window=32,
                                                  max_configs=12, min_trades=3, top_k=3,
                                                  seed=0, persist=False)
        best_val = max(status["top"], key=lambda r: r["val"]["sharpe"])
        assert status["champion"] == best_val["name"]
        assert all(t["entry_reason"] and t["exit_reason"]
                   for t in finalists[0]["holdout_trades"])

    def test_luck_bar_rises_with_more_tries_and_falls_with_more_bars(self):
        from search import luck_bar
        bpy = 288 * 365
        assert luck_bar(1000, 400, bpy) > luck_bar(1000, 10, bpy) > 0
        assert luck_bar(20000, 400, bpy) < luck_bar(1000, 400, bpy)
        assert luck_bar(1000, 1, bpy) == 0.0

    def test_targets_all_must_pass(self):
        from search import check_targets
        good = {"sharpe": 2.0, "profit_factor": 1.5, "trade_count": 50,
                "max_drawdown": -0.05}
        passed, checks = check_targets(good, search_targets(), holdout_luck_bar=1.0)
        assert passed and all(c["pass"] for c in checks.values())
        bad = dict(good, max_drawdown=-0.5)
        passed, checks = check_targets(bad, search_targets(), holdout_luck_bar=1.0)
        assert not passed and not checks["max_drawdown"]["pass"]
        # A great-looking Sharpe that noise could have produced is not a pass.
        passed, checks = check_targets(good, search_targets(), holdout_luck_bar=9.0)
        assert not passed and not checks["above_noise"]["pass"]


def search_targets():
    from search import DEFAULT_TARGETS
    return dict(DEFAULT_TARGETS)


# --------------------------------------------------------------------------- #
# status.py — the daily search's Telegram surfaces
# --------------------------------------------------------------------------- #
class TestSearchStatus:
    SAMPLE = {
        "inst": "BTC-USDT-SWAP", "bar": "5m", "ts_utc": "2026-01-01T00:00:00+00:00",
        "run_date": "2026-01-01", "n_configs_tried": 392,
        "champion": "momentum(deadzone=0.15) -> bracket(sl_mult=1.0,tp_mult=2.0)",
        "champion_val": {"sharpe": 1.8, "total_return": 0.06},
        "champion_holdout": {"sharpe": 1.2, "total_return": 0.04, "profit_factor": 1.3,
                             "max_drawdown": -0.05, "trade_count": 44, "win_rate": 0.52,
                             "avg_bars_held": 9.0},
        "champion_exit_breakdown": {"take_profit": {"n": 20, "avg_pnl": 0.004},
                                    "stop_loss": {"n": 24, "avg_pnl": -0.002}},
        "baseline": {"sharpe": -0.4, "total_return": -0.02},
        "luck_bar_sharpe": 1.0, "beats_luck_bar": True, "satisfied": True,
        "target_checks": {"sharpe": {"value": 1.2, "target": 1.0, "pass": True}},
    }

    def test_no_search_yet_is_cold_start_safe(self, tmp_path, monkeypatch):
        import status as status_mod
        monkeypatch.setattr(status_mod, "SEARCH_JSON", tmp_path / "missing.json")
        assert "No search run yet" in status_mod.format_search_telegram()

    def test_search_summary_reports_holdout_and_verdict(self):
        import status as status_mod
        msg = status_mod.format_search_telegram(self.SAMPLE)
        assert "BTC-USDT-SWAP" in msg
        assert "392 strategies tried" in msg
        assert "SATISFIED" in msg
        assert "take_profit: 20 trades" in msg

    def test_below_luck_bar_is_called_out(self):
        import status as status_mod
        msg = status_mod.format_search_telegram(
            dict(self.SAMPLE, beats_luck_bar=False, satisfied=False))
        assert "below the luck bar" in msg
        assert "Not there yet" in msg

    def test_trade_log_shows_both_reasons(self, tmp_path, monkeypatch):
        import csv as csv_mod
        import status as status_mod
        path = tmp_path / "champion_trades.csv"
        rows = [{
            "entry_ts": "2026-01-01 00:00:00+00:00", "exit_ts": "2026-01-01 01:00:00+00:00",
            "side": "long", "entry_px": 50000.0, "exit_px": 50500.0, "bars_held": 12,
            "gross_ret": 0.01, "fees": 0.001, "pnl": 0.009, "regime": "trending",
            "xgb_prob": 0.61, "chg_short": 0.4, "mr_zscore": -0.2, "vol_ratio": 2.1,
            "poc_dist_pct": 0.3,
            "entry_reason": "momentum: chg_short +0.40% beyond dead zone 0.15",
            "exit_reason": "take_profit: high 50510.00 >= target 50500.00",
        }]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        monkeypatch.setattr(status_mod, "CHAMPION_TRADES_CSV", path)
        monkeypatch.setattr(status_mod, "SEARCH_JSON", tmp_path / "missing.json")
        msg = status_mod.format_trades_telegram(limit=5)
        assert "momentum: chg_short" in msg
        assert "take_profit" in msg
        assert "+0.90%" in msg

    def test_no_trades_yet_is_cold_start_safe(self, tmp_path, monkeypatch):
        import status as status_mod
        monkeypatch.setattr(status_mod, "CHAMPION_TRADES_CSV", tmp_path / "missing.csv")
        assert "No trades logged yet" in status_mod.format_trades_telegram()

    def test_leaderboard_counts_holdout_reuse(self, tmp_path, monkeypatch):
        import status as status_mod
        monkeypatch.setattr(status_mod, "LEADERBOARD_CSV", tmp_path / "leaderboard.csv")

        class FakeStrat:
            name = "momentum(deadzone=0.15) -> flip()"

        finalists = [{"strategy": FakeStrat(), "metrics": {"sharpe": 1.0, "trade_count": 30},
                      "holdout": {"sharpe": 0.8, "trade_count": 25}}]
        status = {"run_date": "2026-01-01", "ts_utc": "2026-01-01T00:00:00+00:00",
                  "inst": "BTC-USDT-SWAP", "bar": "5m", "luck_bar_sharpe": 1.0,
                  "satisfied": False}
        assert status_mod.holdout_use_count(FakeStrat.name) == 0
        status_mod.append_leaderboard(status, finalists)
        status_mod.append_leaderboard(status, finalists)
        assert status_mod.holdout_use_count(FakeStrat.name) == 2
        assert len(status_mod.read_leaderboard()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
