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
            "inst": "BTC-USDT-SWAP", "bar": "5m", "xgb_auc": 0.61, "xgb_acc": 0.55,
            "ddqn_episodes": 100, "ddqn_avg_reward": 0.01,
            "bt_total_return": 0.05, "bt_sharpe": 1.5, "bt_sortino": 2.0,
            "bt_max_drawdown": -0.03, "bt_profit_factor": 1.4, "bt_trade_count": 50,
            "bt_win_rate": 0.5, "bt_turnover": 100.0, "bt_fees_paid": 0.01,
            "bt_gated_entries": 3, "ts_utc": "2026-01-01T00:00:00+00:00",
        })
        assert "BTC-USDT-SWAP" in msg
        assert "XGBoost AUC: 0.610" in msg
        assert "Trades: 50" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
