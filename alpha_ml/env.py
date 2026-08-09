"""
alpha_ml/env.py — PHASE 2 environment: turn a historical feature+price series
into an episodic trading env a DDQN agent can act in.

Action space is the target position for the NEXT bar: 0=flat, 1=long,
2=short. Reward is that position's realized bar-over-bar return minus a
turnover fee — the standard formulation for "learn a position policy",
letting the agent discover entries, exits, holds AND direction flips as one
policy instead of hand-coded rules. This is where "new alpha" gets searched:
the agent is free to disagree with the momentum-following default the phase-1
XGBoost model was trained against.

Deps: numpy, pandas
"""

import numpy as np


class PerpTradingEnv:
    ACTIONS = {0: 0.0, 1: 1.0, 2: -1.0}  # flat, long, short
    ACTION_DIM = 3

    def __init__(self, df, feature_cols, episode_len=200, fee=0.0005, seed=None):
        """df needs columns `feature_cols` + `c` (close), oldest-first,
        already leakage-safe (i.e. features.py's assemble() output)."""
        self.df = df.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.episode_len = episode_len
        self.fee = fee
        self.rng = np.random.default_rng(seed)
        self.n = len(self.df)
        if self.n <= episode_len + 1:
            raise ValueError(f"need > episode_len+1 rows, got {self.n}")
        self.state_dim = len(self.feature_cols) + 2  # + position + bars_held_frac
        self._closes = self.df["c"].to_numpy(dtype=float)
        self._feats = self.df[self.feature_cols].to_numpy(dtype=float)
        self.position = 0.0
        self.bars_held = 0
        self.equity = 0.0
        self.t = 0
        self.start = 0

    def reset(self, start_idx=None):
        max_start = self.n - self.episode_len - 1
        self.start = int(start_idx) if start_idx is not None else int(self.rng.integers(0, max_start + 1))
        self.t = self.start
        self.position = 0.0
        self.bars_held = 0
        self.equity = 0.0
        return self._obs()

    def _obs(self):
        bars_frac = min(self.bars_held / max(self.episode_len, 1), 1.0)
        return np.concatenate([self._feats[self.t], [self.position, bars_frac]]).astype(np.float32)

    def step(self, action):
        target = self.ACTIONS[int(action)]
        turnover = abs(target - self.position)
        c0, c1 = self._closes[self.t], self._closes[self.t + 1]
        bar_ret = (c1 - c0) / c0 if c0 else 0.0
        reward = target * bar_ret - self.fee * turnover

        self.bars_held = 0 if target != self.position else self.bars_held + 1
        self.position = target
        self.equity += reward
        self.t += 1
        done = (self.t - self.start) >= self.episode_len or self.t >= self.n - 1
        info = {"bar_ret": bar_ret, "turnover": turnover, "equity": self.equity, "position": self.position}
        return self._obs(), float(reward), done, info


if __name__ == "__main__":
    from features import assemble, synthetic_market
    market = synthetic_market(n=800, planted=True)
    df, cols = assemble(market, window=48)
    env = PerpTradingEnv(df, cols, episode_len=100, fee=0.0005, seed=1)
    obs = env.reset(start_idx=0)
    print(f"state_dim={env.state_dim}  action_dim={env.ACTION_DIM}  obs.shape={obs.shape}")
    total = 0.0
    done = False
    while not done:
        a = env.rng.integers(0, env.ACTION_DIM)
        obs, r, done, info = env.step(a)
        total += r
    print(f"random-policy 100-bar episode total reward: {total:+.5f}")
