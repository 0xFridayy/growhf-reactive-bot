"""
alpha_ml/ddqn.py — PHASE 2: Double DQN agent (PyTorch).

Standard DDQN: a small MLP Q-network, a target network updated on a lag, a
replay buffer, epsilon-greedy exploration, and the double-Q target
(action SELECTED by the online network, VALUED by the target network — the
fix for vanilla DQN's overestimation bias). Nothing exotic; this is the
well-tested reference algorithm, not a novel design — the "search for new
alpha" comes from what it's trained against (env.py's PerpTradingEnv), not
from the RL algorithm itself.

Run:  py ddqn.py --selftest      # tiny deterministic env, proves the wiring
Deps: numpy, torch
"""

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).with_name("models")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000, seed=None):
        self.buf = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DDQNAgent:
    def __init__(self, state_dim, action_dim, hidden=64, lr=1e-3, gamma=0.99,
                buffer_size=50000, batch_size=64, eps_start=1.0, eps_end=0.05,
                eps_decay_episodes=200, target_update_every=200, seed=None,
                device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_episodes = max(eps_decay_episodes, 1)
        self.target_update_every = target_update_every
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

        self.online = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, seed=seed)
        self.eps = eps_start
        self.learn_steps = 0

    def epsilon_for_episode(self, episode):
        frac = min(episode / self.eps_decay_episodes, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = torch.argmax(self.online(s2), dim=1)               # double-Q: select w/ online
            next_q = self.target(s2).gather(1, best_a.unsqueeze(1)).squeeze(1)  # value w/ target
            y = r + self.gamma * (1.0 - done) * next_q
        loss = F.smooth_l1_loss(q, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def train(self, env, episodes=100, warmup_steps=200, log_every=20):
        """Runs full episodes against `env` (env.py's PerpTradingEnv-shaped
        interface: reset()->obs, step(a)->(obs,r,done,info)). Returns per-
        episode total reward list."""
        history = []
        for ep in range(episodes):
            self.eps = self.epsilon_for_episode(ep)
            s = env.reset()
            done, ep_reward, losses = False, 0.0, []
            while not done:
                a = self.act(s)
                s2, r, done, _info = env.step(a)
                self.remember(s, a, r, s2, done)
                if len(self.buffer) >= max(warmup_steps, self.batch_size):
                    loss = self.learn()
                    if loss is not None:
                        losses.append(loss)
                s = s2
                ep_reward += r
            history.append(ep_reward)
            if log_every and (ep + 1) % log_every == 0:
                avg_loss = np.mean(losses) if losses else float("nan")
                print(f"  ep {ep+1:>4}/{episodes}  reward {ep_reward:+.5f}  "
                      f"eps {self.eps:.3f}  loss {avg_loss:.5f}")
        return history

    def evaluate(self, env, episodes=20, start_indices=None):
        """Greedy (eps=0) rollout, no learning. Returns per-episode reward list."""
        rewards = []
        for ep in range(episodes):
            start = start_indices[ep] if start_indices is not None else None
            s = env.reset(start_idx=start)
            done, ep_reward = False, 0.0
            while not done:
                a = self.act(s, greedy=True)
                s, r, done, _info = env.step(a)
                ep_reward += r
            rewards.append(ep_reward)
        return rewards

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dim": self.state_dim, "action_dim": self.action_dim,
            "online": self.online.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, **kwargs):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(ckpt["state_dim"], ckpt["action_dim"], **kwargs)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["online"])
        agent.eps = agent.eps_end  # a loaded checkpoint is for inference, not exploration
        return agent


# ----------------------------------------------------------------------
# SELF-TEST — tiny deterministic env: state reveals the correct action,
# proves the DDQN wiring (act/remember/learn/double-Q target/save-load)
# actually learns something, fast and without depending on env.py/features.py
# picking up a noisy real-market edge.
# ----------------------------------------------------------------------
class _SignEnv:
    """state = [signal, position, bars_held_frac]; optimal action tracks
    sign(signal). One-step episodes keep this a pure contextual-bandit check."""
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


def _selftest():
    print("alpha_ml/ddqn.py self-test — tiny deterministic sign-following env\n")
    env = _SignEnv(seed=0)
    agent = DDQNAgent(env.state_dim, env.ACTION_DIM, hidden=16, lr=5e-3,
                      buffer_size=2000, batch_size=32, eps_decay_episodes=150,
                      target_update_every=25, seed=0)

    pre_rewards = agent.evaluate(env, episodes=200)
    pre_mean = float(np.mean(pre_rewards))

    history = agent.train(env, episodes=300, warmup_steps=32, log_every=100)

    post_rewards = agent.evaluate(env, episodes=200)
    post_mean = float(np.mean(post_rewards))
    print(f"\n  pre-training  greedy avg reward (200 eval eps): {pre_mean:+.5f}")
    print(f"  post-training greedy avg reward (200 eval eps): {post_mean:+.5f}")
    assert post_mean > pre_mean, "DDQN did not improve on the trivially-learnable sign env"
    assert post_mean > 0.005, f"post-training reward too low ({post_mean:+.5f}) — optimal is ~0.01"

    ckpt = MODELS_DIR / "_selftest_ddqn.pt"
    agent.save(ckpt)
    loaded = DDQNAgent.load(ckpt, hidden=16)
    loaded_rewards = loaded.evaluate(env, episodes=200)
    assert abs(float(np.mean(loaded_rewards)) - post_mean) < 0.003, \
        "loaded checkpoint behaves differently from the agent that saved it"
    print(f"  save/load round-trip ✓ (loaded avg reward {np.mean(loaded_rewards):+.5f})")

    print(f"\n  PASS — learned to follow the state's sign "
          f"({pre_mean:+.5f} -> {post_mean:+.5f} avg reward, optimal ~+0.01000).")
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("give --selftest (train.py drives real training against env.PerpTradingEnv)")
