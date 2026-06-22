"""Replay buffers for single- and multi-agent algorithms."""
from __future__ import annotations
import numpy as np
from typing import Sequence


class ReplayBuffer:
    """Standard FIFO replay buffer for single-agent off-policy algorithms."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.next_obs[i] = next_obs
        self.dones[i, 0] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idxs = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idxs].copy(),
            self.actions[idxs].copy(),
            self.rewards[idxs].copy(),
            self.next_obs[idxs].copy(),
            self.dones[idxs].copy(),
        )


class MAReplayBuffer:
    """Replay buffer storing per-agent observations and actions for CTDE algorithms.

    Stores:
      - obs[a]: (capacity, obs_dim_a)
      - actions[a]: (capacity, act_dim_a)
      - rewards[a]: (capacity, 1) — supports per-agent reward; for cooperative,
        the shared reward is broadcast across agents.
    """
    def __init__(self, capacity: int, obs_dims: Sequence[int], act_dims: Sequence[int]):
        assert len(obs_dims) == len(act_dims)
        self.capacity = int(capacity)
        self.n_agents = len(obs_dims)
        self.obs = [np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims]
        self.next_obs = [np.zeros((self.capacity, d), dtype=np.float32) for d in obs_dims]
        self.actions = [np.zeros((self.capacity, d), dtype=np.float32) for d in act_dims]
        self.rewards = [np.zeros((self.capacity, 1), dtype=np.float32) for _ in range(self.n_agents)]
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs_list, action_list, reward_list, next_obs_list, done):
        i = self.idx
        for a in range(self.n_agents):
            self.obs[a][i] = obs_list[a]
            self.next_obs[a][i] = next_obs_list[a]
            self.actions[a][i] = action_list[a]
            self.rewards[a][i, 0] = float(reward_list[a])
        self.dones[i, 0] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idxs = rng.integers(0, self.size, size=batch_size)
        obs = [o[idxs].copy() for o in self.obs]
        next_obs = [o[idxs].copy() for o in self.next_obs]
        actions = [a[idxs].copy() for a in self.actions]
        rewards = [r[idxs].copy() for r in self.rewards]
        dones = self.dones[idxs].copy()
        return obs, actions, rewards, next_obs, dones
