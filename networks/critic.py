"""Critic networks: single-agent Q, twin Q (TD3), centralized multi-agent Q."""
from __future__ import annotations
import torch
import torch.nn as nn
from .utils import MLP, orthogonal_init


class Critic(nn.Module):
    """Single Q(s, a) head."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int],
                 activation: str = "relu", layer_norm: bool = True, ortho: bool = True):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, hidden_sizes, 1,
                     activation=activation, layer_norm=layer_norm, last_activation=None)
        if ortho:
            orthogonal_init(self.q, gain=1.0)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))


class TwinCritic(nn.Module):
    """Two independent Q heads for TD3's min-double-Q target."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int],
                 activation: str = "relu", layer_norm: bool = True, ortho: bool = True):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, hidden_sizes, 1,
                      activation=activation, layer_norm=layer_norm)
        self.q2 = MLP(obs_dim + act_dim, hidden_sizes, 1,
                      activation=activation, layer_norm=layer_norm)
        if ortho:
            orthogonal_init(self.q1, gain=1.0)
            orthogonal_init(self.q2, gain=1.0)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs: torch.Tensor, act: torch.Tensor):
        return self.q1(torch.cat([obs, act], dim=-1))


class CentralizedCritic(nn.Module):
    """Q_i(o_1, ..., o_N, a_1, ..., a_N) for MADDPG."""
    def __init__(self, total_obs_dim: int, total_act_dim: int, hidden_sizes: list[int],
                 activation: str = "relu", layer_norm: bool = True, ortho: bool = True):
        super().__init__()
        self.q = MLP(total_obs_dim + total_act_dim, hidden_sizes, 1,
                     activation=activation, layer_norm=layer_norm)
        if ortho:
            orthogonal_init(self.q, gain=1.0)

    def forward(self, obs_concat: torch.Tensor, act_concat: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs_concat, act_concat], dim=-1))


class ValueNet(nn.Module):
    """State-value V(s) — used by PPO."""
    def __init__(self, obs_dim: int, hidden_sizes: list[int],
                 activation: str = "tanh", layer_norm: bool = False, ortho: bool = True):
        super().__init__()
        self.v = MLP(obs_dim, hidden_sizes, 1, activation=activation,
                     layer_norm=layer_norm, last_activation=None)
        if ortho:
            orthogonal_init(self.v, gain=1.0)
            last_linear = [m for m in self.v.net if isinstance(m, nn.Linear)][-1]
            nn.init.orthogonal_(last_linear.weight, gain=1.0)
            nn.init.zeros_(last_linear.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)
