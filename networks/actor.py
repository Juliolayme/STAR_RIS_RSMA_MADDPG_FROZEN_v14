"""Deterministic and stochastic actor networks."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import MLP, orthogonal_init


class Actor(nn.Module):
    """Deterministic actor outputting actions in [-1, 1] via tanh.

    The environment is responsible for mapping [-1, 1] to physical quantities
    (powers, splits, phases). Keeping the network output bounded is the simplest
    way to stabilize training across heterogeneous action semantics.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int],
                 activation: str = "relu", layer_norm: bool = True, ortho: bool = True):
        super().__init__()
        self.net = MLP(obs_dim, hidden_sizes, act_dim,
                       activation=activation,
                       layer_norm=layer_norm,
                       last_activation=None)
        if ortho:
            orthogonal_init(self.net, gain=1.0)
            # Final layer with smaller gain for numerical stability at init.
            last_linear = [m for m in self.net.net if isinstance(m, nn.Linear)][-1]
            nn.init.orthogonal_(last_linear.weight, gain=0.01)
            nn.init.zeros_(last_linear.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        # Use tanh squashing; environment denormalizes.
        return torch.tanh(logits)


class StochasticActor(nn.Module):
    """Gaussian policy with state-independent log-std (used by PPO)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int],
                 activation: str = "tanh", layer_norm: bool = False, ortho: bool = True,
                 log_std_init: float = -0.5):
        super().__init__()
        self.mean_net = MLP(obs_dim, hidden_sizes, act_dim,
                            activation=activation, layer_norm=layer_norm,
                            last_activation=None)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)
        if ortho:
            orthogonal_init(self.mean_net, gain=1.0)
            last_linear = [m for m in self.mean_net.net if isinstance(m, nn.Linear)][-1]
            nn.init.orthogonal_(last_linear.weight, gain=0.01)
            nn.init.zeros_(last_linear.bias)

    def forward(self, obs: torch.Tensor):
        mean = self.mean_net(obs)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.forward(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()
        # Apply tanh squash; environment maps [-1,1] to physical action.
        squashed = torch.tanh(raw)
        # Tanh log-prob correction (Haarnoja 2018, SAC appendix).
        log_prob = dist.log_prob(raw).sum(-1) - torch.log(1.0 - squashed.pow(2) + 1e-6).sum(-1)
        return squashed, log_prob, raw

    def log_prob(self, obs: torch.Tensor, squashed_action: torch.Tensor):
        dist = self.forward(obs)
        # Invert tanh; clamp for numerical safety.
        clamped = torch.clamp(squashed_action, -1.0 + 1e-6, 1.0 - 1e-6)
        raw = 0.5 * torch.log((1 + clamped) / (1 - clamped))
        log_prob = dist.log_prob(raw).sum(-1) - torch.log(1 - clamped.pow(2) + 1e-6).sum(-1)
        return log_prob, dist.entropy().sum(-1)
