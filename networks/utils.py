"""Common network initialization / update helpers."""
from __future__ import annotations
import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal initialization with proper bias zeroing — used as `module.apply(orthogonal_init)`."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


@torch.no_grad()
def hard_update(source: nn.Module, target: nn.Module) -> None:
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(p.data)


class MLP(nn.Module):
    """Multi-layer perceptron with optional LayerNorm and configurable activation."""
    def __init__(self, in_dim: int, hidden_sizes: list[int], out_dim: int,
                 activation: str = "relu", layer_norm: bool = True, last_activation: str | None = None):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}[activation.lower()]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if last_activation is not None:
            la = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[last_activation.lower()]
            layers.append(la())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
