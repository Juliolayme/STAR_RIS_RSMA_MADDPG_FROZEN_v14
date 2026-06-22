"""Ornstein-Uhlenbeck process for temporally correlated exploration noise."""
import numpy as np


class OUNoise:
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2,
                 seed: int | None = None):
        self.size = size
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.state = np.full(self.size, self.mu, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = np.full(self.size, self.mu, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.size)
        self.state = self.state + dx
        return self.state.astype(np.float32)

    def set_sigma(self, sigma: float):
        self.sigma = float(sigma)
