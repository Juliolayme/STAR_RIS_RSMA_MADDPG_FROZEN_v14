"""Single-agent TD3 (Twin Delayed DDPG) on flattened action space."""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import Actor, soft_update, hard_update
from networks.critic import TwinCritic
from utils import ReplayBuffer
from algorithms.maddpg.noise import OUNoise


def _to_t(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int],
                 td3_cfg: dict, net_cfg: dict, device: str = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, act_dim, hidden_sizes,
                           activation=net_cfg.get("activation", "relu"),
                           layer_norm=net_cfg.get("layer_norm", True),
                           ortho=net_cfg.get("ortho_init", True)).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_sizes,
                                  activation=net_cfg.get("activation", "relu"),
                                  layer_norm=net_cfg.get("layer_norm", True),
                                  ortho=net_cfg.get("ortho_init", True)).to(self.device)
        hard_update(self.actor, self.actor_target)
        self.critic = TwinCritic(obs_dim, act_dim, hidden_sizes,
                                 activation=net_cfg.get("activation", "relu"),
                                 layer_norm=net_cfg.get("layer_norm", True),
                                 ortho=net_cfg.get("ortho_init", True)).to(self.device)
        self.critic_target = TwinCritic(obs_dim, act_dim, hidden_sizes,
                                        activation=net_cfg.get("activation", "relu"),
                                        layer_norm=net_cfg.get("layer_norm", True),
                                        ortho=net_cfg.get("ortho_init", True)).to(self.device)
        hard_update(self.critic, self.critic_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=td3_cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=td3_cfg["critic_lr"])

        self.gamma = float(td3_cfg["gamma"])
        self.tau = float(td3_cfg["tau"])
        self.batch_size = int(td3_cfg["batch_size"])
        self.warmup_steps = int(td3_cfg["warmup_steps"])
        self.grad_clip = float(td3_cfg["grad_clip"])
        self.policy_noise = float(td3_cfg.get("policy_noise", 0.2))
        self.noise_clip = float(td3_cfg.get("noise_clip", 0.5))
        self.policy_delay = int(td3_cfg.get("policy_delay", 2))
        self.noise_start = float(td3_cfg["noise_sigma_start"])
        self.noise_end = float(td3_cfg["noise_sigma_end"])
        self.noise_decay = int(td3_cfg["noise_decay_steps"])

        self.noise = OUNoise(act_dim, sigma=self.noise_start, seed=seed)
        self.buffer = ReplayBuffer(int(td3_cfg["buffer_size"]), obs_dim, act_dim)
        self._global_step = 0
        self._learn_step = 0
        self._rng = np.random.default_rng(seed)

    def _sigma(self) -> float:
        frac = min(1.0, self._global_step / max(self.noise_decay, 1))
        return float(self.noise_start + (self.noise_end - self.noise_start) * frac)

    def reset_noise(self):
        self.noise.reset()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if explore and self._global_step < self.warmup_steps:
            return self._rng.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
        obs_t = _to_t(obs, self.device).unsqueeze(0)
        act = self.actor(obs_t).cpu().numpy()[0]
        if explore:
            self.noise.set_sigma(self._sigma())
            act = act + self.noise.sample()
        act = np.clip(act, -1.0, 1.0)
        if not np.all(np.isfinite(act)):
            act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)
        return act.astype(np.float32)

    def add_transition(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def increment_step(self):
        self._global_step += 1

    def learn(self) -> dict:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return {}
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size, rng=self._rng)
        obs_t = _to_t(obs, self.device)
        next_obs_t = _to_t(next_obs, self.device)
        act_t = _to_t(actions, self.device)
        rew_t = _to_t(rewards, self.device)
        done_t = _to_t(dones, self.device)

        with torch.no_grad():
            noise = (torch.randn_like(act_t) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(next_obs_t) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_target(next_obs_t, next_act)
            q_next = torch.min(q1_t, q2_t)
            y = rew_t + self.gamma * (1.0 - done_t) * q_next

        q1, q2 = self.critic(obs_t, act_t)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        info = {}
        if torch.isfinite(critic_loss):
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            gn = nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_opt.step()
            info["critic_loss"] = float(critic_loss.detach().cpu().item())
            info["critic_gradnorm"] = float(gn.detach().cpu().item() if hasattr(gn, "detach") else float(gn))

        self._learn_step += 1
        if self._learn_step % self.policy_delay == 0:
            actor_act = self.actor(obs_t)
            actor_loss = -self.critic.q1_only(obs_t, actor_act).mean()
            if torch.isfinite(actor_loss):
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                gn = nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                self.actor_opt.step()
                info["actor_loss"] = float(actor_loss.detach().cpu().item())
                info["actor_gradnorm"] = float(gn.detach().cpu().item() if hasattr(gn, "detach") else float(gn))

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)

        return info

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        s = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.actor_target.load_state_dict(s["actor_target"])
        self.critic_target.load_state_dict(s["critic_target"])
