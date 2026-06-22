"""MADDPG: Multi-agent DDPG with centralized critics and decentralized actors (CTDE)."""
from __future__ import annotations
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import Actor, CentralizedCritic, soft_update, hard_update
from utils import MAReplayBuffer
from .noise import OUNoise


def _to_t(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


class _PerAgent:
    """Holds actor, target actor, centralized critic, target critic, optimizers, and noise."""
    def __init__(self, obs_dim: int, act_dim: int,
                 total_obs_dim: int, total_act_dim: int,
                 hidden_sizes: list[int], cfg: dict, device: torch.device,
                 seed: int = 0):
        self.device = device
        self.act_dim = act_dim
        self.actor = Actor(obs_dim, act_dim, hidden_sizes,
                           activation=cfg.get("activation", "relu"),
                           layer_norm=cfg.get("layer_norm", True),
                           ortho=cfg.get("ortho_init", True)).to(device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_sizes,
                                  activation=cfg.get("activation", "relu"),
                                  layer_norm=cfg.get("layer_norm", True),
                                  ortho=cfg.get("ortho_init", True)).to(device)
        hard_update(self.actor, self.actor_target)

        self.critic = CentralizedCritic(total_obs_dim, total_act_dim, hidden_sizes,
                                        activation=cfg.get("activation", "relu"),
                                        layer_norm=cfg.get("layer_norm", True),
                                        ortho=cfg.get("ortho_init", True)).to(device)
        self.critic_target = CentralizedCritic(total_obs_dim, total_act_dim, hidden_sizes,
                                               activation=cfg.get("activation", "relu"),
                                               layer_norm=cfg.get("layer_norm", True),
                                               ortho=cfg.get("ortho_init", True)).to(device)
        hard_update(self.critic, self.critic_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

        self.noise = OUNoise(act_dim, sigma=cfg["noise_sigma_start"], seed=seed)


class MADDPG:
    def __init__(self, env_spec, hidden_sizes: list[int],
                 maddpg_cfg: dict, net_cfg: dict, device: str = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.cfg = maddpg_cfg
        self.n_agents = env_spec.n_agents
        self.obs_dims = env_spec.obs_dims
        self.act_dims = env_spec.act_dims
        self.total_obs_dim = int(sum(self.obs_dims))
        self.total_act_dim = int(sum(self.act_dims))

        cfg = {**net_cfg, **maddpg_cfg}
        self.agents = [
            _PerAgent(self.obs_dims[i], self.act_dims[i],
                      self.total_obs_dim, self.total_act_dim,
                      hidden_sizes, cfg, self.device, seed=seed + i)
            for i in range(self.n_agents)
        ]
        self.buffer = MAReplayBuffer(maddpg_cfg["buffer_size"], self.obs_dims, self.act_dims)
        self.gamma = float(maddpg_cfg["gamma"])
        self.tau = float(maddpg_cfg["tau"])
        self.batch_size = int(maddpg_cfg["batch_size"])
        self.warmup_steps = int(maddpg_cfg["warmup_steps"])
        self.grad_clip = float(maddpg_cfg["grad_clip"])
        self.noise_start = float(maddpg_cfg["noise_sigma_start"])
        self.noise_end = float(maddpg_cfg["noise_sigma_end"])
        self.noise_decay = int(maddpg_cfg["noise_decay_steps"])
        self.policy_update_every = int(maddpg_cfg.get("policy_update_every", 1))

        self._learn_step = 0
        self._global_step = 0
        self._rng = np.random.default_rng(seed)

    # -------------------------------------------------- exploration noise
    def _current_noise_sigma(self) -> float:
        frac = min(1.0, self._global_step / max(self.noise_decay, 1))
        return float(self.noise_start + (self.noise_end - self.noise_start) * frac)

    def reset_noise(self):
        for a in self.agents:
            a.noise.reset()

    # -------------------------------------------------- action selection
    @torch.no_grad()
    def select_actions(self, per_agent_obs: list[np.ndarray], explore: bool = True) -> list[np.ndarray]:
        # During warmup, ignore the (untrained) policy and sample uniformly — standard
        # practice (TD3, SpinningUp). Critical for high-dim action spaces like RIS phases.
        if explore and self._global_step < self.warmup_steps:
            return [self._rng.uniform(-1.0, 1.0, size=d).astype(np.float32) for d in self.act_dims]
        actions: list[np.ndarray] = []
        sigma = self._current_noise_sigma()
        for i, a in enumerate(self.agents):
            obs_t = _to_t(per_agent_obs[i], self.device).unsqueeze(0)
            act = a.actor(obs_t).cpu().numpy()[0]
            if explore:
                a.noise.set_sigma(sigma)
                act = act + a.noise.sample()
            act = np.clip(act, -1.0, 1.0)
            if not np.all(np.isfinite(act)):
                act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)
            actions.append(act.astype(np.float32))
        return actions

    def step_count(self) -> int:
        return self._global_step

    def increment_step(self):
        self._global_step += 1

    # -------------------------------------------------- buffer
    def add_transition(self, obs_list, action_list, reward, next_obs_list, done):
        """Cooperative reward broadcast across agents."""
        rewards = [reward] * self.n_agents
        self.buffer.add(obs_list, action_list, rewards, next_obs_list, done)

    # -------------------------------------------------- learning
    def learn(self) -> dict:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return {}
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size, rng=self._rng)
        obs_t = [_to_t(o, self.device) for o in obs]
        next_obs_t = [_to_t(o, self.device) for o in next_obs]
        act_t = [_to_t(a, self.device) for a in actions]
        rew_t = [_to_t(r, self.device) for r in rewards]
        done_t = _to_t(dones, self.device)

        # Pre-compute joint vectors.
        joint_obs = torch.cat(obs_t, dim=-1)
        joint_next_obs = torch.cat(next_obs_t, dim=-1)
        joint_act = torch.cat(act_t, dim=-1)

        # Target actions from each agent's TARGET actor (CTDE).
        with torch.no_grad():
            target_actions = [self.agents[i].actor_target(next_obs_t[i]) for i in range(self.n_agents)]
            joint_target_act = torch.cat(target_actions, dim=-1)

        info: dict = {}

        # ---- Critic updates ----
        for i, agent in enumerate(self.agents):
            with torch.no_grad():
                q_next = agent.critic_target(joint_next_obs, joint_target_act)
                y = rew_t[i] + self.gamma * (1.0 - done_t) * q_next
            q = agent.critic(joint_obs, joint_act)
            critic_loss = F.mse_loss(q, y)
            if not torch.isfinite(critic_loss):
                # Defensive: skip if NaN/Inf.
                continue
            agent.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            gn = nn.utils.clip_grad_norm_(agent.critic.parameters(), self.grad_clip)
            agent.critic_opt.step()
            info[f"critic_loss_{i}"] = float(critic_loss.detach().cpu().item())
            info[f"critic_gradnorm_{i}"] = float(gn.detach().cpu().item() if hasattr(gn, "detach") else float(gn))

        # ---- Critic target soft update (every learn step, per Lowe et al. 2017) ----
        # R2 reviewer fix: critic targets must track main critic continuously to maintain
        # Q-value stability. Only the actor (and its target) are subject to policy_delay.
        for agent in self.agents:
            soft_update(agent.critic, agent.critic_target, self.tau)

        # ---- Actor updates (with optional policy delay à la TD3) ----
        if self._learn_step % self.policy_update_every == 0:
            for i, agent in enumerate(self.agents):
                # Replace agent i's action with current actor output; others use sampled actions.
                actor_act = agent.actor(obs_t[i])
                act_list = list(act_t)
                act_list[i] = actor_act
                joint_act_pi = torch.cat(act_list, dim=-1)
                actor_loss = -agent.critic(joint_obs, joint_act_pi).mean()
                if not torch.isfinite(actor_loss):
                    continue
                agent.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                gn = nn.utils.clip_grad_norm_(agent.actor.parameters(), self.grad_clip)
                agent.actor_opt.step()
                info[f"actor_loss_{i}"] = float(actor_loss.detach().cpu().item())
                info[f"actor_gradnorm_{i}"] = float(gn.detach().cpu().item() if hasattr(gn, "detach") else float(gn))

            # Actor target soft update (gated by policy_delay).
            for agent in self.agents:
                soft_update(agent.actor, agent.actor_target, self.tau)

        self._learn_step += 1
        return info

    # -------------------------------------------------- checkpoint
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {}
        for i, a in enumerate(self.agents):
            state[f"actor_{i}"] = a.actor.state_dict()
            state[f"critic_{i}"] = a.critic.state_dict()
            state[f"actor_target_{i}"] = a.actor_target.state_dict()
            state[f"critic_target_{i}"] = a.critic_target.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location=self.device)
        for i, a in enumerate(self.agents):
            a.actor.load_state_dict(state[f"actor_{i}"])
            a.critic.load_state_dict(state[f"critic_{i}"])
            a.actor_target.load_state_dict(state[f"actor_target_{i}"])
            a.critic_target.load_state_dict(state[f"critic_target_{i}"])
