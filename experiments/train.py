"""Training drivers for MADDPG / DDPG / TD3 / PPO on the STAR-RIS RSMA env.

Includes:
- Adaptive Lagrangian QoS multiplier (raise lambda when satisfaction below target).
- Episode-level info aggregation: per-component rewards, |h_eff|, phase entropy,
  gradient norms, action-distribution stats, NaN watchdog.
- Deterministic periodic evaluation.
"""
from __future__ import annotations
import math
import os
import time
from collections import deque, defaultdict
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from env import StarRisRsmaEnv
from algorithms import MADDPG, DDPGAgent, TD3Agent, PPOAgent
from utils import Logger, ObservationNormalizer


# --------------------------------------------------------------------- helpers
def _set_seed(seed: int):
    """Deterministic seeding (M6 reviewer fix). Sets numpy, torch (CPU + CUDA),
    and PyTorch's deterministic algorithms. Required for paper-grade reproducibility.
    """
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism (matters on GPU; harmless on CPU).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _make_env(cfg: dict, seed: int, ris_mode: str = "optimized",
              equal_power: bool = False, qos_lambda_override: float | None = None) -> StarRisRsmaEnv:
    env_cfg = dict(cfg["env"])
    env_cfg["equal_power_mode"] = bool(equal_power)
    env = StarRisRsmaEnv(env_cfg, seed=seed, ris_mode=ris_mode)
    if qos_lambda_override is not None:
        env.set_qos_lambda(qos_lambda_override)
    return env


def _action_stats(action_arr: np.ndarray) -> dict:
    a = np.asarray(action_arr).reshape(-1)
    return {
        "act_mean": float(np.mean(a)),
        "act_std": float(np.std(a)),
        "act_abs_max": float(np.max(np.abs(a))),
        "act_sat_frac": float(np.mean(np.abs(a) > 0.95)),
    }


def _qos_lambda_update(env: StarRisRsmaEnv, cfg: dict, qos_window: deque,
                       ep: int | None = None, total_episodes: int | None = None) -> float:
    """Primal-dual style adaptive multiplier — moves toward target QoS satisfaction.

    Two-phase schedule: λ adapts during the primal-dual phase, then is FROZEN for the
    final fraction of training (qos_lambda_freeze_fraction). Freezing makes the reward
    stationary in the tail so the convergence curve (ma_return) can actually flatten —
    an adaptive λ otherwise keeps shifting the reward scale and the curve never settles.
    """
    freeze_frac = float(cfg["env"].get("qos_lambda_freeze_fraction", 1.0))
    if (total_episodes and ep is not None and freeze_frac < 1.0
            and ep >= freeze_frac * total_episodes):
        return env.qos_lambda            # frozen — reward scale held constant
    if not qos_window:
        return env.qos_lambda
    cur = float(np.mean(qos_window))
    tgt = float(cfg["env"].get("qos_target_satisfaction", 0.6))
    up = float(cfg["env"].get("qos_lambda_increase", 1.10))
    dn = float(cfg["env"].get("qos_lambda_decrease", 0.97))
    if cur < tgt:
        env.set_qos_lambda(env.qos_lambda * up)
    elif cur > tgt + 0.10:
        env.set_qos_lambda(env.qos_lambda * dn)
    return env.qos_lambda


def _aggregate_info(buf: dict, info: dict, reward: float):
    """Per-step info aggregation in a single episode."""
    for k in ("sum_rate", "rate_common", "qos_violation_l1", "qos_violation_l2",
              "h_eff_abs_mean", "h_eff_abs_R", "h_eff_abs_T",
              "phase_entropy_R", "phase_entropy_T", "phase_var_R", "phase_var_T",
              "beta_r_mean", "common_power_frac", "total_power_W",
              "reward_sr", "reward_qos", "reward_pwr"):
        if k in info:
            buf[k].append(float(info[k]))
    buf["qos_satisfied"].append(float(info.get("qos_satisfied", False)))
    buf["reward"].append(float(reward))


def _summarize(buf: dict) -> dict:
    out = {}
    for k, v in buf.items():
        if not v:
            continue
        out[f"{k}_mean"] = float(np.mean(v))
    return out


# ============================================================ MADDPG
def train_maddpg(cfg: dict, total_episodes: int | None = None,
                 run_name: str = "maddpg", log_dir: str = "logs",
                 ckpt_dir: str = "checkpoints", ris_mode: str = "optimized",
                 seed_override: int | None = None,
                 disable_qos_penalty: bool = False,
                 disable_obs_norm: bool = False) -> dict:
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    _set_seed(seed)
    device = _select_device(cfg.get("device", "auto"))
    cfg2 = dict(cfg)
    if disable_qos_penalty:
        cfg2 = {**cfg, "env": dict(cfg["env"])}
        cfg2["env"]["qos_lambda_init"] = 0.0
        cfg2["env"]["qos_lambda_max"] = 0.0
        cfg2["env"]["reward_qos_bonus"] = 0.0
    env = _make_env(cfg2, seed, ris_mode=ris_mode)
    spec = env.spec()
    agent = MADDPG(spec,
                   hidden_sizes=cfg["networks"]["hidden_sizes"],
                   maddpg_cfg=cfg["maddpg"],
                   net_cfg=cfg["networks"],
                   device=device, seed=seed)
    # Per-agent obs normalizers (CTDE: each agent has its own dim).
    obs_norms = [ObservationNormalizer(shape=(d,)) for d in spec.obs_dims]
    if disable_obs_norm:
        for o in obs_norms:
            o.enabled = False
    obs_norm = obs_norms  # legacy variable name kept; now a list
    logger = Logger(log_dir, run_name)

    total_episodes = int(total_episodes or cfg["training"]["total_episodes"])
    eval_every = int(cfg["training"]["eval_every"])
    eval_eps = int(cfg["training"]["eval_episodes"])
    ckpt_every = int(cfg["training"]["checkpoint_every"])
    ckpt_path = os.path.join(ckpt_dir, run_name, "latest.pt")
    best_path = os.path.join(ckpt_dir, run_name, "best.pt")
    smoothw = int(cfg["training"].get("reward_smoothing_window", 20))

    history = {"episode_return": [], "sum_rate": [], "qos_satisfied": [],
               "qos_lambda": [], "h_eff_abs_T": [], "phase_entropy_T": [],
               "rate_common": [], "common_power_frac": [], "ma_return": []}
    best_return = -math.inf
    return_window = deque(maxlen=smoothw)
    qos_window = deque(maxlen=10)  # episodes used for adaptive λ

    pbar = tqdm(range(total_episodes), desc=run_name, ncols=110)
    for ep in pbar:
        env.reset(seed=seed + ep)
        # CTDE: get per-agent local observations directly from env.
        per_agent_obs = [n(o, update=True) for n, o in zip(obs_norms, env.per_agent_observations(None))]
        agent.reset_noise()
        buf = defaultdict(list)
        ep_return, steps = 0.0, 0
        for t in range(env.max_steps):
            actions = agent.select_actions(per_agent_obs, explore=True)
            next_obs, reward, term, trunc, info = env.step(actions)
            done = term or trunc
            if not math.isfinite(reward):
                logger.buffer("nan_reward_count", 1.0)
                continue
            next_per_agent = [n(o, update=True) for n, o in zip(obs_norms, env.per_agent_observations(None))]
            agent.add_transition(per_agent_obs, actions, reward, next_per_agent, float(done))
            agent.increment_step()
            losses = agent.learn()
            for k, v in losses.items():
                logger.buffer(k, v)
            # Action stats (only sample sometimes to avoid log spam).
            if (steps % 10) == 0:
                flat = np.concatenate([np.asarray(a).reshape(-1) for a in actions])
                for ak, av in _action_stats(flat).items():
                    logger.buffer(ak, av)
            _aggregate_info(buf, info, reward)
            ep_return += reward
            per_agent_obs = next_per_agent
            steps += 1
            if done:
                break

        ep_summary = _summarize(buf)
        qos_prob = ep_summary.get("qos_satisfied_mean", 0.0)
        return_window.append(ep_return)
        qos_window.append(qos_prob)
        cur_lambda = _qos_lambda_update(env, cfg, qos_window, ep=ep, total_episodes=total_episodes)
        history["episode_return"].append(ep_return)
        history["sum_rate"].append(ep_summary.get("sum_rate_mean", 0.0))
        history["qos_satisfied"].append(qos_prob)
        history["qos_lambda"].append(cur_lambda)
        history["h_eff_abs_T"].append(ep_summary.get("h_eff_abs_T_mean", 0.0))
        history["phase_entropy_T"].append(ep_summary.get("phase_entropy_T_mean", 0.0))
        history["rate_common"].append(ep_summary.get("rate_common_mean", 0.0))
        history["common_power_frac"].append(ep_summary.get("common_power_frac_mean", 0.0))
        history["ma_return"].append(float(np.mean(return_window)))

        log_row = {"episode_return": ep_return,
                   "ma_return": float(np.mean(return_window)),
                   "qos_prob": qos_prob, "qos_lambda": cur_lambda,
                   "noise_sigma": agent._current_noise_sigma()}
        log_row.update({k: v for k, v in ep_summary.items()})
        logger.log(ep, log_row)
        logger.flush_buffers(ep)
        pbar.set_postfix({"ret": f"{ep_return:.2f}", "MA": f"{np.mean(return_window):.2f}",
                          "qos": f"{qos_prob:.2f}", "λ": f"{cur_lambda:.2f}",
                          "|h_T|": f"{ep_summary.get('h_eff_abs_T_mean', 0.0):.2e}"})
        if (ep + 1) % ckpt_every == 0:
            agent.save(ckpt_path)
        if ep_return > best_return:
            best_return = ep_return
            agent.save(best_path)
        if (ep + 1) % eval_every == 0:
            em = evaluate_agent(env_cfg=cfg, agent=agent, kind="maddpg",
                                obs_norm=obs_norms, episodes=eval_eps,
                                seed=seed + 9000 + ep,
                                qos_lambda=env.qos_lambda)
            logger.log(ep, {f"eval_{k}": v for k, v in em.items()})

    agent.save(ckpt_path)
    # Persist per-agent observation-normalizer stats so --skip-train can reproduce eval.
    for i, on in enumerate(obs_norms):
        on.save(os.path.join(ckpt_dir, run_name, f"obs_norm_{i}.npz"))
    logger.close()
    return {"agent": agent, "obs_norm": obs_norms, "history": history,
            "best_ckpt": best_path, "latest_ckpt": ckpt_path,
            "trained_qos_lambda": float(env.qos_lambda)}


# ============================================================ DDPG / TD3
def train_single_agent(cfg: dict, kind: str, total_episodes: int | None = None,
                       run_name: str | None = None, log_dir: str = "logs",
                       ckpt_dir: str = "checkpoints",
                       seed_override: int | None = None,
                       disable_qos_penalty: bool = False,
                       disable_obs_norm: bool = False) -> dict:
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    _set_seed(seed)
    device = _select_device(cfg.get("device", "auto"))
    cfg2 = cfg
    if disable_qos_penalty:
        cfg2 = {**cfg, "env": dict(cfg["env"])}
        cfg2["env"]["qos_lambda_init"] = 0.0
        cfg2["env"]["qos_lambda_max"] = 0.0
        cfg2["env"]["reward_qos_bonus"] = 0.0
    env = _make_env(cfg2, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if kind == "ddpg":
        agent = DDPGAgent(obs_dim, act_dim, cfg["networks"]["hidden_sizes"],
                          cfg["ddpg"], cfg["networks"], device=device, seed=seed)
    elif kind == "td3":
        agent = TD3Agent(obs_dim, act_dim, cfg["networks"]["hidden_sizes"],
                         cfg["td3"], cfg["networks"], device=device, seed=seed)
    else:
        raise ValueError(f"Unknown single-agent algorithm: {kind}")

    obs_norm = ObservationNormalizer(shape=(obs_dim,))
    if disable_obs_norm:
        obs_norm.enabled = False
    run_name = run_name or kind
    logger = Logger(log_dir, run_name)
    total_episodes = int(total_episodes or cfg["training"]["total_episodes"])
    eval_every = int(cfg["training"]["eval_every"])
    eval_eps = int(cfg["training"]["eval_episodes"])
    ckpt_every = int(cfg["training"]["checkpoint_every"])
    smoothw = int(cfg["training"].get("reward_smoothing_window", 20))
    ckpt_path = os.path.join(ckpt_dir, run_name, "latest.pt")
    best_path = os.path.join(ckpt_dir, run_name, "best.pt")

    history = {"episode_return": [], "sum_rate": [], "qos_satisfied": [],
               "qos_lambda": [], "h_eff_abs_T": [], "phase_entropy_T": [],
               "rate_common": [], "common_power_frac": [], "ma_return": []}
    best_return = -math.inf
    return_window = deque(maxlen=smoothw)
    qos_window = deque(maxlen=10)

    pbar = tqdm(range(total_episodes), desc=run_name, ncols=110)
    for ep in pbar:
        obs, _ = env.reset(seed=seed + ep)
        obs = obs_norm(obs, update=True)
        agent.reset_noise()
        buf = defaultdict(list)
        ep_return, steps = 0.0, 0
        for t in range(env.max_steps):
            action = agent.select_action(obs, explore=True)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if not math.isfinite(reward):
                logger.buffer("nan_reward_count", 1.0)
                continue
            next_obs_norm = obs_norm(next_obs, update=True)
            agent.add_transition(obs, action, reward, next_obs_norm, float(done))
            agent.increment_step()
            losses = agent.learn()
            for k, v in losses.items():
                logger.buffer(k, v)
            if (steps % 10) == 0:
                for ak, av in _action_stats(action).items():
                    logger.buffer(ak, av)
            _aggregate_info(buf, info, reward)
            ep_return += reward
            obs = next_obs_norm
            steps += 1
            if done:
                break

        ep_summary = _summarize(buf)
        qos_prob = ep_summary.get("qos_satisfied_mean", 0.0)
        return_window.append(ep_return)
        qos_window.append(qos_prob)
        cur_lambda = _qos_lambda_update(env, cfg, qos_window, ep=ep, total_episodes=total_episodes)
        history["episode_return"].append(ep_return)
        history["sum_rate"].append(ep_summary.get("sum_rate_mean", 0.0))
        history["qos_satisfied"].append(qos_prob)
        history["qos_lambda"].append(cur_lambda)
        history["h_eff_abs_T"].append(ep_summary.get("h_eff_abs_T_mean", 0.0))
        history["phase_entropy_T"].append(ep_summary.get("phase_entropy_T_mean", 0.0))
        history["rate_common"].append(ep_summary.get("rate_common_mean", 0.0))
        history["common_power_frac"].append(ep_summary.get("common_power_frac_mean", 0.0))
        history["ma_return"].append(float(np.mean(return_window)))

        log_row = {"episode_return": ep_return, "ma_return": float(np.mean(return_window)),
                   "qos_prob": qos_prob, "qos_lambda": cur_lambda}
        log_row.update({k: v for k, v in ep_summary.items()})
        logger.log(ep, log_row)
        logger.flush_buffers(ep)
        pbar.set_postfix({"ret": f"{ep_return:.2f}", "MA": f"{np.mean(return_window):.2f}",
                          "qos": f"{qos_prob:.2f}", "λ": f"{cur_lambda:.2f}",
                          "|h_T|": f"{ep_summary.get('h_eff_abs_T_mean', 0.0):.2e}"})
        if (ep + 1) % ckpt_every == 0:
            agent.save(ckpt_path)
        if ep_return > best_return:
            best_return = ep_return
            agent.save(best_path)
        if (ep + 1) % eval_every == 0:
            em = evaluate_agent(env_cfg=cfg, agent=agent, kind=kind,
                                obs_norm=obs_norm, episodes=eval_eps,
                                seed=seed + 9000 + ep,
                                qos_lambda=env.qos_lambda)
            logger.log(ep, {f"eval_{k}": v for k, v in em.items()})

    agent.save(ckpt_path)
    obs_norm.save(os.path.join(ckpt_dir, run_name, "obs_norm.npz"))
    logger.close()
    return {"agent": agent, "obs_norm": obs_norm, "history": history,
            "best_ckpt": best_path, "latest_ckpt": ckpt_path,
            "trained_qos_lambda": float(env.qos_lambda)}


# ============================================================ PPO
def train_ppo(cfg: dict, total_episodes: int | None = None,
              run_name: str = "ppo", log_dir: str = "logs",
              ckpt_dir: str = "checkpoints",
              seed_override: int | None = None) -> dict:
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    _set_seed(seed)
    device = _select_device(cfg.get("device", "auto"))
    env = _make_env(cfg, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, cfg["networks"]["hidden_sizes"],
                     cfg["ppo"], cfg["networks"], device=device, seed=seed)
    obs_norm = ObservationNormalizer(shape=(obs_dim,))
    logger = Logger(log_dir, run_name)
    total_episodes = int(total_episodes or cfg["training"]["total_episodes"])
    ckpt_every = int(cfg["training"]["checkpoint_every"])
    eval_every = int(cfg["training"]["eval_every"])
    eval_eps = int(cfg["training"]["eval_episodes"])
    smoothw = int(cfg["training"].get("reward_smoothing_window", 20))
    ckpt_path = os.path.join(ckpt_dir, run_name, "latest.pt")
    best_path = os.path.join(ckpt_dir, run_name, "best.pt")

    history = {"episode_return": [], "sum_rate": [], "qos_satisfied": [],
               "qos_lambda": [], "h_eff_abs_T": [], "phase_entropy_T": [],
               "rate_common": [], "common_power_frac": [], "ma_return": []}
    best_return = -math.inf
    return_window = deque(maxlen=smoothw)
    qos_window = deque(maxlen=10)

    pbar = tqdm(range(total_episodes), desc=run_name, ncols=110)
    for ep in pbar:
        obs, _ = env.reset(seed=seed + ep)
        obs = obs_norm(obs, update=True)
        buf = defaultdict(list)
        ep_return, steps = 0.0, 0
        for t in range(env.max_steps):
            action, log_prob, value = agent.select_action(obs, explore=True)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if not math.isfinite(reward):
                logger.buffer("nan_reward_count", 1.0)
                continue
            next_obs_norm = obs_norm(next_obs, update=True)
            agent.store(obs, action, log_prob, reward, value, float(done))
            if (steps % 10) == 0:
                for ak, av in _action_stats(action).items():
                    logger.buffer(ak, av)
            _aggregate_info(buf, info, reward)
            ep_return += reward
            obs = next_obs_norm
            steps += 1
            if agent.buffer_full():
                last_v = 0.0 if done else agent.value(obs)
                losses = agent.learn(last_v)
                for k, v in losses.items():
                    logger.buffer(k, v)
            if done:
                break
        if agent.rollout.size > 0 and (ep == total_episodes - 1 or agent.buffer_full()):
            last_v = 0.0 if (term or trunc) else agent.value(obs)
            losses = agent.learn(last_v)
            for k, v in losses.items():
                logger.buffer(k, v)

        ep_summary = _summarize(buf)
        qos_prob = ep_summary.get("qos_satisfied_mean", 0.0)
        return_window.append(ep_return)
        qos_window.append(qos_prob)
        cur_lambda = _qos_lambda_update(env, cfg, qos_window, ep=ep, total_episodes=total_episodes)
        history["episode_return"].append(ep_return)
        history["sum_rate"].append(ep_summary.get("sum_rate_mean", 0.0))
        history["qos_satisfied"].append(qos_prob)
        history["qos_lambda"].append(cur_lambda)
        history["h_eff_abs_T"].append(ep_summary.get("h_eff_abs_T_mean", 0.0))
        history["phase_entropy_T"].append(ep_summary.get("phase_entropy_T_mean", 0.0))
        history["rate_common"].append(ep_summary.get("rate_common_mean", 0.0))
        history["common_power_frac"].append(ep_summary.get("common_power_frac_mean", 0.0))
        history["ma_return"].append(float(np.mean(return_window)))

        log_row = {"episode_return": ep_return, "ma_return": float(np.mean(return_window)),
                   "qos_prob": qos_prob, "qos_lambda": cur_lambda}
        log_row.update({k: v for k, v in ep_summary.items()})
        logger.log(ep, log_row)
        logger.flush_buffers(ep)
        pbar.set_postfix({"ret": f"{ep_return:.2f}", "MA": f"{np.mean(return_window):.2f}",
                          "qos": f"{qos_prob:.2f}", "λ": f"{cur_lambda:.2f}"})
        if (ep + 1) % ckpt_every == 0:
            agent.save(ckpt_path)
        if ep_return > best_return:
            best_return = ep_return
            agent.save(best_path)
        if (ep + 1) % eval_every == 0:
            em = evaluate_agent(env_cfg=cfg, agent=agent, kind="ppo",
                                obs_norm=obs_norm, episodes=eval_eps,
                                seed=seed + 9000 + ep,
                                qos_lambda=env.qos_lambda)
            logger.log(ep, {f"eval_{k}": v for k, v in em.items()})

    agent.save(ckpt_path)
    obs_norm.save(os.path.join(ckpt_dir, run_name, "obs_norm.npz"))
    logger.close()
    return {"agent": agent, "obs_norm": obs_norm, "history": history,
            "best_ckpt": best_path, "latest_ckpt": ckpt_path,
            "trained_qos_lambda": float(env.qos_lambda)}


# ============================================================ evaluation
def evaluate_agent(env_cfg: dict, agent, kind: str,
                   obs_norm,                                  # ObservationNormalizer or list[ObservationNormalizer]
                   episodes: int = 5, seed: int = 12345,
                   ris_mode: str = "optimized",
                   equal_power: bool = False,
                   qos_lambda: float | None = None) -> dict:
    """Deterministic-policy evaluation with rich per-step diagnostics.

    qos_lambda: if provided, sets the env's QoS Lagrangian multiplier (P3 reviewer fix —
    keeps reward scale consistent with the training-end λ).
    """
    env = _make_env(env_cfg, seed, ris_mode=ris_mode, equal_power=equal_power,
                    qos_lambda_override=qos_lambda)
    rets, srs, qoss, lats = [], [], [], []
    rates_common, h_T_abs, phase_ent_T, common_frac = [], [], [], []
    per_user_rates = []
    is_maddpg = (kind == "maddpg")
    is_list_norm = isinstance(obs_norm, list)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if is_maddpg:
            per_agent_obs = env.per_agent_observations(None)
            if is_list_norm and obs_norm is not None:
                per_agent_obs = [n(o, update=False) for n, o in zip(obs_norm, per_agent_obs)]
        else:
            obs_in = obs_norm(obs, update=False) if (obs_norm is not None and not is_list_norm) else obs.astype(np.float32)
        ep_ret, steps = 0.0, 0
        ep_buf = defaultdict(list)
        ep_per_user = []
        for t in range(env.max_steps):
            t0 = time.perf_counter()
            if is_maddpg:
                actions = agent.select_actions(per_agent_obs, explore=False)
            elif kind in ("ddpg", "td3"):
                actions = agent.select_action(obs_in, explore=False)
            elif kind == "ppo":
                actions, _, _ = agent.select_action(obs_in, explore=False)
            else:
                raise ValueError(kind)
            lats.append((time.perf_counter() - t0) * 1000.0)
            next_obs, reward, term, trunc, info = env.step(actions)
            if is_maddpg:
                per_agent_obs = env.per_agent_observations(None)
                if is_list_norm and obs_norm is not None:
                    per_agent_obs = [n(o, update=False) for n, o in zip(obs_norm, per_agent_obs)]
            else:
                obs_in = obs_norm(next_obs, update=False) if (obs_norm is not None and not is_list_norm) else next_obs.astype(np.float32)
            ep_ret += reward
            _aggregate_info(ep_buf, info, reward)
            ep_per_user.append(info.get("per_user_rate", np.zeros(env.K)))
            steps += 1
            if term or trunc:
                break
        rets.append(ep_ret)
        srs.append(float(np.mean(ep_buf.get("sum_rate", [0.0]))))
        qoss.append(float(np.mean(ep_buf.get("qos_satisfied", [0.0]))))
        rates_common.append(float(np.mean(ep_buf.get("rate_common", [0.0]))))
        h_T_abs.append(float(np.mean(ep_buf.get("h_eff_abs_T", [0.0]))))
        phase_ent_T.append(float(np.mean(ep_buf.get("phase_entropy_T", [0.0]))))
        common_frac.append(float(np.mean(ep_buf.get("common_power_frac", [0.0]))))
        per_user_rates.append(np.mean(np.stack(ep_per_user, axis=0), axis=0))
    per_user_rates = np.stack(per_user_rates, axis=0)

    return {
        "return_mean": float(np.mean(rets)),
        "return_std":  float(np.std(rets)),
        "sum_rate_mean": float(np.mean(srs)),
        "sum_rate_std":  float(np.std(srs)),
        "qos_prob": float(np.mean(qoss)),
        "qos_prob_std": float(np.std(qoss)),
        "rate_common_mean": float(np.mean(rates_common)),
        "h_eff_abs_T_mean": float(np.mean(h_T_abs)),
        "phase_entropy_T_mean": float(np.mean(phase_ent_T)),
        "common_power_frac_mean": float(np.mean(common_frac)),
        "per_user_rate_mean": per_user_rates.mean(axis=0).tolist(),
        "latency_ms_mean": float(np.mean(lats)) if lats else 0.0,
        "latency_ms_std":  float(np.std(lats)) if lats else 0.0,
    }
