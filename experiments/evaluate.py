"""Post-training evaluation sweeps with multi-seed mean ± std and 95% CI."""
from __future__ import annotations
import copy
import time
import numpy as np

from env import StarRisRsmaEnv
from utils.metrics import dbm_to_watt, confidence_interval
from experiments.train import evaluate_agent, _make_env


_KIND_ALIAS = {
    "MADDPG": "maddpg", "DDPG": "ddpg", "TD3": "td3", "PPO": "ppo",
    "FixedRIS": "maddpg", "RandomRIS": "maddpg", "NoRIS": "maddpg",
    "AnalyticalRIS": "maddpg", "EqualPowerLearned": "maddpg",
    "EqualPowerFixed": "maddpg",
    "NoQoSPenalty": "maddpg", "NoRewardNorm": "maddpg",
}


def _eval_multi_seed(agent, kind, obs_norm, env_cfg, seeds,
                     ris_mode="optimized", equal_power=False,
                     qos_lambda: float | None = None) -> dict:
    kind = _KIND_ALIAS.get(kind, kind).lower()
    rets, srs, qoss, lats, rc, htabs, pent, cfrac = [], [], [], [], [], [], [], []
    for s in seeds:
        m = evaluate_agent(env_cfg=env_cfg, agent=agent, kind=kind,
                           obs_norm=obs_norm,
                           episodes=env_cfg["evaluation"]["num_episodes"],
                           seed=int(s), ris_mode=ris_mode, equal_power=equal_power,
                           qos_lambda=qos_lambda)
        rets.append(m["return_mean"]);  srs.append(m["sum_rate_mean"])
        qoss.append(m["qos_prob"]);     lats.append(m["latency_ms_mean"])
        rc.append(m["rate_common_mean"]); htabs.append(m["h_eff_abs_T_mean"])
        pent.append(m["phase_entropy_T_mean"]); cfrac.append(m["common_power_frac_mean"])
    sr_m, sr_ci, sr_std = confidence_interval(np.array(srs))
    q_m,  q_ci,  q_std  = confidence_interval(np.array(qoss))
    r_m,  r_ci,  r_std  = confidence_interval(np.array(rets))
    l_m,  l_ci,  l_std  = confidence_interval(np.array(lats))
    return {
        "sum_rate_mean": sr_m, "sum_rate_ci": sr_ci, "sum_rate_std": sr_std,
        "qos_mean": q_m, "qos_ci": q_ci, "qos_std": q_std,
        "return_mean": r_m, "return_ci": r_ci, "return_std": r_std,
        "latency_ms_mean": l_m, "latency_ms_ci": l_ci,
        "rate_common_mean": float(np.mean(rc)),
        "h_eff_abs_T_mean": float(np.mean(htabs)),
        "phase_entropy_T_mean": float(np.mean(pent)),
        "common_power_frac_mean": float(np.mean(cfrac)),
    }


def sweep_power(agents: dict, obs_norms: dict, cfg: dict) -> dict:
    seeds = cfg["evaluation"]["seeds"]
    p_list = cfg["evaluation"]["power_sweep_dbm"]
    results = {algo: {"x": p_list, "mean": [], "ci": [], "std": [],
                      "qos_mean": [], "qos_ci": []} for algo in agents}
    for p_dbm in p_list:
        env_cfg = copy.deepcopy(cfg)
        env_cfg["env"]["p_max_dbm"] = float(p_dbm)
        for algo, agent in agents.items():
            m = _eval_multi_seed(agent, algo, obs_norms[algo], env_cfg, seeds,
                                 ris_mode=("fixed" if algo == "FixedRIS" else "optimized"))
            results[algo]["mean"].append(m["sum_rate_mean"])
            results[algo]["ci"].append(m["sum_rate_ci"])
            results[algo]["std"].append(m["sum_rate_std"])
            results[algo]["qos_mean"].append(m["qos_mean"])
            results[algo]["qos_ci"].append(m["qos_ci"])
    return results


def sweep_N(agents: dict, obs_norms: dict, cfg: dict) -> dict:
    seeds = cfg["evaluation"]["seeds"]
    n_list = cfg["evaluation"]["n_sweep"]
    results = {algo: {"x": n_list, "mean": [], "ci": [], "std": []} for algo in agents}
    trained_N = cfg["env"]["num_ris_elements"]
    for N in n_list:
        if int(N) != int(trained_N):
            for algo in agents:
                results[algo]["mean"].append(float("nan"))
                results[algo]["ci"].append(0.0); results[algo]["std"].append(0.0)
            continue
        env_cfg = copy.deepcopy(cfg); env_cfg["env"]["num_ris_elements"] = int(N)
        for algo, agent in agents.items():
            m = _eval_multi_seed(agent, algo, obs_norms[algo], env_cfg, seeds,
                                 ris_mode=("fixed" if algo == "FixedRIS" else "optimized"))
            results[algo]["mean"].append(m["sum_rate_mean"])
            results[algo]["ci"].append(m["sum_rate_ci"])
            results[algo]["std"].append(m["sum_rate_std"])
    return results


def sweep_K(agents: dict, obs_norms: dict, cfg: dict) -> dict:
    seeds = cfg["evaluation"]["seeds"]
    k_list = cfg["evaluation"]["k_sweep"]
    results = {algo: {"x": k_list, "mean": [], "ci": [], "std": []} for algo in agents}
    trained_K = cfg["env"]["num_users"]
    for K in k_list:
        if int(K) != int(trained_K):
            for algo in agents:
                results[algo]["mean"].append(float("nan"))
                results[algo]["ci"].append(0.0); results[algo]["std"].append(0.0)
            continue
        # K matches trained — keep trained K_r to preserve obs dims.
        env_cfg = copy.deepcopy(cfg)
        env_cfg["env"]["num_users"] = int(K)
        # Do NOT override num_users_reflection: preserves obs dim used during training.
        for algo, agent in agents.items():
            m = _eval_multi_seed(agent, algo, obs_norms[algo], env_cfg, seeds,
                                 ris_mode=("fixed" if algo == "FixedRIS" else "optimized"))
            results[algo]["mean"].append(m["sum_rate_mean"])
            results[algo]["ci"].append(m["sum_rate_ci"])
            results[algo]["std"].append(m["sum_rate_std"])
    return results


def qos_satisfaction(agents: dict, obs_norms: dict, cfg: dict) -> dict:
    seeds = cfg["evaluation"]["seeds"]
    out = {}
    for algo, agent in agents.items():
        m = _eval_multi_seed(agent, algo, obs_norms[algo], cfg, seeds,
                             ris_mode=("fixed" if algo == "FixedRIS" else "optimized"))
        out[algo] = {"mean": m["qos_mean"], "ci": m["qos_ci"], "std": m["qos_std"]}
    return out


def latency_benchmark(agents: dict, obs_norms: dict, cfg: dict, num_calls: int = 1000) -> dict:
    results = {}
    for algo, agent in agents.items():
        env = _make_env(cfg, seed=int(cfg["seed"]))
        obs, _ = env.reset(seed=int(cfg["seed"]))
        on = obs_norms[algo]
        is_maddpg = hasattr(agent, "select_actions")
        is_ppo = (not is_maddpg) and (algo == "PPO")

        if is_maddpg:
            per_agent_obs = env.per_agent_observations(None)
            if isinstance(on, list) and on is not None:
                per_agent_obs = [n(o, update=False) for n, o in zip(on, per_agent_obs)]
        else:
            obs_in = on(obs, update=False) if (on is not None and not isinstance(on, list)) else obs.astype(np.float32)

        def _act():
            if is_maddpg:
                return agent.select_actions(per_agent_obs, explore=False)
            if is_ppo:
                return agent.select_action(obs_in, explore=False)
            return agent.select_action(obs_in, explore=False)

        for _ in range(20):
            _act()
        t0 = time.perf_counter()
        for _ in range(num_calls):
            _act()
        dt = (time.perf_counter() - t0) * 1000.0 / num_calls
        results[algo] = dt
    return results
