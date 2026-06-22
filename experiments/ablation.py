"""Expanded ablation: 7 cells covering learned/analytical/fixed/random RIS x
learned/equal-power BS. Uses ONE trained MADDPG agent + multi-seed env evaluation."""
from __future__ import annotations
import numpy as np
from utils.metrics import confidence_interval
from experiments.train import evaluate_agent


# (label, ris_mode, equal_power)
# Notes:
#  * "BCD" — Block Coordinate Descent joint phase+power+amplitude optimization.
#    Iterates (closed-form phase) ↔ (grid-search amplitude) ↔ (grid-search power).
#    Serves as the true optimization-based upper bound the RL methods approach.
#  * "MaxMinAlignedRIS" — single-shot closed-form max-min single-user phase alignment.
#    Sub-optimal relative to BCD (no power/amplitude joint update).
ABLATION_CELLS = [
    ("Learned",               "optimized",  False),
    ("BCD",                   "bcd",        False),
    ("MaxMinAlignedRIS",      "analytical", False),
    ("FixedRIS",              "fixed",      False),
    ("RandomRIS",             "random",     False),
    ("NoRIS",                 "none",       False),
    ("EqualPower+Learned",    "optimized",  True),
    ("EqualPower+Fixed",      "fixed",      True),
]


def ablation_study(maddpg_agent, obs_norm, cfg: dict,
                   qos_lambda: float | None = None) -> dict:
    """Run the 7-cell ablation. `qos_lambda` (R4 reviewer fix) propagates the trained
    QoS multiplier so ablation cells use the SAME reward scale as the main algorithm
    comparison; otherwise eval would silently use qos_lambda_init."""
    seeds = cfg["evaluation"]["seeds"]
    n_eps = cfg["evaluation"]["num_episodes"]
    out = {}
    for label, ris_mode, equal_power in ABLATION_CELLS:
        srs, qoss, rc, htabs, pent, cfrac = [], [], [], [], [], []
        for s in seeds:
            m = evaluate_agent(env_cfg=cfg, agent=maddpg_agent, kind="maddpg",
                               obs_norm=obs_norm, episodes=n_eps, seed=int(s),
                               ris_mode=ris_mode, equal_power=equal_power,
                               qos_lambda=qos_lambda)
            srs.append(m["sum_rate_mean"]); qoss.append(m["qos_prob"])
            rc.append(m["rate_common_mean"]); htabs.append(m["h_eff_abs_T_mean"])
            pent.append(m["phase_entropy_T_mean"]); cfrac.append(m["common_power_frac_mean"])
        sr_m, sr_ci, sr_std = confidence_interval(np.array(srs))
        q_m, q_ci, q_std = confidence_interval(np.array(qoss))
        out[label] = {
            "sum_rate_mean": sr_m, "sum_rate_ci": sr_ci, "sum_rate_std": sr_std,
            "qos_mean": q_m, "qos_ci": q_ci, "qos_std": q_std,
            "rate_common": float(np.mean(rc)),
            "h_eff_abs_T": float(np.mean(htabs)),
            "phase_entropy_T": float(np.mean(pent)),
            "common_power_frac": float(np.mean(cfrac)),
        }
    return out
