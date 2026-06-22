"""STAR-RIS assisted RSMA SISO downlink Gymnasium environment.

System model
------------
- One BS (M = 1 antenna), one STAR-RIS with N elements (ES mode), K users
  partitioned into a reflection (R) region and a transmission (T) region.
- All channels: Rayleigh small-scale + log-distance large-scale path loss.
- Direct link h_dk: BS -> user k.
- BS->RIS:  G in C^{N}.
- RIS->user_k: g_k in C^{N}.
- STAR-RIS coefficients per element n:
     beta_n^r, beta_n^t in [0, 1] with beta_n^r + beta_n^t = 1  (ES, amplitude-squared sum)
     phi_n^r, phi_n^t in [0, 2pi)
  Reflection vector for user in R region:  diag(sqrt(beta^r) * exp(j phi^r))
  Transmission vector for user in T region: diag(sqrt(beta^t) * exp(j phi^t))
- Effective channel:  h_eff,k = h_dk + g_k^H * Phi_k * G   (scalar, M = 1).

RSMA SISO
---------
Transmit signal:
    x_BS = sqrt(P_c) s_c + sum_k sqrt(P_k) s_k,   total power = P_c + sum P_k <= P_max.
Receiver k:
    y_k = h_eff,k * x_BS + n_k,   n_k ~ CN(0, sigma^2)
SINR (common, decoded first by every user, treat private as noise):
    gamma_c,k = (|h_eff,k|^2 * P_c) / (|h_eff,k|^2 * sum_j P_j + sigma^2)
Common rate (must be decodable by ALL users):
    R_c = min_k log2(1 + gamma_c,k)
Each user gets a share c_k of the common rate, sum_k c_k = 1.
After SIC of common, user k decodes its private treating others' private as noise:
    gamma_k = (|h_eff,k|^2 P_k) / (|h_eff,k|^2 sum_{j != k} P_j + sigma^2)
Per-user rate: R_user_k = c_k * R_c + log2(1 + gamma_k)
Sum rate:     R_sum    = R_c + sum_k log2(1 + gamma_k)

Action mapping (per agent, networks output in [-1, 1])
------------------------------------------------------
Agent 0 (BS resource allocator):  size = (K + 1) + K
    - first (K+1): softmax-derived power weights for [common, p_1, ..., p_K] * P_max
    - next K: softmax-derived common-stream split c_k (sum to 1)
Agent 1 (STAR-RIS reflection):    size = 2N
    - first N: amplitude ratio u_n in (0,1) -> beta_n^r = u_n, beta_n^t = 1 - u_n
    - last N: reflection phases phi^r in [0, 2pi)
Agent 2 (STAR-RIS transmission):  size = N
    - transmission phases phi^t in [0, 2pi)

Observation per agent (concatenated for centralized critic via env.global_state()):
    - real/imag of h_eff,k (length 2K)  (computed using previous RIS coefficients)
    - previous power allocation (K + 1)
    - previous common split (K)
    - previous instantaneous reward (scalar)
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.metrics import dbm_to_watt, safe_log2, free_space_path_loss_db


@dataclass
class EnvSpec:
    obs_dims: list[int]
    act_dims: list[int]
    global_state_dim: int
    n_agents: int


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


class StarRisRsmaEnv(gym.Env):
    """Gymnasium-compatible multi-agent friendly env for STAR-RIS RSMA SISO."""
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, cfg: dict, seed: int | None = None,
                 ris_mode: str = "optimized"):
        """
        ris_mode:
          - "optimized": RIS phases/amplitudes set by agent action.
          - "fixed":     RIS phases all 0, amplitudes 50/50.
          - "random":    RIS phases & amplitudes drawn uniformly at random each step.
          - "none":      RIS contribution disabled (effective channel = h_dk only).
        """
        super().__init__()
        self.cfg = cfg
        self.ris_mode = ris_mode
        self.rng = np.random.default_rng(seed)

        # ----------- Topology -----------
        self.M = int(cfg["num_bs_antennas"])
        assert self.M == 1, "Current implementation is SISO (M=1)."
        self.K = int(cfg["num_users"])
        self.K_r = int(cfg["num_users_reflection"])
        assert 0 <= self.K_r <= self.K
        self.K_t = self.K - self.K_r
        self.N = int(cfg["num_ris_elements"])
        self.star_mode = cfg.get("star_mode", "ES")
        assert self.star_mode == "ES", "Only ES mode implemented."

        # ----------- Power / noise -----------
        self.p_max = float(dbm_to_watt(cfg["p_max_dbm"]))
        self.sigma2 = float(dbm_to_watt(cfg["noise_power_dbm"]))
        self.qos_min = float(cfg["qos_rate_min"])

        # ----------- Path-loss -----------
        self.pl_exp_d = float(cfg["path_loss_exp_direct"])
        self.pl_exp_br = float(cfg["path_loss_exp_bs_ris"])
        self.pl_exp_ru = float(cfg["path_loss_exp_ris_user"])
        self.ref_pl_db = float(cfg["ref_path_loss_db"])
        self.ref_d = float(cfg["ref_distance"])

        # ----------- Geometry -----------
        self.bs_pos = np.array(cfg["bs_position"], dtype=np.float64)
        self.ris_pos = np.array(cfg["ris_position"], dtype=np.float64)
        self.area_r = np.array(cfg["user_area_reflection"], dtype=np.float64)
        self.area_t = np.array(cfg["user_area_transmission"], dtype=np.float64)

        # ----------- Episode -----------
        self.max_steps = int(cfg["max_steps"])
        self.channel_block_steps = int(cfg.get("channel_block_steps", 1))

        # ----------- Reward shaping -----------
        self.r_alpha = float(cfg.get("reward_alpha", 1.0))
        self.r_beta = float(cfg.get("reward_beta", 2.0))           # static fallback (overridden by Lagrangian)
        self.r_gamma = float(cfg.get("reward_gamma", 0.05))
        self.r_scale = float(cfg.get("reward_scale", 0.1))
        self.r_clip = float(cfg.get("reward_clip", 50.0))
        self.eps = float(cfg.get("epsilon", 1e-12))
        # Quadratic QoS penalty with adaptive Lagrangian multiplier.
        self.qos_penalty_type = str(cfg.get("qos_penalty_type", "quadratic")).lower()
        self.qos_lambda = float(cfg.get("qos_lambda_init", 1.0))
        self.qos_lambda_min = float(cfg.get("qos_lambda_min", 0.1))
        self.qos_lambda_max = float(cfg.get("qos_lambda_max", 30.0))
        self.r_qos_bonus = float(cfg.get("reward_qos_bonus", 1.0))   # weight of per-user satisfaction bonus

        # Ablation: bypass agent's power and/or RIS actions.
        self.equal_power_mode = bool(cfg.get("equal_power_mode", False))

        # Physics-informed phase action: "absolute" or "residual" (analytical prior + RL residual).
        self.phase_action_mode = str(cfg.get("phase_action_mode", "absolute")).lower()
        self.phase_residual_scale = float(cfg.get("phase_residual_scale", 0.3))

        # ----------- User positions (fixed per env instance) -----------
        self.user_positions = self._sample_user_positions()

        # ----------- Pre-compute large-scale gains -----------
        self._compute_path_losses()

        # ----------- State variables -----------
        self._prev_power_weights = None   # length K+1, sums to <=1 of P_max
        self._prev_common_split = None    # length K, sums to 1
        self._prev_reward = 0.0
        self._step_count = 0
        self._h_eff = None                # last computed effective channel (complex K,)
        self._h_d = None                  # direct channels (K,)
        self._G = None                    # (N,)
        self._g = None                    # (K, N)
        self._beta_r = None               # (N,) in [0,1]
        self._phi_r = None                # (N,)
        self._phi_t = None                # (N,)

        # ----------- Spaces -----------
        self.n_agents = 3
        self.act_dims = [
            (self.K + 1) + self.K,   # BS: power softmax + common split softmax
            2 * self.N,              # RIS reflection: amplitudes + phases
            self.N,                  # RIS transmission: phases
        ]

        # Whether to expose raw per-element channel coefficients in the observation.
        # Critical for the agent to learn closed-form-style phase alignment.
        self.obs_include_channel = bool(cfg.get("obs_include_channel_state", True))
        # True CTDE: each MADDPG actor sees a LOCAL view containing only the channel
        # state relevant to its action. Single-agent algorithms (DDPG/TD3/PPO) still
        # see the full FLAT global observation.
        self.local_obs = bool(cfg.get("local_obs_for_maddpg", True))

        # Common per-agent prefix: Re/Im h_eff (2K) + prev_pw (K+1) + prev_cs (K) + prev_reward (1)
        base = 2 * self.K + (self.K + 1) + self.K + 1

        if self.obs_include_channel:
            # Channel state pieces for local views:
            #  BS power:    Re/Im h_d (2K)                  — needs all users' direct + h_eff
            #  RIS R agent: Re/Im h_d_R + G + g_R (2*K_r + 2N + 2*K_r*N)
            #  RIS T agent: Re/Im h_d_T + G + g_T (2*K_t + 2N + 2*K_t*N)
            if self.local_obs:
                self.obs_dims = [
                    base + 2 * self.K,                                          # BS power agent
                    base + 2 * self.K_r + 2 * self.N + 2 * self.K_r * self.N,   # RIS R agent
                    base + 2 * self.K_t + 2 * self.N + 2 * self.K_t * self.N,   # RIS T agent
                ]
            else:
                full_extra = 2 * self.K + 2 * self.N + 2 * self.K * self.N
                self.obs_dims = [base + full_extra] * self.n_agents
        else:
            self.obs_dims = [base] * self.n_agents

        # Global state used by centralized critic = concatenation of all per-agent obs.
        self.obs_dim_per_agent = max(self.obs_dims)  # legacy single-obs dim (max)
        self.global_state_dim = int(sum(self.obs_dims))

        # Flattened single-agent spaces (used by DDPG/TD3/PPO).
        # Single-agent algos use the FULL global observation (all channel state).
        self.act_dim_flat = int(sum(self.act_dims))
        # Single-agent obs dim = base + full channel state (regardless of local_obs flag).
        if self.obs_include_channel:
            single_extra = 2 * self.K + 2 * self.N + 2 * self.K * self.N
        else:
            single_extra = 0
        self.single_agent_obs_dim = base + single_extra
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.single_agent_obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_dim_flat,), dtype=np.float32,
        )

        # Per-agent action spaces (helpful for MADDPG bookkeeping).
        self.agent_action_spaces = [
            spaces.Box(low=-1.0, high=1.0, shape=(d,), dtype=np.float32) for d in self.act_dims
        ]
        self.agent_observation_spaces = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32) for d in self.obs_dims
        ]

    # ------------------------------------------------------------------ utils
    def spec(self) -> EnvSpec:
        return EnvSpec(
            obs_dims=list(self.obs_dims),
            act_dims=list(self.act_dims),
            global_state_dim=self.global_state_dim,
            n_agents=self.n_agents,
        )

    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    # ---------------------------------------------------------- geometry
    def _sample_user_positions(self) -> np.ndarray:
        positions = np.zeros((self.K, 3), dtype=np.float64)
        # Reflection-region users.
        for k in range(self.K_r):
            positions[k] = self._sample_in_box(self.area_r)
        # Transmission-region users.
        for k in range(self.K_r, self.K):
            positions[k] = self._sample_in_box(self.area_t)
        return positions

    def _sample_in_box(self, box: np.ndarray) -> np.ndarray:
        # box: 3x2 [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        p = np.array([
            self.rng.uniform(box[0, 0], box[0, 1]),
            self.rng.uniform(box[1, 0], box[1, 1]),
            self.rng.uniform(box[2, 0], box[2, 1]),
        ])
        return p

    def _compute_path_losses(self):
        # Distances (m).
        d_bs_user = np.linalg.norm(self.user_positions - self.bs_pos[None, :], axis=1)
        d_bs_ris = float(np.linalg.norm(self.ris_pos - self.bs_pos))
        d_ris_user = np.linalg.norm(self.user_positions - self.ris_pos[None, :], axis=1)

        pl_d_db = free_space_path_loss_db(d_bs_user, self.ref_pl_db, self.ref_d, self.pl_exp_d)
        pl_br_db = free_space_path_loss_db(np.array([d_bs_ris]), self.ref_pl_db, self.ref_d, self.pl_exp_br)[0]
        pl_ru_db = free_space_path_loss_db(d_ris_user, self.ref_pl_db, self.ref_d, self.pl_exp_ru)

        # Optional NLoS / blockage loss for the T-region (canonical STAR-RIS scenario:
        # the BS is on the reflection side, T users are behind an obstacle).
        block_db = float(self.cfg.get("direct_block_loss_db", 0.0))
        if bool(self.cfg.get("direct_block_T", False)) and self.K_t > 0 and block_db > 0:
            pl_d_db[self.K_r:] = pl_d_db[self.K_r:] + block_db
        # Optional NLoS loss for R-region (rare; usually 0).
        block_r_db = float(self.cfg.get("direct_block_R_loss_db", 0.0))
        if block_r_db > 0 and self.K_r > 0:
            pl_d_db[: self.K_r] = pl_d_db[: self.K_r] + block_r_db

        # Convert dB -> linear amplitude gain factor a = 10^(-PL/20).
        self.alpha_d = 10.0 ** (-pl_d_db / 20.0)           # (K,)
        self.alpha_br = 10.0 ** (-pl_br_db / 20.0)         # scalar
        self.alpha_ru = 10.0 ** (-pl_ru_db / 20.0)         # (K,)

    # ---------------------------------------------------------- channels
    def _sample_channels(self):
        """Sample fresh Rayleigh fading channels (mean-zero, unit variance per dimension)."""
        # CN(0, 1) realizations: real and imag each ~ N(0, 0.5).
        def cn(*shape):
            return (self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)) / math.sqrt(2.0)

        # Direct BS->user (M=1, so scalar per user).
        h_d_small = cn(self.K)                             # (K,)
        # BS->RIS (M=1).
        G_small = cn(self.N)                               # (N,)
        # RIS->user_k.
        g_small = cn(self.K, self.N)                       # (K, N)

        # Apply large-scale gains.
        self._h_d = (self.alpha_d * h_d_small).astype(np.complex128)
        self._G = (self.alpha_br * G_small).astype(np.complex128)
        self._g = (self.alpha_ru[:, None] * g_small).astype(np.complex128)

    # ---------------------------------------------------------- analytical phases
    def _analytical_phases(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form constructive-alignment phases for the weakest R / T user."""
        if self.K_r > 0:
            k_R = int(np.argmin(np.abs(self._h_d[: self.K_r])))
            phi_r = np.mod(
                np.angle(self._h_d[k_R]) - np.angle(np.conj(self._g[k_R]) * self._G),
                2 * math.pi,
            )
        else:
            phi_r = np.zeros(self.N)
        if self.K_t > 0:
            k_T = self.K_r + int(np.argmin(np.abs(self._h_d[self.K_r:])))
            phi_t = np.mod(
                np.angle(self._h_d[k_T]) - np.angle(np.conj(self._g[k_T]) * self._G),
                2 * math.pi,
            )
        else:
            phi_t = np.zeros(self.N)
        return phi_r, phi_t

    # ---------------------------------------------------------- BCD baseline
    def _bcd_optimize(self, n_iter: int = 20) -> dict:
        """Block coordinate descent baseline (optimization upper bound).

        Alternates between three blocks (joint over the current channel realization):
          B1 — phases: closed-form max-min single-user alignment (reuse _analytical_phases)
          B2 — beta_r (amplitude split): 1-D grid search over discrete levels
          B3 — power allocation (P_c fraction): 1-D grid search

        Returns the decoded dict (P_c, P_k, common_split, beta_r, phi_r, phi_t) just like
        the RL action decoder. Used as `ris_mode == "bcd"` for ablation.

        Complexity per call: n_iter × (|beta_grid| + |Pc_grid|) RSMA evaluations.
        With n_iter=20, |beta|=5, |Pc|=6 → 220 evaluations of `_rsma_rates`.
        """
        K = self.K
        N = self.N
        # ----- Initialise -----
        beta_r = 0.5 * np.ones(N)
        P_c = float(self.p_max * 0.5)
        P_k = np.full(K, (self.p_max - P_c) / max(K, 1), dtype=np.float64)
        common_split = np.ones(K, dtype=np.float64) / K
        phi_r = np.zeros(N); phi_t = np.zeros(N)

        beta_grid = np.array([0.2, 0.3, 0.5, 0.7, 0.8])
        pc_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.85, 0.95])

        for _ in range(int(n_iter)):
            # B1: closed-form constructive-alignment phases for the bottleneck user.
            phi_r, phi_t = self._analytical_phases()

            # B2: grid-search beta_r (uniform across elements — full per-element search is 2^N).
            best_beta = float(beta_r[0]); best_sr = -np.inf
            for b in beta_grid:
                b_arr = float(b) * np.ones(N)
                h = self._effective_channels(b_arr, phi_r, phi_t)
                rs = self._rsma_rates(h, P_c, P_k, common_split)
                if rs["sum_rate"] > best_sr:
                    best_sr = rs["sum_rate"]; best_beta = float(b)
            beta_r = best_beta * np.ones(N)

            # B3: grid-search P_c/Pmax fraction (private split equal).
            h_eff = self._effective_channels(beta_r, phi_r, phi_t)
            best_pc = P_c; best_sr = -np.inf
            for f in pc_grid:
                Pc_t = float(self.p_max * f)
                Pk_t = np.full(K, (self.p_max - Pc_t) / max(K, 1), dtype=np.float64)
                rs = self._rsma_rates(h_eff, Pc_t, Pk_t, common_split)
                if rs["sum_rate"] > best_sr:
                    best_sr = rs["sum_rate"]; best_pc = Pc_t
            P_c = best_pc
            P_k = np.full(K, (self.p_max - P_c) / max(K, 1), dtype=np.float64)

        # Recompute power weights vector so info dict / observation stay consistent.
        powers = np.concatenate([[P_c], P_k])
        power_weights = powers / max(self.p_max, 1e-12)
        return {
            "P_c": P_c, "P_k": P_k,
            "power_weights": power_weights, "common_split": common_split,
            "beta_r": beta_r, "phi_r": phi_r, "phi_t": phi_t,
        }

    # ---------------------------------------------------------- action decoding
    def _decode_action(self, action_list: list[np.ndarray]):
        """Map normalized [-1,1] actions -> physical decision variables."""
        a_bs, a_ris_r, a_ris_t = action_list

        # ---- BS agent ----
        a_bs = np.clip(a_bs, -1.0, 1.0)
        pw_logits = a_bs[: self.K + 1]
        cs_logits = a_bs[self.K + 1:]
        # Scale logits before softmax for sharper output range.
        power_weights = _softmax(2.0 * pw_logits)           # sums to 1 over [common, K privates]
        common_split = _softmax(2.0 * cs_logits)            # sums to 1 over K users
        powers = power_weights * self.p_max                 # (K+1,)
        P_c = float(powers[0])
        P_k = powers[1:].astype(np.float64)                 # (K,)

        # ---- STAR-RIS reflection agent ----
        a_ris_r = np.clip(a_ris_r, -1.0, 1.0)
        beta_logits = a_ris_r[: self.N]
        phi_r_raw = a_ris_r[self.N:]
        beta_r = 0.5 * (beta_logits + 1.0)                  # in [0,1]
        beta_r = np.clip(beta_r, 1e-4, 1.0 - 1e-4)          # avoid degenerate 0/1

        # ---- STAR-RIS transmission agent ----
        a_ris_t = np.clip(a_ris_t, -1.0, 1.0)

        # Physics-informed phase action mapping:
        # - "absolute":  phi_n = pi * (action + 1)                   ∈ [0, 2π]
        # - "residual":  phi_n = analytical_prior + scale*pi*action  ∈ analytical ± scale·π
        #
        # The residual mode biases the policy toward closed-form alignment while letting RL
        # learn corrections (useful when analytical is suboptimal for multi-user RIS).
        if self.phase_action_mode == "residual" and self.phase_residual_scale > 0:
            prior_phi_r, prior_phi_t = self._analytical_phases()
            phi_r = prior_phi_r + self.phase_residual_scale * math.pi * phi_r_raw
            phi_t = prior_phi_t + self.phase_residual_scale * math.pi * a_ris_t
            phi_r = np.mod(phi_r, 2 * math.pi)
            phi_t = np.mod(phi_t, 2 * math.pi)
        else:
            phi_r = math.pi * (phi_r_raw + 1.0)             # in [0, 2π]
            phi_t = math.pi * (a_ris_t + 1.0)               # in [0, 2π]

        # ----- Equal-power ablation override -----
        if self.equal_power_mode:
            uniform = np.ones(self.K + 1, dtype=np.float64) / (self.K + 1)
            power_weights = uniform
            common_split = np.ones(self.K, dtype=np.float64) / self.K
            P_c = float(self.p_max / (self.K + 1))
            P_k = np.full(self.K, self.p_max / (self.K + 1), dtype=np.float64)

        # ----- RIS ablation overrides -----
        if self.ris_mode == "bcd":
            # BCD baseline (joint phase+power+amplitude optimization). Overrides the
            # agent's action entirely and returns the BCD-optimal decision.
            return self._bcd_optimize()
        if self.ris_mode == "fixed":
            beta_r = 0.5 * np.ones(self.N)
            phi_r = np.zeros(self.N)
            phi_t = np.zeros(self.N)
        elif self.ris_mode == "random":
            beta_r = self.rng.uniform(0.1, 0.9, size=self.N)
            phi_r = self.rng.uniform(0.0, 2 * math.pi, size=self.N)
            phi_t = self.rng.uniform(0.0, 2 * math.pi, size=self.N)
        elif self.ris_mode == "none":
            beta_r = 0.5 * np.ones(self.N)
            phi_r = np.zeros(self.N)
            phi_t = np.zeros(self.N)
        elif self.ris_mode == "analytical":
            # Closed-form constructive-alignment optimum (upper bound).
            beta_r = 0.5 * np.ones(self.N)
            phi_r, phi_t = self._analytical_phases()

        return {
            "P_c": P_c, "P_k": P_k,
            "power_weights": power_weights, "common_split": common_split,
            "beta_r": beta_r, "phi_r": phi_r, "phi_t": phi_t,
        }

    # ---------------------------------------------------------- RSMA
    def _effective_channels(self, beta_r: np.ndarray, phi_r: np.ndarray, phi_t: np.ndarray) -> np.ndarray:
        # Each element n contributes a_n * exp(j*phi_n) to the cascaded path,
        # with a_n = sqrt(beta_r_n)  for R users, sqrt(1 - beta_r_n) for T users.
        beta_t = np.clip(1.0 - beta_r, 1e-4, 1.0 - 1e-4)
        amp_r = np.sqrt(beta_r)
        amp_t = np.sqrt(beta_t)
        coeff_r = amp_r * np.exp(1j * phi_r)               # (N,)
        coeff_t = amp_t * np.exp(1j * phi_t)               # (N,)

        h_eff = np.zeros(self.K, dtype=np.complex128)
        h_ris = np.zeros(self.K, dtype=np.complex128)
        for k in range(self.K):
            if self.ris_mode == "none":
                cascaded = 0.0 + 0.0j
            else:
                coeff = coeff_r if k < self.K_r else coeff_t
                cascaded = np.sum(np.conj(self._g[k]) * coeff * self._G)
            h_ris[k] = cascaded
            h_eff[k] = self._h_d[k] + cascaded
        # Cache the cascaded part for diagnostics.
        self._h_ris = h_ris
        return h_eff

    @staticmethod
    def _phase_entropy(phi: np.ndarray, n_bins: int = 16) -> float:
        """Shannon entropy of phase distribution (nats), 0..log(n_bins)."""
        if phi.size == 0:
            return 0.0
        hist, _ = np.histogram(np.mod(phi, 2 * math.pi), bins=n_bins, range=(0.0, 2 * math.pi))
        p = hist.astype(np.float64)
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    def _rsma_rates(self, h_eff: np.ndarray, P_c: float, P_k: np.ndarray, common_split: np.ndarray):
        # |h_eff_k|^2
        h2 = (h_eff.real ** 2 + h_eff.imag ** 2).astype(np.float64)
        total_priv = float(P_k.sum())
        # Common SINR per user.
        denom_c = h2 * total_priv + self.sigma2 + self.eps  # interference + noise (treat ALL privates as noise pre-SIC of common)
        sinr_c = (h2 * P_c) / denom_c
        # Common rate is the minimum (decodable by all).
        rate_c = float(np.min(safe_log2(1.0 + sinr_c, self.eps)))

        # Private SINR per user (after SIC of common).
        rates_p = np.zeros(self.K, dtype=np.float64)
        for k in range(self.K):
            interf = float(P_k.sum() - P_k[k])
            denom_k = h2[k] * interf + self.sigma2 + self.eps
            sinr_k = (h2[k] * P_k[k]) / denom_k
            rates_p[k] = float(safe_log2(np.array([1.0 + sinr_k]), self.eps)[0])

        # Per-user rate = c_k * R_c + R_priv_k.
        per_user = common_split * rate_c + rates_p
        sum_rate = float(rate_c + rates_p.sum())            # equivalent to per_user.sum() since sum(common_split)=1
        return {
            "rate_c": rate_c,
            "rate_p": rates_p,
            "per_user": per_user,
            "sum_rate": sum_rate,
            "h2": h2,
            "sinr_c": sinr_c,
        }

    # ---------------------------------------------------------- observation
    def _build_observation(self) -> np.ndarray:
        """Build FULL single-agent observation (used by DDPG/TD3/PPO and as a base)."""
        h = self._h_eff if self._h_eff is not None else np.zeros(self.K, dtype=np.complex128)
        scale = float(self.alpha_d[: self.K_r].mean()) + 1e-12
        re = (h.real / scale).astype(np.float32)
        im = (h.imag / scale).astype(np.float32)
        pw = self._prev_power_weights if self._prev_power_weights is not None \
            else np.ones(self.K + 1, dtype=np.float32) / (self.K + 1)
        cs = self._prev_common_split if self._prev_common_split is not None \
            else np.ones(self.K, dtype=np.float32) / self.K
        prev_r = np.array([self._prev_reward], dtype=np.float32)
        parts = [re, im, pw.astype(np.float32), cs.astype(np.float32), prev_r]

        if self.obs_include_channel and self._h_d is not None:
            hd_re = (self._h_d.real / (self.alpha_d + 1e-30)).astype(np.float32)
            hd_im = (self._h_d.imag / (self.alpha_d + 1e-30)).astype(np.float32)
            G_re = (self._G.real / (self.alpha_br + 1e-30)).astype(np.float32)
            G_im = (self._G.imag / (self.alpha_br + 1e-30)).astype(np.float32)
            g_re = (self._g.real / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32).reshape(-1)
            g_im = (self._g.imag / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32).reshape(-1)
            parts.extend([hd_re, hd_im, G_re, G_im, g_re, g_im])

        obs = np.concatenate(parts).astype(np.float32)
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    def _build_per_agent_observations(self) -> list[np.ndarray]:
        """Per-agent LOCAL observations (true CTDE).
        BS power: h_eff + h_d (all users) + bookkeeping.
        RIS-R:    h_eff + h_d_R + G + g_R + bookkeeping.
        RIS-T:    h_eff + h_d_T + G + g_T + bookkeeping.
        """
        h = self._h_eff if self._h_eff is not None else np.zeros(self.K, dtype=np.complex128)
        scale = float(self.alpha_d[: self.K_r].mean()) + 1e-12
        re = (h.real / scale).astype(np.float32)
        im = (h.imag / scale).astype(np.float32)
        pw = self._prev_power_weights if self._prev_power_weights is not None \
            else np.ones(self.K + 1, dtype=np.float32) / (self.K + 1)
        cs = self._prev_common_split if self._prev_common_split is not None \
            else np.ones(self.K, dtype=np.float32) / self.K
        prev_r = np.array([self._prev_reward], dtype=np.float32)
        base = np.concatenate([re, im, pw.astype(np.float32), cs.astype(np.float32), prev_r])

        if not self.obs_include_channel or self._h_d is None:
            obs_all = [base.copy() for _ in range(self.n_agents)]
        else:
            hd_re = (self._h_d.real / (self.alpha_d + 1e-30)).astype(np.float32)
            hd_im = (self._h_d.imag / (self.alpha_d + 1e-30)).astype(np.float32)
            G_re = (self._G.real / (self.alpha_br + 1e-30)).astype(np.float32)
            G_im = (self._G.imag / (self.alpha_br + 1e-30)).astype(np.float32)
            # Per-user g normalized.
            g_re = (self._g.real / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32)
            g_im = (self._g.imag / (self.alpha_ru[:, None] + 1e-30)).astype(np.float32)

            if self.local_obs:
                # BS power agent: needs all users' direct channel + h_eff.
                bs_obs = np.concatenate([base, hd_re, hd_im])
                # RIS-R agent: R-region direct + G + g_R.
                if self.K_r > 0:
                    ris_r_obs = np.concatenate([
                        base, hd_re[: self.K_r], hd_im[: self.K_r],
                        G_re, G_im,
                        g_re[: self.K_r].reshape(-1), g_im[: self.K_r].reshape(-1),
                    ])
                else:
                    ris_r_obs = np.concatenate([base, G_re, G_im])
                # RIS-T agent: T-region direct + G + g_T.
                if self.K_t > 0:
                    ris_t_obs = np.concatenate([
                        base, hd_re[self.K_r:], hd_im[self.K_r:],
                        G_re, G_im,
                        g_re[self.K_r:].reshape(-1), g_im[self.K_r:].reshape(-1),
                    ])
                else:
                    ris_t_obs = np.concatenate([base, G_re, G_im])
                obs_all = [bs_obs, ris_r_obs, ris_t_obs]
            else:
                full = np.concatenate([base, hd_re, hd_im, G_re, G_im,
                                       g_re.reshape(-1), g_im.reshape(-1)])
                obs_all = [full.copy() for _ in range(self.n_agents)]

        out = []
        for o in obs_all:
            o = o.astype(np.float32)
            if not np.all(np.isfinite(o)):
                o = np.nan_to_num(o, nan=0.0, posinf=10.0, neginf=-10.0)
            out.append(o)
        return out

    # ---------------------------------------------------------- Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._prev_power_weights = np.ones(self.K + 1, dtype=np.float32) / (self.K + 1)
        self._prev_common_split = np.ones(self.K, dtype=np.float32) / self.K
        self._prev_reward = 0.0
        # Initial RIS state (identity-like).
        self._beta_r = 0.5 * np.ones(self.N)
        self._phi_r = np.zeros(self.N)
        self._phi_t = np.zeros(self.N)
        self._sample_channels()
        self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)
        obs = self._build_observation()
        return obs, {}

    def step(self, action):
        action_list = self._split_action(action)
        # Validate action lengths.
        for a, d in zip(action_list, self.act_dims):
            assert a.shape[0] == d, f"Action shape mismatch: got {a.shape[0]}, expected {d}"

        # Refresh small-scale fading FIRST so that:
        #  (i) analytical-mode decode uses the current realization (not the previous one), and
        #  (ii) the agent's action observed at this step is applied to the same channel that
        #       was visible in the observation.
        if (self._step_count % max(1, self.channel_block_steps)) == 0 and self._step_count > 0:
            self._sample_channels()

        decoded = self._decode_action(action_list)
        self._beta_r = decoded["beta_r"]
        self._phi_r = decoded["phi_r"]
        self._phi_t = decoded["phi_t"]

        self._h_eff = self._effective_channels(self._beta_r, self._phi_r, self._phi_t)
        rsma = self._rsma_rates(self._h_eff, decoded["P_c"], decoded["P_k"], decoded["common_split"])

        # ------------ Reward shaping ------------
        sum_rate = rsma["sum_rate"]
        per_user = rsma["per_user"]
        deficit = np.maximum(self.qos_min - per_user, 0.0)
        qos_viol_l1 = float(deficit.sum())
        qos_viol_l2 = float((deficit ** 2).sum())
        # Per-user binary satisfaction (gives partial credit so the policy gradient
        # has a clear positive signal as soon as any user crosses the QoS threshold).
        per_user_sat = (per_user >= self.qos_min).astype(np.float64)
        frac_sat = float(per_user_sat.mean())
        total_power = float(decoded["P_c"] + decoded["P_k"].sum())
        power_excess = max(0.0, total_power - self.p_max) / max(self.p_max, 1e-12)

        normalized_sr = sum_rate / max(self.K * 5.0, 1.0)
        if self.qos_penalty_type == "quadratic":
            qos_term = self.qos_lambda * qos_viol_l2
        elif self.qos_penalty_type == "linear":
            qos_term = self.qos_lambda * qos_viol_l1
        else:
            margin = np.maximum(per_user - self.qos_min, 1e-3)
            qos_term = -self.qos_lambda * 0.05 * float(np.log(margin).sum())

        r_sr = self.r_alpha * normalized_sr
        r_qos = -qos_term
        r_pwr = -self.r_gamma * power_excess
        r_bonus = self.r_qos_bonus * frac_sat                 # NEW: positive shaping signal
        reward_raw = r_sr + r_qos + r_pwr + r_bonus
        reward = float(np.clip(self.r_scale * reward_raw, -self.r_clip, self.r_clip))
        if not math.isfinite(reward):
            reward = -self.r_clip

        # Bookkeeping for observation.
        self._prev_power_weights = decoded["power_weights"].astype(np.float32)
        self._prev_common_split = decoded["common_split"].astype(np.float32)
        self._prev_reward = reward

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        # ------------ Diagnostics (per-step) ------------
        h2 = rsma["h2"]
        phi_eff = self._phi_r if self.K_r > 0 else self._phi_t
        info = {
            "sum_rate": sum_rate,
            "rate_common": rsma["rate_c"],
            "per_user_rate": per_user.copy(),
            "rate_private": rsma["rate_p"].copy(),
            "qos_violation_l1": qos_viol_l1,
            "qos_violation_l2": qos_viol_l2,
            "qos_satisfied": bool(qos_viol_l1 < 1e-6),
            "qos_lambda": float(self.qos_lambda),
            "total_power_W": total_power,
            "power_excess_norm": power_excess,
            "reward_sr": float(self.r_scale * r_sr),
            "reward_qos": float(self.r_scale * r_qos),
            "reward_pwr": float(self.r_scale * r_pwr),
            "reward_bonus": float(self.r_scale * r_bonus),
            "per_user_satisfied_frac": frac_sat,
            "h_eff_abs_mean": float(np.mean(np.abs(self._h_eff))),
            "h_eff_abs_R": float(np.mean(np.abs(self._h_eff[: self.K_r]))) if self.K_r > 0 else 0.0,
            "h_eff_abs_T": float(np.mean(np.abs(self._h_eff[self.K_r:]))) if self.K_t > 0 else 0.0,
            "h_direct_abs_T": float(np.mean(np.abs(self._h_d[self.K_r:]))) if self.K_t > 0 else 0.0,
            "h_ris_abs_T": float(np.mean(np.abs(self._h_ris[self.K_r:]))) if (self.K_t > 0 and getattr(self, "_h_ris", None) is not None) else 0.0,
            "h_direct_abs_R": float(np.mean(np.abs(self._h_d[: self.K_r]))) if self.K_r > 0 else 0.0,
            "h_ris_abs_R": float(np.mean(np.abs(self._h_ris[: self.K_r]))) if (self.K_r > 0 and getattr(self, "_h_ris", None) is not None) else 0.0,
            "ris_to_direct_ratio_T": float(np.mean(np.abs(self._h_ris[self.K_r:])) / max(np.mean(np.abs(self._h_d[self.K_r:])), 1e-30)) if (self.K_t > 0 and getattr(self, "_h_ris", None) is not None) else 0.0,
            "h2_mean": float(np.mean(h2)),
            "phase_entropy_R": self._phase_entropy(self._phi_r),
            "phase_entropy_T": self._phase_entropy(self._phi_t),
            "phase_var_R": float(np.var(self._phi_r)) if self._phi_r.size > 0 else 0.0,
            "phase_var_T": float(np.var(self._phi_t)) if self._phi_t.size > 0 else 0.0,
            "beta_r_mean": float(np.mean(self._beta_r)),
            "common_power_frac": float(decoded["power_weights"][0]),
        }
        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------- adaptive QoS λ
    def set_qos_lambda(self, value: float) -> None:
        """Set QoS Lagrangian multiplier, clamping to [qos_lambda_min, qos_lambda_max].

        R9 reviewer fix: when the requested value is clamped, log a warning so the
        train/eval reward-scale mismatch is visible rather than silent.
        """
        clamped = float(np.clip(value, self.qos_lambda_min, self.qos_lambda_max))
        if abs(float(value) - clamped) > 1e-9:
            import warnings
            warnings.warn(
                f"[StarRisRsmaEnv] qos_lambda clamped from {float(value):.4f} to {clamped:.4f} "
                f"(bounds=[{self.qos_lambda_min}, {self.qos_lambda_max}]). "
                "Train/eval reward scales may differ — verify intent.",
                RuntimeWarning, stacklevel=2,
            )
        self.qos_lambda = clamped

    def render(self):
        h = self._h_eff if self._h_eff is not None else np.zeros(self.K, dtype=complex)
        print(f"step={self._step_count}  sum_rate?  "
              f"|h_eff|^2={np.abs(h)**2}  prev_reward={self._prev_reward:.3f}")

    # ------------------------------------------------------------------ helpers
    def _split_action(self, action) -> list[np.ndarray]:
        """Accepts either a flat np.ndarray (single-agent algorithms) or a list of arrays (MADDPG)."""
        if isinstance(action, (list, tuple)):
            return [np.asarray(a, dtype=np.float32).reshape(-1) for a in action]
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        out = []
        idx = 0
        for d in self.act_dims:
            out.append(arr[idx: idx + d])
            idx += d
        assert idx == arr.size, f"Flat action length {arr.size} != expected {idx}"
        return out

    def per_agent_observations(self, obs: np.ndarray) -> list[np.ndarray]:
        """Per-agent local observations (true CTDE) — ignores `obs` argument and
        rebuilds from current env state. Kept signature-compatible with prior code."""
        return self._build_per_agent_observations()
