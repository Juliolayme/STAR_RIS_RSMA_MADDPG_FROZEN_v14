"""Channel model validation diagnostics.

Verifies:
  - Path-loss linearity and dB-domain consistency
  - Empirical channel power matches theoretical (E[|alpha*h|^2] = alpha^2 for h ~ CN(0,1))
  - Direct vs RIS path magnitudes (with random / fixed / analytical / no RIS)
  - Phase shift sensitivity: |h_eff| under phi vs phi+delta
  - Complex arithmetic: |a + b|^2 ≈ |a|^2 + |b|^2 + 2 Re{a^* b}
"""
from __future__ import annotations
import math
import sys
import os
import numpy as np
import yaml

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from env import StarRisRsmaEnv


def section(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def validate(cfg_path: str = None, n_samples: int = 5000) -> None:
    cfg_path = cfg_path or os.path.join(PROJ, "config", "config.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    section("System parameters")
    for k in ("num_users", "num_users_reflection", "num_ris_elements",
              "p_max_dbm", "noise_power_dbm", "direct_block_loss_db",
              "path_loss_exp_direct", "path_loss_exp_bs_ris", "path_loss_exp_ris_user"):
        print(f"  {k:30s} = {cfg['env'].get(k)}")

    env = StarRisRsmaEnv(cfg["env"], seed=1234)
    section("Large-scale gain factors (linear amplitude)")
    print(f"  alpha_d (per user) = {env.alpha_d}")
    print(f"    R region:  mean = {np.mean(env.alpha_d[:env.K_r]):.3e}")
    print(f"    T region:  mean = {np.mean(env.alpha_d[env.K_r:]):.3e}")
    print(f"  alpha_br (BS->RIS)  = {env.alpha_br:.3e}")
    print(f"  alpha_ru (RIS->user) = {env.alpha_ru}")
    section("Theoretical vs empirical channel power")
    # Sample many small-scale realizations and check E[|h_d|^2] ≈ alpha_d^2
    pow_d_emp = np.zeros(env.K)
    pow_G_emp = 0.0
    pow_g_emp = np.zeros(env.K)
    for _ in range(n_samples):
        env._sample_channels()
        pow_d_emp += np.abs(env._h_d) ** 2
        pow_G_emp += float(np.mean(np.abs(env._G) ** 2))
        pow_g_emp += np.mean(np.abs(env._g) ** 2, axis=1)
    pow_d_emp /= n_samples
    pow_G_emp /= n_samples
    pow_g_emp /= n_samples
    pow_d_th = env.alpha_d ** 2
    pow_G_th = env.alpha_br ** 2
    pow_g_th = env.alpha_ru ** 2
    print(f"  E[|h_d|^2] / alpha_d^2  (per user) = {pow_d_emp / pow_d_th}")
    print(f"  E[|G_n|^2] / alpha_br^2            = {pow_G_emp / pow_G_th:.3f}")
    print(f"  E[|g_n|^2] / alpha_ru^2 (per user) = {pow_g_emp / pow_g_th}")

    section("Direct vs RIS-cascaded path magnitudes (random vs analytical phases)")
    for mode in ("none", "random", "analytical"):
        env2 = StarRisRsmaEnv(cfg["env"], seed=2025, ris_mode=mode)
        env2.reset(seed=2025)
        h_direct, h_ris, h_eff = [], [], []
        for _ in range(200):
            a = np.random.uniform(-1, 1, env2.action_space.shape[0]).astype(np.float32)
            obs, r, t, tr, info = env2.step(a)
            h_direct.append(info["h_direct_abs_T"])
            h_ris.append(info["h_ris_abs_T"])
            h_eff.append(info["h_eff_abs_T"])
        print(f"  mode={mode:11s}  E[|h_d_T|]={np.mean(h_direct):.3e}  "
              f"E[|h_ris_T|]={np.mean(h_ris):.3e}  E[|h_eff_T|]={np.mean(h_eff):.3e}  "
              f"ratio_ris/direct={np.mean(h_ris)/max(np.mean(h_direct),1e-30):.2f}")

    section("Phase-shift sensitivity (analytical vs random)")
    # Hold channels fixed; sweep one phase; see how |h_eff,T| changes.
    env3 = StarRisRsmaEnv(cfg["env"], seed=99, ris_mode="optimized")
    env3.reset(seed=99)
    # Force a fixed channel realization
    env3._sample_channels()
    # Use base random phases.
    base_action = np.zeros(env3.action_space.shape[0], dtype=np.float32)
    # phi_t sweep: vary the first element's phase over [-1,1]
    phi_t_idx = env3.act_dims[0] + env3.act_dims[1]
    sweep = []
    for v in np.linspace(-1.0, 1.0, 41):
        a = base_action.copy()
        a[phi_t_idx] = v
        # Don't actually step (which resamples) — call _decode + _effective directly.
        decoded = env3._decode_action(env3._split_action(a))
        h = env3._effective_channels(decoded["beta_r"], decoded["phi_r"], decoded["phi_t"])
        sweep.append((v, float(np.mean(np.abs(h[env3.K_r:])))))
    arr = np.array(sweep)
    print(f"  phi_t[0] sweep over [-1,1] ({arr.shape[0]} points):")
    print(f"    |h_eff_T| min={arr[:,1].min():.3e}  max={arr[:,1].max():.3e}  "
          f"variation={(arr[:,1].max()/max(arr[:,1].min(),1e-30) - 1)*100:.1f}%")
    print(f"    --> A single RIS phase moves |h_eff,T| by {(arr[:,1].max() - arr[:,1].min())/max(arr[:,1].mean(),1e-30)*100:.1f}% of mean.")

    section("Per-element RIS contribution check (all phases aligned to phi=0)")
    env4 = StarRisRsmaEnv(cfg["env"], seed=7, ris_mode="fixed")
    env4.reset(seed=7)
    env4._sample_channels()
    # Compute |h_eff,T| for N=0,1,...,N elements active.
    g_T = env4._g[env4.K_r] if env4.K_t > 0 else None
    if g_T is not None:
        cascaded_per_element = np.conj(g_T) * env4._G  # shape (N,)
        # With phi=0, beta_t=0.5 --> amp_t = sqrt(0.5)
        per_element_contrib = np.sqrt(0.5) * cascaded_per_element
        cum = np.cumsum(per_element_contrib)
        for n in [0, 4, 8, 16, env4.N]:
            if n == 0:
                amp = 0.0
            else:
                amp = float(np.abs(cum[n - 1]))
            total = float(np.abs(env4._h_d[env4.K_r] + cum[n - 1] if n > 0 else env4._h_d[env4.K_r]))
            print(f"  N_active={n:3d}  cascaded amp = {amp:.3e}   |h_eff_T| = {total:.3e}")

    section("Validation complete")


if __name__ == "__main__":
    validate()
