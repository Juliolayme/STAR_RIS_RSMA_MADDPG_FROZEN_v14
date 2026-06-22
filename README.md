# DRL Resource Allocation in STAR-RIS Assisted RSMA Networks

Reference implementation of MADDPG + DDPG/TD3/PPO baselines for joint power and
STAR-RIS phase optimization in a downlink SISO RSMA network with NLoS T-region.

This snapshot (v13) is the **frozen reproducibility package** accompanying the paper.
The code, hyperparameters, and random seeds are pinned so that re-running this
package yields outputs that match the included `results_summary.md` exactly on CPU.

---

## 1. Quick reproduction

```bash
# Use the included Python 3.10 environment, or recreate it:
pip install -r requirements.txt

# Reproduce the full v13 paper figures + tables (5 seeds × 4 algorithms × 1000 episodes):
python main.py

# Smoke test (10 episodes per seed, ~1 minute):
python main.py --quick
```

The full run takes **~6 hours on CPU** (single workstation, no GPU needed).
Outputs land in:
- `figures/` — IEEE-quality PDF/PNG/SVG
- `tables/` — CSV + LaTeX (algorithm_comparison, ablation, significance, ...)
- `logs/` — per-seed CSV training histories + TensorBoard events
- `checkpoints/` — `best.pt` and `latest.pt` per (algorithm, seed)
- `results_summary.md` — auto-generated report

## 2. Determinism guarantee

`experiments/train.py::_set_seed` sets:

- `PYTHONHASHSEED`, `random.seed`, `numpy.seed`, `torch.manual_seed`
- `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`

Training seeds used (config: `training.training_seeds`): **[1000, 2000, 3000, 4000, 5000]**.
Evaluation seeds (config: `evaluation.seeds`): **[11, 22, 33, 44, 55]**.

On CPU, re-running `python main.py` reproduces the numbers in `results_summary.md`
to within floating-point precision. On GPU, exact reproducibility depends on CUDA
non-deterministic kernels; expect ≤0.5% deviation across re-runs even with
`cudnn.deterministic=True`.

## 3. Project layout

```
STAR_RIS_RSMA_MADDPG/
├── config/config.yaml          # All hyperparameters (the SINGLE source of truth)
├── env/star_ris_env.py         # System model: STAR-RIS, RSMA, channels, reward
├── algorithms/
│   ├── maddpg/agent.py         # CTDE multi-agent (BS-power, RIS-R, RIS-T)
│   ├── ddpg/agent.py           # Single-agent baseline
│   ├── td3/agent.py            # Twin-delay baseline
│   └── ppo/agent.py            # On-policy baseline
├── networks/                   # Actor / Critic MLPs
├── utils/
│   ├── metrics.py              # Welch's t-test (Student-t CDF), confidence intervals
│   ├── normalization.py        # Running mean/std for observation normalization
│   ├── plotting.py             # IEEE figure utilities (PDF/PNG/SVG, CI bands)
│   └── replay_buffer.py        # MA + single-agent replay buffers
├── experiments/
│   ├── train.py                # Training drivers + deterministic seeding
│   ├── evaluate.py             # Multi-seed evaluation sweeps
│   ├── ablation.py             # 7-cell STAR-RIS ablation
│   └── validate_channel.py     # Channel-model sanity diagnostics (run standalone)
├── main.py                     # End-to-end pipeline
└── requirements.txt            # Pinned dependencies
```

## 4. Headline results (from `results_summary.md`)

### Algorithm comparison (5 training seeds × 5 eval seeds, Student-t 95% CI)

| Algorithm | Sum-rate (b/s/Hz) | QoS satisfaction | P_c/Pmax |
|---|---|---|---|
| **MADDPG** | **2.78 ± 0.10** | **52.5%** ± 0.14 | 0.82 |
| TD3       | 2.71 ± 0.26 | 37.3% ± 0.10 | 0.62 |
| PPO       | 2.20 ± 0.08 | 99.9% (degenerate uniform-power, documented) | 0.33 |
| DDPG      | 1.76 ± 0.02 | 55.7% ± 0.15 | 0.19 |

MADDPG Pareto-dominates TD3 on BOTH sum-rate (+2.4%) AND QoS (+15 percentage points).

### STAR-RIS ablation

| Mode | Sum-rate | vs NoRIS |
|---|---|---|
| MaxMinAlignedRIS (closed-form upper bound) | 4.27 | +114% |
| **Learned (MADDPG)** | **2.77** | **+39%** |
| RandomRIS | 2.35 | +18% |
| FixedRIS | 2.32 | +17% |
| NoRIS | 1.99 | — |

## 5. Verification script

To verify a clean install reproduces the reported numbers without retraining,
load the checkpoints and run the evaluation pipeline only:

```bash
# 1) Sanity-check the system model (deterministic):
python experiments/validate_channel.py

# 2) Re-evaluate frozen checkpoints (skip training):
python main.py --skip-train     # uses included checkpoints/
```

`--skip-train` requires the included `checkpoints/` directory — already in the zip.

## 6. Key methodological choices (briefly)

- **Physics-informed phase action** (`config: phase_action_mode: residual`,
  `phase_residual_scale: 0.5`): the actor outputs a ±π/2 residual on top of the
  closed-form max-min single-user alignment phase. Standard hybrid analytical-RL
  parameterization in RL-for-RIS literature.
- **Adaptive Lagrangian QoS** (`qos_lambda_init: 1.0`, target satisfaction 0.5):
  multiplier raised when QoS unsatisfied, clamped to `[0.3, 15.0]`. Saturates at
  upper bound for this geometry — disclosed in the paper as the binding-constraint
  regime.
- **Per-episode block fading** (`channel_block_steps: 50 = max_steps`): one
  Rayleigh realization per episode; 1000 episodes × 5 seeds = 5000 realizations
  total per algorithm.
- **True CTDE** (`local_obs_for_maddpg: true`): MADDPG actors see factorized
  per-region observations (BS=26-dim, RIS-R=280-dim, RIS-T=148-dim); centralized
  critic sees the joint concatenation. Single-agent baselines see the full
  346-dim global observation for fair comparison.

## 7. Known limitations (disclosed)

1. **PPO converges to a near-uniform-power degenerate policy** (P_c/Pmax = 0.33,
   QoS = 99.9%). Reproduced exactly here; reported as the PPO baseline outcome
   under the chosen reward shaping.
2. **MADDPG vs TD3 sum-rate gap (2.4%) is within Welch t-test significance**
   (p = 0.19); claim relies on Pareto-dominance, not strict significance.
3. **Analytical phase baseline is single-user max-min alignment**, not the true
   multi-user optimum. Renamed `MaxMinAlignedRIS` to be honest about this.
4. **`N`-sweep and `K`-sweep are dimension-locked**: only the trained config
   produces valid eval, other points return NaN. Re-training per topology is
   required for a true scalability claim (omitted from the paper figures).

## 8. Contact

This is a frozen submission snapshot. The accompanying paper documents the
modeling, methodology, and complete reviewer-response history. See `train.log`
for the actual run that produced these results.
