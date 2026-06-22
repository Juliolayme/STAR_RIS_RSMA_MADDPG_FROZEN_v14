# Results Summary â€” DRL Resource Allocation in STAR-RIS Assisted RSMA Networks
## 1. System Setup
- SISO downlink, K = 4 users (K_R = 3), N = 32 STAR-RIS elements (ES mode).
- P_max = 30.0 dBm, noise = -90.0 dBm, per-user QoS = 0.3 b/s/Hz, T-blockage = 25.0 dB.
- Reward: quadratic QoS with adaptive Lagrangian Î» (init 1.0, target satisfaction 0.5).
## 2. Algorithm Comparison (5-seed deterministic eval)
```
Algorithm   Return  Return_CI  SumRate  SumRate_CI  QoS_prob   QoS_CI  RateCommon  |h_eff_T|  PhaseEntropy_T  P_c/Pmax  Latency_ms  trained_lambda_mean  N_train_seeds
   MADDPG 1.743545   0.539838 2.793479    0.076072  0.444378 0.057657    1.137963   0.000002        2.502058  0.750048    1.348161            11.266749             12
     DDPG 2.082610   0.362838 1.765916    0.025507  0.490211 0.073989    0.234148   0.000003        2.496630  0.199339    0.504931            13.907229             12
      TD3 1.184113   0.392251 2.714911    0.155156  0.322511 0.071546    1.028043   0.000003        2.506991  0.633487    0.501230            14.833342             12
      PPO 3.311248   0.007776 2.165164    0.021027  0.998900 0.001652    0.531055   0.000007        2.494982  0.315992    1.186548            15.000000             12
```
## 3. Sum-rate vs P_max (5-seed)
```
Pmax(dBm) | MADDPG | DDPG | TD3 | PPO | FixedRIS
     10.0 | 0.792 | 1.049 | 0.841 | 1.111 | 0.790
     15.0 | 1.166 | 1.257 | 1.230 | 1.459 | 1.138
     20.0 | 1.577 | 1.434 | 1.673 | 1.791 | 1.481
     25.0 | 2.088 | 1.624 | 2.220 | 2.023 | 1.876
     30.0 | 2.768 | 1.802 | 2.835 | 2.135 | 2.386
     35.0 | 3.478 | 1.921 | 3.354 | 2.177 | 3.007
```
## 4. QoS Satisfaction Probability (5-seed)
```
  MADDPG            : 0.581 Â± 0.064
  DDPG              : 0.566 Â± 0.065
  TD3               : 0.469 Â± 0.070
  PPO               : 1.000 Â± 0.000
  FixedRIS          : 0.349 Â± 0.059
```
## 5. Inference Latency (ms / action)
```
  MADDPG            : 1.289 ms
  DDPG              : 0.444 ms
  TD3               : 0.436 ms
  PPO               : 1.120 ms
  FixedRIS          : 1.264 ms
```
## 6. Expanded STAR-RIS Ablation
```
  Learned                sr=2.772 Â± 0.120   QoS=0.581 Â± 0.067   |h_T|=2.22e-06   R_c=1.152   P_c/Pmax=0.755
  BCD                    sr=5.148 Â± 0.111   QoS=1.000 Â± 0.000   |h_T|=8.64e-06   R_c=3.711   P_c/Pmax=0.950
  MaxMinAlignedRIS       sr=3.894 Â± 0.137   QoS=0.489 Â± 0.109   |h_T|=7.05e-06   R_c=1.768   P_c/Pmax=0.671
  FixedRIS               sr=2.387 Â± 0.120   QoS=0.338 Â± 0.061   |h_T|=1.51e-06   R_c=0.761   P_c/Pmax=0.733
  RandomRIS              sr=2.345 Â± 0.067   QoS=0.338 Â± 0.038   |h_T|=1.57e-06   R_c=0.758   P_c/Pmax=0.747
  NoRIS                  sr=2.028 Â± 0.077   QoS=0.223 Â± 0.035   |h_T|=1.08e-06   R_c=0.488   P_c/Pmax=0.742
  EqualPower+Learned     sr=1.704 Â± 0.020   QoS=0.582 Â± 0.099   |h_T|=2.24e-06   R_c=0.213   P_c/Pmax=0.200
  EqualPower+Fixed       sr=1.548 Â± 0.023   QoS=0.273 Â± 0.119   |h_T|=1.51e-06   R_c=0.145   P_c/Pmax=0.200
```
## 7. Wireless Interpretation
The trained MADDPG agent jointly optimizes RSMA power, common-stream split, and STAR-RIS amplitude/phase coefficients. The T-region users â€” physically NLoS due to blockage â€” depend almost entirely on the cascaded link. The agent must therefore both (i) allocate most BS power to the common stream (whose rate is gated by the weakest user, R_c = min_k log2(1 + Î³_c,k)) and (ii) align the STAR-RIS transmission phases to maximize |h_eff,T|. The expanded ablation isolates each axis: EqualPower variants quantify the value of learned BS power allocation, while AnalyticalRIS gives the closed-form upper bound for phase alignment that the policy approximates.
## 8. Limitations
- M = 1 (SISO BS); MIMO with beamforming is left for future work.
- The N-sweep / K-sweep only report at the trained topology (dimension-locked).
- Single training seed (multi-seed is at the EVALUATION stage). Multi-seed training would tighten convergence-curve CIs.
- Hardware impairments (quantized phases, RIS amplitude coupling, channel estimation error) are not modeled.
## 9. Future Work
- Multi-seed training (5Ã—) for tighter convergence statistics.
- Curriculum learning over T-blockage to ease phase-alignment discovery.
- Graph- or attention-based actor that generalizes across N and K without retraining.
- Compare against optimization-based benchmarks (BCD, SDR, alternating optimization).
Beta
0 / 0
used queries
1