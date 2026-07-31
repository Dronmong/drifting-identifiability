```text
Conditioned transport audit runner: 4 targets x 5 arms
  registered arms: cta-exact-adaptive-rollout, cta-exact-adaptive-rollout-safe, cta-exact-adaptive-rollout-rank-balanced, cta-exact-adaptive-rollout-safe-rank-balanced, paper-neural-optimized
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.38s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.41s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.61s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.81s
  cta-exact-adaptive-rollout ED2=0.073434 SW1=0.176368 train=0.052s setup+train=0.057s coverage>=0.500 local=0.250
  cta-exact-adaptive-rollout-safe ED2=0.082395 SW1=0.182130 train=0.051s setup+train=0.060s coverage>=0.500 local=0.167
  cta-exact-adaptive-rollout-rank-balanced ED2=0.070568 SW1=0.174910 train=0.050s setup+train=0.057s coverage>=0.500 local=0.250
  cta-exact-adaptive-rollout-safe-rank-balanced ED2=0.082730 SW1=0.181321 train=0.050s setup+train=0.055s coverage>=0.500 local=0.166
  paper-neural-optimized   ED2=0.125239 SW1=0.251257 train=0.031s setup+train=0.031s coverage>=0.500 local=0.000
```
