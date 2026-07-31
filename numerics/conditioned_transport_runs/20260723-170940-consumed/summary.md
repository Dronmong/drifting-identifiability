```text
Conditioned transport audit runner: 16 targets x 4 arms
  registered arms: cta-exact-adaptive-rollout, cta-exact-adaptive-rollout-safe, cta-exact-adaptive-rollout-rank-balanced, cta-exact-adaptive-rollout-safe-rank-balanced
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 2.46s
  NPR-d2-rare-gmm: L=64, kappa=1.49, 2.35s
  NPR-d2-correlated-t: L=64, kappa=1.58, 2.31s
  NPR-d2-nonlinear: L=64, kappa=1.45, 2.32s
  NPR-d4-balanced-gmm: L=64, kappa=2.57, 2.53s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 2.55s
  NPR-d4-correlated-t: L=64, kappa=2.48, 2.49s
  NPR-d4-nonlinear: L=64, kappa=2.46, 2.45s
  NPR-d8-balanced-gmm: L=64, kappa=8.25, 4.44s
  NPR-d8-rare-gmm: L=64, kappa=9.77, 4.48s
  NPR-d8-correlated-t: L=64, kappa=8.81, 4.29s
  NPR-d8-nonlinear: L=64, kappa=7.63, 4.38s
  NPR-d16-balanced-gmm: L=192, kappa=18.92, 6.23s
  NPR-d16-rare-gmm: L=176, kappa=24.90, 6.61s
  NPR-d16-correlated-t: L=176, kappa=24.02, 6.48s
  NPR-d16-nonlinear: L=192, kappa=19.10, 6.27s
  cta-exact-adaptive-rollout ED2=0.011856 SW1=0.070758 train=0.393s setup+train=0.442s coverage>=1.000 local=0.250
  cta-exact-adaptive-rollout-safe ED2=0.014677 SW1=0.075955 train=0.408s setup+train=0.458s coverage>=1.000 local=0.178
  cta-exact-adaptive-rollout-rank-balanced ED2=0.011699 SW1=0.071097 train=0.429s setup+train=0.480s coverage>=1.000 local=0.250
  cta-exact-adaptive-rollout-safe-rank-balanced ED2=0.012880 SW1=0.074278 train=0.436s setup+train=0.485s coverage>=1.000 local=0.177
```
