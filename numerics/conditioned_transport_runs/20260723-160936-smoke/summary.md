```text
Conditioned transport audit runner: 4 targets x 4 arms
  registered arms: cta-exact-fixed-control, cta-exact-gated-hybrid, cta-exact-rollout2, cta-exact-rollout4-safe-balanced
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.31s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.34s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.45s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.66s
  cta-exact-fixed-control  ED2=0.067944 SW1=0.176608 train=0.035s setup+train=0.041s coverage>=0.500 local=0.250
  cta-exact-gated-hybrid   ED2=0.066748 SW1=0.177862 train=0.045s setup+train=0.050s coverage>=0.500 local=0.250
  cta-exact-rollout2       ED2=0.056614 SW1=0.173287 train=0.047s setup+train=0.054s coverage>=0.500 local=0.250
  cta-exact-rollout4-safe-balanced ED2=0.069146 SW1=0.188862 train=0.057s setup+train=0.065s coverage>=0.500 local=0.070
```
