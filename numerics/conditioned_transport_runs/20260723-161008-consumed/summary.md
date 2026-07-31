```text
Conditioned transport audit runner: 16 targets x 6 arms
  registered arms: cta-exact-fixed-control, cta-exact-gated-hybrid, cta-exact-rollout2, cta-exact-rollout4, cta-exact-rollout4-safe, cta-exact-rollout4-safe-balanced
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 4.27s
  NPR-d2-rare-gmm: L=64, kappa=1.49, 4.16s
  NPR-d2-correlated-t: L=64, kappa=1.58, 3.90s
  NPR-d2-nonlinear: L=64, kappa=1.45, 3.99s
  NPR-d4-balanced-gmm: L=64, kappa=2.57, 4.11s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 4.16s
  NPR-d4-correlated-t: L=64, kappa=2.48, 4.53s
  NPR-d4-nonlinear: L=64, kappa=2.46, 4.57s
  NPR-d8-balanced-gmm: L=64, kappa=8.25, 6.46s
  NPR-d8-rare-gmm: L=64, kappa=9.77, 6.24s
  NPR-d8-correlated-t: L=64, kappa=8.81, 6.29s
  NPR-d8-nonlinear: L=64, kappa=7.63, 6.23s
  NPR-d16-balanced-gmm: L=192, kappa=18.92, 8.57s
  NPR-d16-rare-gmm: L=176, kappa=24.90, 8.67s
  NPR-d16-correlated-t: L=176, kappa=24.02, 8.71s
  NPR-d16-nonlinear: L=192, kappa=19.10, 8.62s
  cta-exact-fixed-control  ED2=0.013562 SW1=0.079468 train=0.287s setup+train=0.339s coverage>=1.000 local=0.250
  cta-exact-gated-hybrid   ED2=0.013853 SW1=0.077713 train=0.385s setup+train=0.436s coverage>=1.000 local=0.250
  cta-exact-rollout2       ED2=0.016364 SW1=0.081748 train=0.416s setup+train=0.466s coverage>=1.000 local=0.250
  cta-exact-rollout4       ED2=0.019106 SW1=0.077013 train=0.479s setup+train=0.529s coverage>=1.000 local=0.250
  cta-exact-rollout4-safe  ED2=0.016616 SW1=0.080289 train=0.506s setup+train=0.556s coverage>=1.000 local=0.074
  cta-exact-rollout4-safe-balanced ED2=0.012790 SW1=0.083255 train=0.518s setup+train=0.567s coverage>=1.000 local=0.074
```
