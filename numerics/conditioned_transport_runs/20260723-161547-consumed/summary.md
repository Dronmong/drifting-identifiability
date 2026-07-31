```text
Conditioned transport audit runner: 16 targets x 4 arms
  registered arms: cta-exact-adaptive-rollout, cta-exact-fixed-control, cta-exact-gated-hybrid, paper-neural-optimized
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 2.47s
  NPR-d2-rare-gmm: L=64, kappa=1.49, 2.49s
  NPR-d2-correlated-t: L=64, kappa=1.58, 2.47s
  NPR-d2-nonlinear: L=64, kappa=1.45, 2.52s
  NPR-d4-balanced-gmm: L=64, kappa=2.57, 2.67s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 2.71s
  NPR-d4-correlated-t: L=64, kappa=2.48, 2.74s
  NPR-d4-nonlinear: L=64, kappa=2.46, 2.74s
  NPR-d8-balanced-gmm: L=64, kappa=8.25, 3.76s
  NPR-d8-rare-gmm: L=64, kappa=9.77, 3.72s
  NPR-d8-correlated-t: L=64, kappa=8.81, 3.73s
  NPR-d8-nonlinear: L=64, kappa=7.63, 3.71s
  NPR-d16-balanced-gmm: L=192, kappa=18.92, 5.37s
  NPR-d16-rare-gmm: L=176, kappa=24.90, 5.42s
  NPR-d16-correlated-t: L=176, kappa=24.02, 5.42s
  NPR-d16-nonlinear: L=192, kappa=19.10, 5.50s
  cta-exact-adaptive-rollout ED2=0.011856 SW1=0.070758 train=0.402s setup+train=0.452s coverage>=1.000 local=0.250
  cta-exact-fixed-control  ED2=0.013562 SW1=0.079468 train=0.289s setup+train=0.342s coverage>=1.000 local=0.250
  cta-exact-gated-hybrid   ED2=0.013853 SW1=0.077713 train=0.389s setup+train=0.439s coverage>=1.000 local=0.250
  paper-neural-optimized   ED2=0.040738 SW1=0.144413 train=0.329s setup+train=0.329s coverage>=0.500 local=0.000
```
