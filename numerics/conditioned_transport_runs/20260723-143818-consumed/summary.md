```text
Conditioned transport audit runner: 16 targets x 1 arms
  registered arms: cta-exact-hybrid
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.76s
  NPR-d2-rare-gmm: L=64, kappa=1.49, 0.82s
  NPR-d2-correlated-t: L=64, kappa=1.58, 0.75s
  NPR-d2-nonlinear: L=64, kappa=1.45, 0.74s
  NPR-d4-balanced-gmm: L=64, kappa=2.57, 0.79s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.76s
  NPR-d4-correlated-t: L=64, kappa=2.48, 0.77s
  NPR-d4-nonlinear: L=64, kappa=2.46, 0.77s
  NPR-d8-balanced-gmm: L=64, kappa=8.25, 1.07s
  NPR-d8-rare-gmm: L=64, kappa=9.77, 1.08s
  NPR-d8-correlated-t: L=64, kappa=8.81, 1.10s
  NPR-d8-nonlinear: L=64, kappa=7.63, 1.09s
  NPR-d16-balanced-gmm: L=192, kappa=18.92, 1.66s
  NPR-d16-rare-gmm: L=176, kappa=24.90, 1.55s
  NPR-d16-correlated-t: L=176, kappa=24.02, 1.54s
  NPR-d16-nonlinear: L=192, kappa=19.10, 1.57s
  cta-exact-hybrid         ED2=0.013972 SW1=0.077713 train=0.454s setup+train=0.508s coverage>=1.000 local=0.250
```
