```text
Conditioned transport audit runner: 16 targets x 1 arms
  registered arms: cta-exact-hybrid
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-with-tail-reserve representatives
  representative tail reserve fraction: 0.062500
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 1.68s
  NPR-d2-rare-gmm: L=64, kappa=1.49, 1.61s
  NPR-d2-correlated-t: L=64, kappa=1.58, 1.61s
  NPR-d2-nonlinear: L=64, kappa=1.45, 1.58s
  NPR-d4-balanced-gmm: L=64, kappa=2.57, 1.68s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 1.69s
  NPR-d4-correlated-t: L=64, kappa=2.48, 1.67s
  NPR-d4-nonlinear: L=64, kappa=2.46, 1.70s
  NPR-d8-balanced-gmm: L=64, kappa=8.25, 1.86s
  NPR-d8-rare-gmm: L=64, kappa=9.77, 1.85s
  NPR-d8-correlated-t: L=64, kappa=8.81, 1.88s
  NPR-d8-nonlinear: L=64, kappa=7.63, 1.87s
  NPR-d16-balanced-gmm: L=192, kappa=18.92, 2.40s
  NPR-d16-rare-gmm: L=176, kappa=24.90, 2.39s
  NPR-d16-correlated-t: L=176, kappa=24.02, 2.34s
  NPR-d16-nonlinear: L=192, kappa=19.10, 2.36s
  cta-exact-hybrid         ED2=0.014672 SW1=0.078461 train=1.279s setup+train=1.334s coverage>=1.000 local=0.250
```
