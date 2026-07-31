```text
Conditioned transport audit runner: 16 targets x 3 arms
  registered arms: cta-exact-crossfit, cta-exact-hybrid, cta-exact-global
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 2.04s
  NPR-d2-rare-gmm: L=64, kappa=1.49, 2.02s
  NPR-d2-correlated-t: L=64, kappa=1.58, 1.96s
  NPR-d2-nonlinear: L=64, kappa=1.45, 1.99s
  NPR-d4-balanced-gmm: L=64, kappa=2.57, 2.09s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 2.13s
  NPR-d4-correlated-t: L=64, kappa=2.48, 2.11s
  NPR-d4-nonlinear: L=64, kappa=2.46, 2.11s
  NPR-d8-balanced-gmm: L=64, kappa=8.25, 2.61s
  NPR-d8-rare-gmm: L=64, kappa=9.77, 2.61s
  NPR-d8-correlated-t: L=64, kappa=8.81, 2.65s
  NPR-d8-nonlinear: L=64, kappa=7.63, 2.63s
  NPR-d16-balanced-gmm: L=192, kappa=18.92, 4.03s
  NPR-d16-rare-gmm: L=176, kappa=24.90, 3.95s
  NPR-d16-correlated-t: L=176, kappa=24.02, 3.89s
  NPR-d16-nonlinear: L=192, kappa=19.10, 3.91s
  cta-exact-crossfit       ED2=0.016032 SW1=0.081663 train=0.508s setup+train=0.586s coverage>=1.000 local=0.234
  cta-exact-hybrid         ED2=0.014876 SW1=0.078704 train=0.372s setup+train=0.423s coverage>=1.000 local=0.250
  cta-exact-global         ED2=0.020607 SW1=0.098488 train=0.132s setup+train=0.183s coverage>=1.000 local=0.000
```
