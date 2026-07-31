```text
Conditioned transport audit runner: 32 targets x 3 arms
  registered arms: cta-exact-hybrid, cta-exact-fixed-control, paper-neural-optimized
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm-v00: L=64, kappa=1.61, 1.96s
  NPR-d2-balanced-gmm-v01: L=64, kappa=1.55, 1.89s
  NPR-d2-rare-gmm-v00: L=64, kappa=1.58, 1.84s
  NPR-d2-rare-gmm-v01: L=64, kappa=1.46, 1.84s
  NPR-d2-correlated-t-v00: L=64, kappa=1.51, 1.84s
  NPR-d2-correlated-t-v01: L=64, kappa=1.56, 1.96s
  NPR-d2-nonlinear-v00: L=64, kappa=1.61, 1.89s
  NPR-d2-nonlinear-v01: L=64, kappa=1.64, 1.95s
  NPR-d4-balanced-gmm-v00: L=64, kappa=2.24, 1.96s
  NPR-d4-balanced-gmm-v01: L=64, kappa=2.48, 1.98s
  NPR-d4-rare-gmm-v00: L=64, kappa=2.53, 2.00s
  NPR-d4-rare-gmm-v01: L=64, kappa=2.34, 2.04s
  NPR-d4-correlated-t-v00: L=64, kappa=2.24, 2.07s
  NPR-d4-correlated-t-v01: L=64, kappa=2.74, 2.03s
  NPR-d4-nonlinear-v00: L=64, kappa=2.57, 1.95s
  NPR-d4-nonlinear-v01: L=64, kappa=2.74, 1.95s
  NPR-d8-balanced-gmm-v00: L=64, kappa=7.39, 2.61s
  NPR-d8-balanced-gmm-v01: L=64, kappa=9.86, 2.66s
  NPR-d8-rare-gmm-v00: L=64, kappa=9.38, 2.66s
  NPR-d8-rare-gmm-v01: L=64, kappa=9.75, 2.59s
  NPR-d8-correlated-t-v00: L=64, kappa=6.97, 2.60s
  NPR-d8-correlated-t-v01: L=64, kappa=7.46, 2.67s
  NPR-d8-nonlinear-v00: L=64, kappa=9.19, 2.68s
  NPR-d8-nonlinear-v01: L=64, kappa=8.42, 2.64s
  NPR-d16-balanced-gmm-v00: L=192, kappa=18.44, 3.80s
  NPR-d16-balanced-gmm-v01: L=192, kappa=19.23, 3.87s
  NPR-d16-rare-gmm-v00: L=176, kappa=23.83, 3.92s
  NPR-d16-rare-gmm-v01: L=192, kappa=20.05, 3.79s
  NPR-d16-correlated-t-v00: L=192, kappa=19.25, 3.96s
  NPR-d16-correlated-t-v01: L=192, kappa=20.08, 4.03s
  NPR-d16-nonlinear-v00: L=192, kappa=18.57, 3.83s
  NPR-d16-nonlinear-v01: L=192, kappa=18.48, 3.93s
  cta-exact-hybrid         ED2=0.015987 SW1=0.078898 train=0.427s setup+train=0.478s coverage>=0.500 local=0.250
  cta-exact-fixed-control  ED2=0.016459 SW1=0.078227 train=0.268s setup+train=0.322s coverage>=1.000 local=0.250
  paper-neural-optimized   ED2=0.044508 SW1=0.154603 train=0.294s setup+train=0.294s coverage>=0.500 local=0.000
```
