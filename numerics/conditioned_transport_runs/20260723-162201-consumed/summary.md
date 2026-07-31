```text
Conditioned transport audit runner: 32 targets x 4 arms
  registered arms: cta-exact-adaptive-rollout, cta-exact-fixed-control, cta-exact-gated-hybrid, paper-neural-optimized
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NCT-d2-balanced-gmm-i0: L=64, kappa=1.51, 2.45s
  NCT-d2-balanced-gmm-i1: L=64, kappa=1.54, 2.51s
  NCT-d2-rare-gmm-i0: L=64, kappa=1.53, 2.37s
  NCT-d2-rare-gmm-i1: L=64, kappa=1.43, 2.47s
  NCT-d2-correlated-t-i0: L=64, kappa=1.77, 2.48s
  NCT-d2-correlated-t-i1: L=64, kappa=1.50, 2.42s
  NCT-d2-nonlinear-i0: L=64, kappa=1.53, 2.38s
  NCT-d2-nonlinear-i1: L=64, kappa=1.52, 2.40s
  NCT-d4-balanced-gmm-i0: L=64, kappa=2.72, 2.61s
  NCT-d4-balanced-gmm-i1: L=64, kappa=2.61, 2.64s
  NCT-d4-rare-gmm-i0: L=64, kappa=2.59, 2.62s
  NCT-d4-rare-gmm-i1: L=64, kappa=2.41, 2.63s
  NCT-d4-correlated-t-i0: L=64, kappa=2.31, 2.61s
  NCT-d4-correlated-t-i1: L=64, kappa=2.22, 2.68s
  NCT-d4-nonlinear-i0: L=64, kappa=2.44, 2.66s
  NCT-d4-nonlinear-i1: L=64, kappa=2.21, 2.62s
  NCT-d8-balanced-gmm-i0: L=64, kappa=10.69, 3.75s
  NCT-d8-balanced-gmm-i1: L=64, kappa=8.15, 3.90s
  NCT-d8-rare-gmm-i0: L=64, kappa=7.38, 3.90s
  NCT-d8-rare-gmm-i1: L=64, kappa=6.56, 3.76s
  NCT-d8-correlated-t-i0: L=64, kappa=8.28, 3.77s
  NCT-d8-correlated-t-i1: L=64, kappa=8.36, 3.81s
  NCT-d8-nonlinear-i0: L=64, kappa=8.00, 3.83s
  NCT-d8-nonlinear-i1: L=64, kappa=8.82, 3.78s
  NCT-d16-balanced-gmm-i0: L=192, kappa=18.43, 5.50s
  NCT-d16-balanced-gmm-i1: L=192, kappa=22.31, 5.41s
  NCT-d16-rare-gmm-i0: L=192, kappa=21.02, 5.55s
  NCT-d16-rare-gmm-i1: L=192, kappa=21.32, 5.55s
  NCT-d16-correlated-t-i0: L=192, kappa=22.16, 5.50s
  NCT-d16-correlated-t-i1: L=192, kappa=19.26, 5.47s
  NCT-d16-nonlinear-i0: L=192, kappa=20.52, 5.57s
  NCT-d16-nonlinear-i1: L=192, kappa=19.94, 5.50s
  cta-exact-adaptive-rollout ED2=0.012812 SW1=0.073439 train=0.403s setup+train=0.454s coverage>=0.500 local=0.250
  cta-exact-fixed-control  ED2=0.016273 SW1=0.082316 train=0.289s setup+train=0.342s coverage>=0.500 local=0.250
  cta-exact-gated-hybrid   ED2=0.015938 SW1=0.080882 train=0.405s setup+train=0.457s coverage>=0.500 local=0.250
  paper-neural-optimized   ED2=0.049473 SW1=0.163153 train=0.333s setup+train=0.333s coverage>=0.500 local=0.000
```
