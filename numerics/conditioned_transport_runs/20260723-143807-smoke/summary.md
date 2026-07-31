```text
Conditioned transport audit runner: 4 targets x 1 arms
  registered arms: cta-exact-hybrid
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.11s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.09s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.11s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.20s
  cta-exact-hybrid         ED2=0.070643 SW1=0.176580 train=0.048s setup+train=0.054s coverage>=0.500 local=0.250
```
