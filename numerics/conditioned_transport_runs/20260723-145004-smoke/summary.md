```text
Conditioned transport audit runner: 4 targets x 2 arms
  registered arms: cta-exact-hybrid, cta-exact-fixed-control
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.15s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.16s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.20s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.33s
  cta-exact-hybrid         ED2=0.070643 SW1=0.176580 train=0.046s setup+train=0.051s coverage>=0.500 local=0.250
  cta-exact-fixed-control  ED2=0.070190 SW1=0.175326 train=0.032s setup+train=0.038s coverage>=0.500 local=0.250
```
