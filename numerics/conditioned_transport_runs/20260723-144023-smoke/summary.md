```text
Conditioned transport audit runner: 4 targets x 3 arms
  registered arms: cta-exact-hybrid, cta-exact-fixed-control, paper-neural-optimized
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.23s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.24s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.31s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.48s
  cta-exact-hybrid         ED2=0.066330 SW1=0.177152 train=0.050s setup+train=0.057s coverage>=0.500 local=0.250
  cta-exact-fixed-control  ED2=0.067944 SW1=0.176608 train=0.034s setup+train=0.042s coverage>=0.500 local=0.250
  paper-neural-optimized   ED2=0.137707 SW1=0.251907 train=0.032s setup+train=0.032s coverage>=0.500 local=0.000
```
