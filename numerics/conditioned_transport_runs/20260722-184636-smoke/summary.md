```text
Conditioned transport development: 4 targets x 5 arms
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted projection-tree representatives
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] quadratic-frame and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.46s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.43s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.55s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.80s
  cta-exact-global         ED2=0.077853 SW1=0.186097 local=0.000
  cta-kll-global           ED2=0.078558 SW1=0.185245 local=0.000
  cta-exact-hybrid         ED2=0.068796 SW1=0.176182 local=0.250
  cta-kll-hybrid           ED2=0.069120 SW1=0.177450 local=0.250
  cta-kll-guarded          ED2=0.075375 SW1=0.173161 local=0.138
```
