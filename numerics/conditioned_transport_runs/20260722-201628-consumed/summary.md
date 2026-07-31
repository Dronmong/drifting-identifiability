```text
Conditioned transport audit runner: 16 targets x 6 arms
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 64 weighted projection-tree representatives
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.43, 3.59s
  NPR-d2-rare-gmm: L=64, kappa=1.52, 3.32s
  NPR-d2-correlated-t: L=64, kappa=1.42, 3.28s
  NPR-d2-nonlinear: L=64, kappa=1.44, 3.50s
  NPR-d4-balanced-gmm: L=64, kappa=2.41, 3.63s
  NPR-d4-rare-gmm: L=64, kappa=2.33, 3.60s
  NPR-d4-correlated-t: L=64, kappa=3.00, 3.77s
  NPR-d4-nonlinear: L=64, kappa=2.54, 3.67s
  NPR-d8-balanced-gmm: L=64, kappa=8.99, 5.00s
  NPR-d8-rare-gmm: L=64, kappa=8.28, 5.31s
  NPR-d8-correlated-t: L=64, kappa=8.04, 4.70s
  NPR-d8-nonlinear: L=64, kappa=7.97, 4.70s
  NPR-d16-balanced-gmm: L=192, kappa=19.42, 7.03s
  NPR-d16-rare-gmm: L=192, kappa=23.24, 7.14s
  NPR-d16-correlated-t: L=192, kappa=21.06, 7.32s
  NPR-d16-nonlinear: L=192, kappa=18.00, 7.31s
  cta-exact-global         ED2=0.017265 SW1=0.101907 train=0.155s setup+train=0.213s coverage>=0.500 local=0.000
  cta-kll-global           ED2=0.017059 SW1=0.096562 train=0.155s setup+train=0.201s coverage>=0.500 local=0.000
  cta-exact-hybrid         ED2=0.013882 SW1=0.090193 train=0.217s setup+train=0.274s coverage>=0.500 local=0.250
  cta-kll-hybrid           ED2=0.016145 SW1=0.089648 train=0.222s setup+train=0.270s coverage>=0.500 local=0.250
  cta-kll-guarded          ED2=0.013732 SW1=0.087516 train=0.264s setup+train=0.319s coverage>=0.500 local=0.184
  paper-neural-optimized   ED2=0.047305 SW1=0.149031 train=0.355s setup+train=0.355s coverage>=0.500 local=0.000
```
