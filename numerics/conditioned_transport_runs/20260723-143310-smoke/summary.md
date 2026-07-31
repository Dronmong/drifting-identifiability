```text
Conditioned transport audit runner: 4 targets x 3 arms
  registered arms: cta-exact-crossfit, cta-exact-hybrid, cta-exact-global
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.23s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.25s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.29s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.45s
  cta-exact-crossfit       ED2=0.070547 SW1=0.174497 train=0.058s setup+train=0.066s coverage>=0.500 local=0.156
  cta-exact-hybrid         ED2=0.070246 SW1=0.176240 train=0.044s setup+train=0.049s coverage>=0.500 local=0.250
  cta-exact-global         ED2=0.077853 SW1=0.186097 train=0.014s setup+train=0.020s coverage>=0.500 local=0.000
```
