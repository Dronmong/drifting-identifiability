```text
Conditioned transport audit runner: 4 targets x 6 arms
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.49s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.51s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.63s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.91s
  cta-exact-global         ED2=0.077853 SW1=0.186097 train=0.015s setup+train=0.020s coverage>=0.500 local=0.000
  cta-kll-global           ED2=0.077030 SW1=0.184801 train=0.014s setup+train=0.025s coverage>=0.500 local=0.000
  cta-exact-hybrid         ED2=0.070246 SW1=0.176240 train=0.070s setup+train=0.075s coverage>=0.500 local=0.250
  cta-kll-hybrid           ED2=0.071013 SW1=0.175857 train=0.058s setup+train=0.068s coverage>=0.500 local=0.250
  cta-kll-guarded          ED2=0.070807 SW1=0.174952 train=0.061s setup+train=0.068s coverage>=0.500 local=0.156
  paper-neural-optimized   ED2=0.125239 SW1=0.251257 train=0.033s setup+train=0.033s coverage>=0.500 local=0.000
```
