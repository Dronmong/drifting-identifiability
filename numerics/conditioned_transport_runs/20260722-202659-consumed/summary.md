```text
Conditioned transport audit runner: 16 targets x 6 arms
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted projection-tree representatives
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.43, 3.29s
  NPR-d2-rare-gmm: L=64, kappa=1.52, 3.23s
  NPR-d2-correlated-t: L=64, kappa=1.42, 3.24s
  NPR-d2-nonlinear: L=64, kappa=1.44, 3.26s
  NPR-d4-balanced-gmm: L=64, kappa=2.41, 3.47s
  NPR-d4-rare-gmm: L=64, kappa=2.33, 3.47s
  NPR-d4-correlated-t: L=64, kappa=3.00, 3.50s
  NPR-d4-nonlinear: L=64, kappa=2.54, 3.52s
  NPR-d8-balanced-gmm: L=64, kappa=8.99, 4.55s
  NPR-d8-rare-gmm: L=64, kappa=8.28, 4.57s
  NPR-d8-correlated-t: L=64, kappa=8.04, 4.57s
  NPR-d8-nonlinear: L=64, kappa=7.97, 4.54s
  NPR-d16-balanced-gmm: L=192, kappa=19.42, 7.04s
  NPR-d16-rare-gmm: L=192, kappa=23.24, 7.19s
  NPR-d16-correlated-t: L=192, kappa=21.06, 7.42s
  NPR-d16-nonlinear: L=192, kappa=18.00, 7.16s
  cta-exact-global         ED2=0.017265 SW1=0.101907 train=0.150s setup+train=0.203s coverage>=0.500 local=0.000
  cta-kll-global           ED2=0.017473 SW1=0.093449 train=0.150s setup+train=0.193s coverage>=0.500 local=0.000
  cta-exact-hybrid         ED2=0.011872 SW1=0.086622 train=0.213s setup+train=0.263s coverage>=0.500 local=0.250
  cta-kll-hybrid           ED2=0.012207 SW1=0.086611 train=0.214s setup+train=0.257s coverage>=0.500 local=0.250
  cta-kll-guarded          ED2=0.012040 SW1=0.089698 train=0.262s setup+train=0.306s coverage>=0.500 local=0.179
  paper-neural-optimized   ED2=0.047305 SW1=0.149031 train=0.346s setup+train=0.346s coverage>=0.500 local=0.000
```
