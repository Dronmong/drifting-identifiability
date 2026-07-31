```text
Conditioned transport audit runner: 16 targets x 6 arms
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted projection-tree representatives
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.43, 3.28s
  NPR-d2-rare-gmm: L=64, kappa=1.52, 3.20s
  NPR-d2-correlated-t: L=64, kappa=1.42, 3.20s
  NPR-d2-nonlinear: L=64, kappa=1.44, 3.27s
  NPR-d4-balanced-gmm: L=64, kappa=2.41, 3.51s
  NPR-d4-rare-gmm: L=64, kappa=2.33, 3.55s
  NPR-d4-correlated-t: L=64, kappa=3.00, 3.62s
  NPR-d4-nonlinear: L=64, kappa=2.54, 3.66s
  NPR-d8-balanced-gmm: L=64, kappa=8.99, 4.49s
  NPR-d8-rare-gmm: L=64, kappa=8.28, 4.63s
  NPR-d8-correlated-t: L=64, kappa=8.04, 4.57s
  NPR-d8-nonlinear: L=64, kappa=7.97, 4.52s
  NPR-d16-balanced-gmm: L=192, kappa=19.42, 6.85s
  NPR-d16-rare-gmm: L=192, kappa=23.24, 7.11s
  NPR-d16-correlated-t: L=192, kappa=21.06, 6.95s
  NPR-d16-nonlinear: L=192, kappa=18.00, 7.76s
  cta-exact-global         ED2=0.017265 SW1=0.101907 train=0.151s setup+train=0.202s coverage>=nan local=0.000
  cta-kll-global           ED2=0.017631 SW1=0.092553 train=0.152s setup+train=0.198s coverage>=nan local=0.000
  cta-exact-hybrid         ED2=0.011872 SW1=0.086622 train=0.213s setup+train=0.264s coverage>=nan local=0.250
  cta-kll-hybrid           ED2=0.013682 SW1=0.087672 train=0.216s setup+train=0.259s coverage>=nan local=0.250
  cta-kll-guarded          ED2=0.013203 SW1=0.089299 train=0.264s setup+train=0.307s coverage>=nan local=0.179
  paper-neural-optimized   ED2=0.047305 SW1=0.149031 train=0.350s setup+train=0.350s coverage>=nan local=0.000
```
