```text
Conditioned transport audit runner: 16 targets x 6 arms
  active directions per macro: 32
  local field calls: every macro-step
  local supports: dense 512-point batches
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.43, 3.37s
  NPR-d2-rare-gmm: L=64, kappa=1.52, 3.25s
  NPR-d2-correlated-t: L=64, kappa=1.42, 3.32s
  NPR-d2-nonlinear: L=64, kappa=1.44, 3.43s
  NPR-d4-balanced-gmm: L=64, kappa=2.41, 3.52s
  NPR-d4-rare-gmm: L=64, kappa=2.33, 3.47s
  NPR-d4-correlated-t: L=64, kappa=3.00, 3.56s
  NPR-d4-nonlinear: L=64, kappa=2.54, 3.61s
  NPR-d8-balanced-gmm: L=64, kappa=8.99, 4.55s
  NPR-d8-rare-gmm: L=64, kappa=8.28, 4.69s
  NPR-d8-correlated-t: L=64, kappa=8.04, 4.63s
  NPR-d8-nonlinear: L=64, kappa=7.97, 4.53s
  NPR-d16-balanced-gmm: L=192, kappa=19.42, 7.11s
  NPR-d16-rare-gmm: L=192, kappa=23.24, 7.08s
  NPR-d16-correlated-t: L=192, kappa=21.06, 7.02s
  NPR-d16-nonlinear: L=192, kappa=18.00, 7.11s
  cta-exact-global         ED2=0.017265 SW1=0.101907 train=0.153s setup+train=0.204s coverage>=0.500 local=0.000
  cta-kll-global           ED2=0.019480 SW1=0.102333 train=0.151s setup+train=0.197s coverage>=0.500 local=0.000
  cta-exact-hybrid         ED2=0.013932 SW1=0.092364 train=0.222s setup+train=0.273s coverage>=0.500 local=0.250
  cta-kll-hybrid           ED2=0.012674 SW1=0.084824 train=0.218s setup+train=0.262s coverage>=0.500 local=0.250
  cta-kll-guarded          ED2=0.013123 SW1=0.091751 train=0.272s setup+train=0.314s coverage>=0.500 local=0.165
  paper-neural-optimized   ED2=0.047305 SW1=0.149031 train=0.348s setup+train=0.348s coverage>=0.500 local=0.000
```
