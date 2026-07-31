```text
Conditioned transport audit runner: 32 targets x 4 arms
  registered arms: cta-exact-adaptive-rollout, cta-exact-fixed-control, cta-exact-gated-hybrid, paper-neural-optimized
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 2 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NCT-d2-balanced-gmm-i0: L=64, kappa=1.51, 2.49s
  NCT-d2-balanced-gmm-i1: L=64, kappa=1.54, 2.44s
  NCT-d2-rare-gmm-i0: L=64, kappa=1.53, 2.38s
  NCT-d2-rare-gmm-i1: L=64, kappa=1.43, 2.39s
  NCT-d2-correlated-t-i0: L=64, kappa=1.77, 2.44s
  NCT-d2-correlated-t-i1: L=64, kappa=1.50, 2.43s
  NCT-d2-nonlinear-i0: L=64, kappa=1.53, 2.36s
  NCT-d2-nonlinear-i1: L=64, kappa=1.52, 2.40s
  NCT-d4-balanced-gmm-i0: L=64, kappa=2.72, 2.56s
  NCT-d4-balanced-gmm-i1: L=64, kappa=2.61, 2.58s
  NCT-d4-rare-gmm-i0: L=64, kappa=2.59, 2.65s
  NCT-d4-rare-gmm-i1: L=64, kappa=2.41, 2.56s
  NCT-d4-correlated-t-i0: L=64, kappa=2.31, 2.55s
  NCT-d4-correlated-t-i1: L=64, kappa=2.22, 2.58s
  NCT-d4-nonlinear-i0: L=64, kappa=2.44, 2.60s
  NCT-d4-nonlinear-i1: L=64, kappa=2.21, 2.66s
  NCT-d8-balanced-gmm-i0: L=64, kappa=10.69, 3.76s
  NCT-d8-balanced-gmm-i1: L=64, kappa=8.15, 3.74s
  NCT-d8-rare-gmm-i0: L=64, kappa=7.38, 3.73s
  NCT-d8-rare-gmm-i1: L=64, kappa=6.56, 3.72s
  NCT-d8-correlated-t-i0: L=64, kappa=8.28, 3.68s
  NCT-d8-correlated-t-i1: L=64, kappa=8.36, 3.71s
  NCT-d8-nonlinear-i0: L=64, kappa=8.00, 3.66s
  NCT-d8-nonlinear-i1: L=64, kappa=8.82, 3.71s
  NCT-d16-balanced-gmm-i0: L=192, kappa=18.43, 5.52s
  NCT-d16-balanced-gmm-i1: L=192, kappa=22.31, 5.50s
  NCT-d16-rare-gmm-i0: L=192, kappa=21.02, 5.55s
  NCT-d16-rare-gmm-i1: L=192, kappa=21.32, 5.46s
  NCT-d16-correlated-t-i0: L=192, kappa=22.16, 5.44s
  NCT-d16-correlated-t-i1: L=192, kappa=19.26, 5.47s
  NCT-d16-nonlinear-i0: L=192, kappa=20.52, 5.60s
  NCT-d16-nonlinear-i1: L=192, kappa=19.94, 5.33s
  cta-exact-adaptive-rollout ED2=0.012220 SW1=0.079118 train=0.412s setup+train=0.463s coverage>=0.500 local=0.250
  cta-exact-fixed-control  ED2=0.013010 SW1=0.085313 train=0.286s setup+train=0.337s coverage>=0.500 local=0.250
  cta-exact-gated-hybrid   ED2=0.013134 SW1=0.082174 train=0.389s setup+train=0.442s coverage>=0.500 local=0.250
  paper-neural-optimized   ED2=0.038028 SW1=0.139084 train=0.331s setup+train=0.331s coverage>=0.500 local=0.000
```
