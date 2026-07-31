```text
Conditioned transport audit runner: 4 targets x 3 arms
  registered arms: cta-exact-fixed-control, cta-exact-adaptive-rollout, paper-neural-optimized
  active directions per macro: 32
  local field calls: every macro-step
  local supports: 128 weighted variance-per-node representatives
  adaptive representative capacity: 128 below d=8, 256 at/above
  representative field audit calls per local arm: 1 (excluded from model kernel ledger)
  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite
  NPR-d2-balanced-gmm: L=64, kappa=1.45, 0.23s
  NPR-d4-rare-gmm: L=64, kappa=2.24, 0.24s
  NPR-d8-correlated-t: L=64, kappa=8.81, 0.33s
  NPR-d16-nonlinear: L=192, kappa=19.10, 0.50s
  cta-exact-fixed-control  ED2=0.070190 SW1=0.175326 train=0.035s setup+train=0.040s coverage>=0.500 local=0.250
  cta-exact-adaptive-rollout ED2=0.073434 SW1=0.176368 train=0.048s setup+train=0.061s coverage>=0.500 local=0.250
  paper-neural-optimized   ED2=0.125239 SW1=0.251257 train=0.032s setup+train=0.032s coverage>=0.500 local=0.000
```
