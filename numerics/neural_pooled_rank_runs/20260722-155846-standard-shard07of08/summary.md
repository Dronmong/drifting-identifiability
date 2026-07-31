```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  2 targets x 3 reps x 2 inits x 7 rows
  shard 7/8
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d4-nonlinear in 44.84s
  completed NPR-d16-nonlinear in 84.23s
  paper-neural                 ED2=0.065902 heldoutSW1=0.169197 div=0
  minibatch-sw                 ED2=0.077339 heldoutSW1=0.168754 div=0
  exact-atlas-rsr              ED2=0.063698 heldoutSW1=0.160310 div=0
  kll-atlas-rsr                ED2=0.062944 heldoutSW1=0.162279 div=0
  kll-small-rsr                ED2=0.074590 heldoutSW1=0.180959 div=0
  kll-paper-hybrid             ED2=0.055326 heldoutSW1=0.154160 div=0
  exact-free-particle-ceiling  ED2=0.016067 heldoutSW1=0.065320 div=0
```
