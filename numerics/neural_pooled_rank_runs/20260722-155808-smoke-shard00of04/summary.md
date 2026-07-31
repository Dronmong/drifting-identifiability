```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  1 targets x 1 reps x 1 inits x 7 rows
  shard 0/4
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d2-balanced-gmm in 1.72s
  paper-neural                 ED2=0.077607 heldoutSW1=0.186498 div=0
  minibatch-sw                 ED2=0.063726 heldoutSW1=0.201914 div=0
  exact-atlas-rsr              ED2=0.185445 heldoutSW1=0.377742 div=0
  kll-atlas-rsr                ED2=0.177086 heldoutSW1=0.350008 div=0
  kll-small-rsr                ED2=0.118243 heldoutSW1=0.278803 div=0
  kll-paper-hybrid             ED2=0.199126 heldoutSW1=0.373309 div=0
  exact-free-particle-ceiling  ED2=0.002139 heldoutSW1=0.031185 div=0
```
