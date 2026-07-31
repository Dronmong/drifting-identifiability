```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  2 targets x 3 reps x 2 inits x 7 rows
  shard 0/8
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d2-balanced-gmm in 42.32s
  completed NPR-d8-balanced-gmm in 65.54s
  paper-neural                 ED2=0.028095 heldoutSW1=0.102818 div=0
  minibatch-sw                 ED2=0.053414 heldoutSW1=0.149721 div=0
  exact-atlas-rsr              ED2=0.075799 heldoutSW1=0.167802 div=0
  kll-atlas-rsr                ED2=0.074488 heldoutSW1=0.167509 div=0
  kll-small-rsr                ED2=0.051813 heldoutSW1=0.137140 div=0
  kll-paper-hybrid             ED2=0.089238 heldoutSW1=0.177366 div=0
  exact-free-particle-ceiling  ED2=0.006584 heldoutSW1=0.052905 div=0
```
