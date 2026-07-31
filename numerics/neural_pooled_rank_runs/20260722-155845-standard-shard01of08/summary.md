```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  2 targets x 3 reps x 2 inits x 7 rows
  shard 1/8
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d2-rare-gmm in 42.08s
  completed NPR-d8-rare-gmm in 65.64s
  paper-neural                 ED2=0.062583 heldoutSW1=0.183518 div=0
  minibatch-sw                 ED2=0.046006 heldoutSW1=0.165721 div=0
  exact-atlas-rsr              ED2=0.045481 heldoutSW1=0.180763 div=0
  kll-atlas-rsr                ED2=0.043283 heldoutSW1=0.180277 div=0
  kll-small-rsr                ED2=0.039867 heldoutSW1=0.139517 div=0
  kll-paper-hybrid             ED2=0.044936 heldoutSW1=0.163702 div=0
  exact-free-particle-ceiling  ED2=0.004588 heldoutSW1=0.049118 div=0
```
