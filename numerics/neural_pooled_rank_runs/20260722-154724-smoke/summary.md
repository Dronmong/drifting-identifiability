```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  4 targets x 1 reps x 1 inits x 7 rows
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d2-balanced-gmm in 1.66s
  completed NPR-d4-rare-gmm in 0.56s
  completed NPR-d8-correlated-t in 0.63s
  completed NPR-d16-nonlinear in 0.76s
  paper-neural                 ED2=0.120729 heldoutSW1=0.203680 div=0
  minibatch-sw                 ED2=0.101569 heldoutSW1=0.193502 div=0
  exact-atlas-rsr              ED2=0.152559 heldoutSW1=0.231389 div=0
  kll-atlas-rsr                ED2=0.148158 heldoutSW1=0.236103 div=0
  kll-small-rsr                ED2=0.096638 heldoutSW1=0.203667 div=0
  kll-paper-hybrid             ED2=0.160932 heldoutSW1=0.265160 div=0
  exact-free-particle-ceiling  ED2=0.021915 heldoutSW1=0.106145 div=0
```
