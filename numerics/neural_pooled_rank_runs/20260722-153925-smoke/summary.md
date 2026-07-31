```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  4 targets x 1 reps x 1 inits x 7 rows
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d2-balanced-gmm in 6.11s
  completed NPR-d4-rare-gmm in 0.60s
  completed NPR-d8-correlated-t in 0.65s
  completed NPR-d16-nonlinear in 0.76s
  paper-neural                 ED2=0.236559 heldoutSW1=0.392356 div=0
  minibatch-sw                 ED2=0.161311 heldoutSW1=0.257669 div=0
  exact-atlas-rsr              ED2=0.914802 heldoutSW1=0.669288 div=0
  kll-atlas-rsr                ED2=0.921221 heldoutSW1=0.670850 div=0
  kll-small-rsr                ED2=0.275769 heldoutSW1=0.388423 div=0
  kll-paper-hybrid             ED2=0.924325 heldoutSW1=0.672121 div=0
  exact-free-particle-ceiling  ED2=0.021915 heldoutSW1=0.106145 div=0
```
