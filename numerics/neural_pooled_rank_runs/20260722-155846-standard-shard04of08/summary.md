```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  2 targets x 3 reps x 2 inits x 7 rows
  shard 4/8
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d4-balanced-gmm in 44.85s
  completed NPR-d16-balanced-gmm in 83.93s
  paper-neural                 ED2=0.043284 heldoutSW1=0.115699 div=0
  minibatch-sw                 ED2=0.091425 heldoutSW1=0.169913 div=0
  exact-atlas-rsr              ED2=0.100030 heldoutSW1=0.194970 div=0
  kll-atlas-rsr                ED2=0.097606 heldoutSW1=0.195197 div=0
  kll-small-rsr                ED2=0.087942 heldoutSW1=0.167902 div=0
  kll-paper-hybrid             ED2=0.094785 heldoutSW1=0.195195 div=0
  exact-free-particle-ceiling  ED2=0.047454 heldoutSW1=0.095203 div=0
```
