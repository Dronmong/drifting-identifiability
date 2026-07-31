```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  2 targets x 3 reps x 2 inits x 7 rows
  shard 6/8
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d4-correlated-t in 44.63s
  completed NPR-d16-correlated-t in 84.19s
  paper-neural                 ED2=0.042609 heldoutSW1=0.135159 div=0
  minibatch-sw                 ED2=0.069772 heldoutSW1=0.142647 div=0
  exact-atlas-rsr              ED2=0.045984 heldoutSW1=0.133204 div=0
  kll-atlas-rsr                ED2=0.044957 heldoutSW1=0.127513 div=0
  kll-small-rsr                ED2=0.055388 heldoutSW1=0.136873 div=0
  kll-paper-hybrid             ED2=0.054418 heldoutSW1=0.134376 div=0
  exact-free-particle-ceiling  ED2=0.019929 heldoutSW1=0.068112 div=0
```
