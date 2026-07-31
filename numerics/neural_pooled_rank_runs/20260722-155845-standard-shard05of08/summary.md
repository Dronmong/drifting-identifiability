```text
Neural pooled-rank Phase-1 development
  registry 7b4cc89a68d73784303b0e2bc417d53b2f91f02511099c88c3c8e6be535a3dc6
  2 targets x 3 reps x 2 inits x 7 rows
  shard 5/8
  [PASS] complete Phase-0 regression suite
  [PASS] Torch paper field agrees with repository NumPy port
  [PASS] all registered training/held-out frames are tight
  [PASS] paired initial generator states are bitwise equal
  completed NPR-d4-rare-gmm in 45.18s
  completed NPR-d16-rare-gmm in 84.18s
  paper-neural                 ED2=0.067808 heldoutSW1=0.173066 div=0
  minibatch-sw                 ED2=0.067022 heldoutSW1=0.153604 div=0
  exact-atlas-rsr              ED2=0.068533 heldoutSW1=0.176350 div=0
  kll-atlas-rsr                ED2=0.068226 heldoutSW1=0.177578 div=0
  kll-small-rsr                ED2=0.070628 heldoutSW1=0.158831 div=0
  kll-paper-hybrid             ED2=0.055311 heldoutSW1=0.168048 div=0
  exact-free-particle-ceiling  ED2=0.018813 heldoutSW1=0.071238 div=0
```
