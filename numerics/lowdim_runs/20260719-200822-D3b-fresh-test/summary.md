```text
D3b test: frozen factorial candidate on fresh held-out targets
  invariant paper: translation/finite PASS
  invariant snis: translation/finite PASS
  invariant paper matched-batch cancellation: PASS
  invariant paper-ND vs driftlab.compute_v_paper: PASS
  tau-only  : ratio=1.000 hierCI=[1.0, 1.0]
  mask-only : ratio=1.000 hierCI=[0.6460113875681766, 1.0]
  step-only : ratio=1.000 hierCI=[1.0, 1.0]
  combined  : ratio=1.000 hierCI=[0.6460113875681766, 1.0]
  nonGaussian combined ratio=0.905 hierCI=[0.5577950190550448, 1.0]
  missing KM base=51 modified=53; degraded=0/14
  GATE D3b: FAIL
```
