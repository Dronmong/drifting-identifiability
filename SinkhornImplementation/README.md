# Sinkhorn-balanced drifting (extension track)

**This directory is NOT part of the paper formalization.**  It develops and
evaluates a *proposed modification* of Algorithm 2 of "Generative Modeling via
Drifting" (arXiv:2602.04770v2), motivated by this repository's formal results.
Design rationale, math, and the resumable work plan: `PLAN.md`.

The idea in one line: the paper's affinity `A = sqrt(A_row * A_col)` is one
geometric-mean Sinkhorn balancing step of the kernel matrix; treat the
balancing depth `t` as a design dimension (`t = 1` is the paper,
`t -> inf` an entropic-OT coupling).

## Contents

| file | role |
|---|---|
| `PLAN.md` | goals, math, staged spec, resumption notes |
| `sinkhorn_drift.py` | balanced-affinity drift library (`iters = 1` reproduces the paper exactly) |
| `run_sinkhorn.py` | experiments S0-S4, writes `RESULTS.md` |
| `RESULTS.md` | generated report (seed 20260707) |
| `DriftingIdentifiability/SinkhornBalanced.lean` | certified theory (lives in the audited source tree, marked as extension) |
| `DriftingIdentifiability/BalancedSampling.lean` | certified `t = 2` balanced-affinity sampling theorem and full matrix/weight-form reconciliation |

Run:

```
uv run --with numpy python SinkhornImplementation/run_sinkhorn.py
```

## Certified theory (Lean crosswalk)

| declaration | statement |
|---|---|
| `interactionFrameBound_of_probeScaling` | per-probe positive scaling transfers a frame bound (`rhomin * c`); axiom-free |
| `interactionFrameBound_of_biScaling` | simultaneous per-probe/per-pair scaling (`rhomin * smin * c`); axiom-free |
| `inducedInteractionVector_sinkhornOrbit01_eq` | exact orbit rescaling identity `U_orbit(n) = u(x_n)^2 v(0) v(1) U_bare(n)`; axiom-free; verified numerically to 1e-17 (S4) |
| `sinkhornOrbit01Setup` / `sinkhornOrbit01_identifies_of_probeEnergy_eq_zero` | certified two-atom identifiability for **every** positive rescaling `u(x) k(x,y) v(y)` — the whole Sinkhorn orbit at once |
| `oneStepBalanced01Setup` / `oneStepBalanced01_identifies_of_probeEnergy_eq_zero` | the explicit one-full-balancing-step kernel `k/sqrt(r(x) g(y))` |
| `balancedTwoStepCentroid_deviation_prob_le` | high-probability deviation bound for the realized `t = 2` batch-balanced centroid, reducing batch-dependence to row-mass concentration plus fixed-weight SNIS |
| `balancedTwoStepNormalizedDrift_deviation_prob_le_of_centroids` | positive/negative centroid bounds compose into a normalized two-branch drift bound |
| `twoStepBalancedMatrixAffinity_eq_commonScale_mul_weight` | literal two-step finite matrix affinity equals a per-row common factor times the weight-form `twoStepWeight` |
| `twoStepBalancedMatrixCentroid_eq_weightCentroid` | the full matrix-form `t = 2` centroid equals the weight-form centroid used by the sampling theorem |
| `threeStepWeight_rel_of_rowMass_rel` | deterministic relative-error propagation for the row-cancelled `t = 3` weights from raw and first-balanced row-mass errors |
| `balancedThreeStepCentroid_deviation_prob_le_of_mass_tails` | fixed-depth `t = 3` centroid bridge assuming explicit raw/first-balanced row-mass tail bounds |
| `balancedThreeStepNormalizedDrift_deviation_prob_le_of_centroids` | positive/negative `t = 3` centroid bounds compose into a normalized two-branch drift bound |

Identifiability theorems use only the three reviewed equation-11/31/
antisymmetry paper axioms; no new axioms anywhere.

## Findings so far (toy scale; see RESULTS.md)

- **S0**: `iters = 1` reproduces the paper's estimator to `1e-16`.
- **S1**: balancing amplifies the field and the frame constant together;
  signal-normalized dispersion is flat-to-better at `t = 2-3` and degrades at
  `t = 10` for small `tau` — moderate depth is the sweet spot.
- **S2**: row/column mass CV falls with depth (at `tau = 0.2`, CV 0.94 at the
  paper's `t = 1` vs 0.03 at `t = 10`): the SNIS denominators become
  deterministic, removing the dominant certified-chain slack measured in
  `numerics/RESULTS.md` E5.
- **S3** (particle descent, paper's Figure-3 methodology, unequal masses
  0.3/0.7): balanced `t = 3` beats the paper's `t = 1` on mode-mass error in
  every initialization (between: 0.050 vs 0.065; collapsed: 0.638 vs 0.678);
  neither recovers from the far initialization at this step budget.
- **S4**: the Lean rescaling identity holds to `1e-17` on realized balanced
  matrices; the biscaling transfer bound is valid at every depth; the *direct*
  sharp constant grows ~18x along the orbit (balancing amplifies weak
  cross-mode interactions).
- **S6**: on the `t = 2` row-mass good event, the observed weight and centroid
  perturbations are far below the explicit Lean constants (`4 delta` and
  `16 delta R`) on the two-atom testbed.

## Honest limitations

- All empirical results are toy scale (1-D two-atom, 2-D particle descent);
  no FID/ImageNet claims are made or implied.
- The algorithm balances over the random batch. The Lean track certifies the
  orbit of *fixed* positive rescalings and now proves a first sampling theorem
  for the realized `t = 2` centroid by concentrating the random row masses.
  A fixed `t = 3` bridge is now formalized conditional on tail bounds for the
  first-balanced row masses; deriving those tails from primitive iid
  assumptions, and general `t >= 4` sampling theory, remain future work.
- The `t = 1` estimator's statistical theory (SNIS consistency etc.) is
  developed in the main track; the `t = 2` balanced centroid now has a
  high-probability bridge to a reference balanced population field, and two
  branch centroid bounds now assemble into a normalized drift bound. Full
  matrix reconciliation with the implementation form is certified for `t = 2`;
  the `t = 3` unrolling core is certified with explicit level-1 mass-tail
  hypotheses.
