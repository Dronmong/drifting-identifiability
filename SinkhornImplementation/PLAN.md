# Sinkhorn-balanced drifting: plan and goals

**STATUS: this document is the resumable specification.  If a session ends
mid-implementation, continue from the checklist at the bottom.**

## What this is (and is not)

This directory is an **extension track**.  Everything here goes *beyond* the
paper "Generative Modeling via Drifting" (arXiv:2602.04770v2): it proposes and
analyzes a **modification** of the paper's Algorithm 2, motivated by this
repository's formal results.  Nothing in this directory is a formalization of
a paper claim.  Files that formalize or evaluate the paper itself live in
`DriftingIdentifiability/` and `numerics/` and are documented there.

One exception for trust hygiene: the extension's Lean module lives at
`DriftingIdentifiability/SinkhornBalanced.lean` (with an explicit extension
header), NOT in this folder, because `scripts/TrustAudit.ps1` only scans the
package source tree and extension Lean code must remain under the
no-new-axioms discipline and the promoted-theorem audit.

## The insight

Algorithm 2 computes affinities `A = sqrt(A_row * A_col)`, where `A_row` is
the row softmax (over samples) and `A_col` is the column softmax (over
anchors) of the same logit matrix.  **This is exactly one geometric-mean
iteration of Sinkhorn-Knopp matrix balancing** applied to the positive kernel
matrix `K[i,s] = k(x_i, y_s)`.  The paper introduces the column softmax as an
unexplained empirical improvement ("slightly improves performance", Sec. 3.3);
this repository's Objective-4 analysis already proved the induced population
target is the *column-reweighted* field with kernel `k(x,y)/sqrt(g(y))` — the
population shadow of that half-step of balancing.

Proposal: treat the balancing depth as a design dimension.  Iterating the
geometric-mean balancing `t` times drives the affinity matrix toward a
doubly-stochastic coupling `D1 * K * D2` (diagonal positive scalings), i.e.
toward an **entropic optimal-transport plan** at temperature `tau`.

Why the formal stack predicts this helps:

1. **Certified constants improve monotonically toward balance.**  The single
   dominant slack in the certified finite-sample chain — measured in
   `numerics/RESULTS.md` E5 and patched by `DenominatorTail.lean` — is the
   randomness of the softmax denominators.  A balanced affinity matrix has
   (near-)uniform row and column masses: the random denominator disappears,
   `dmin` becomes deterministic and maximal, and the SNIS bounds tighten by
   orders of magnitude.
2. **Identifiability is invariant along the entire Sinkhorn orbit.**  Every
   balancing iterate has the form `u(x) * k(x,y) * v(y)` with positive `u, v`.
   Row factors `u` scale the interaction vectors per probe; column factors `v`
   scale them per strict pair.  Both scalings transfer a certified
   `InteractionFrameBound` with explicit constants (one transfer lemma already
   exists; the other is Stage L1 below).  So the *whole orbit* is certified at
   once — the algorithm modification costs no identifiability.
3. **External evidence:** minibatch-OT couplings are known to improve flow
   matching empirically; the paper's own ablation (column softmax helps FID)
   is the first point on this curve.

Honest scope caveats (state these in every write-up):

- The *algorithm's* balancing normalizes over the random batch, so the
  population balanced kernel depends on `p, q` through the batch law.  The
  Lean track avoids this by certifying the **whole orbit of positive diagonal
  rescalings of a fixed kernel** (which contains every batch-balanced iterate
  pointwise, whatever the scalings turn out to be).  Statements about the
  batch-dependent scalings themselves are numerics-track only.
- Nothing here claims an FID number.  The deliverable is: certified theory for
  the modified estimator + toy-scale evidence (particle descent, the paper's
  own Figure-3 methodology) + a precise experiment spec for a GPU test.

## Mathematical content

Fix the two-atom certified class (atoms `{0,1}`, kernel
`k = algorithm2Kernel tau`, anchors = probes).  For positive functions
`u, v : R -> R`, define `k_uv(x,y) := u(x) * k(x,y) * v(y)`.

* Two-atom interaction identity (from the kernel-generic
  `basisInteraction_empirical2`):

  ```text
  U^{uv}_01(x_n) = u(x_n)^2 * v(0) * v(1) * U^bare_01(x_n).
  ```

  Per-probe factor `u(x_n)^2`, per-pair factor `v(0)v(1)` — exactly the two
  scaling shapes with transfer lemmas.

* Frame transfers:
  - per strict pair: `interactionFrameBound_of_strictPairScaling`
    (already proved, `FiniteStability.lean`);
  - per probe (Stage L1, new): if `U'_p(n) = rho(n) . U_p(n)` with
    `rho(n) >= rhomin > 0` then `InteractionFrameBound U' (rhomin * c)`
    (sup-norm argument: each coordinate scales by at least `rhomin`).
  - combined (Stage L2): the biscaling transfer for `u^2` rows and
    `v(z_i)v(z_j)` pair columns.

* For `m = 2` the sharp route is even simpler: `k_uv > 0` pointwise makes
  `U^{uv}_01` nonzero, so `interactionFrameBound_two` certifies the sharp
  constant `||U^{uv}_01||` directly; the identity above converts it to the
  bare constant exactly.

* One-step balanced corollary: with `r(x) := k(x,0) + k(x,1)` (row mass over
  the atom support) and `g(y) := sum_r k(anchor_r, y)`
  (`algorithm2ColumnKernelMass`), the kernel
  `k1(x,y) := k(x,y) / sqrt(r(x) * g(y))` is the population shadow of one full
  geometric-mean balancing step of the anchor-by-atom matrix; instantiate
  `u = 1/sqrt(r)`, `v = 1/sqrt(g)` (both positive, measurable).

## Stages and deliverables

### Lean (`DriftingIdentifiability/SinkhornBalanced.lean`)

- **L1** `interactionFrameBound_of_probeScaling`: per-probe positive scaling
  transfers a frame bound with constant `rhomin * c`.  Proof via
  `pi_norm_le_iff_of_nonneg` / `norm_le_pi_norm`-style sup-norm bookkeeping;
  mirror the style of `interactionFrameBound_of_strictPairScaling`.
- **L2** `interactionFrameBound_of_biScaling`: `U'_p(n) = rho(n) * s(p) * U_p(n)`
  with `rho >= rhomin > 0`, `|s(p)| >= smin > 0` gives constant
  `rhomin * smin * c`.  (Compose L1 with the existing strict-pair lemma.)
- **L3** `sinkhornOrbit01Setup`: a certified `PopulationMeanShiftFiniteSetup`
  for kernel `fun x y => u x * algorithm2Kernel tau x y * v y`, hypotheses
  `hu : forall x, 0 < u x`, `hv : forall y, 0 < v y`, `Measurable u/v`.
  Reuse: `kernelNormalizer_empirical01` (+ `_pos`) from
  `ColumnReweightedTwoAtom.lean` for normalizers; `empirical01Basis_integrable*`
  for integrability; `interactionFrameBound_two` +
  `basisInteraction_empirical2` positivity for the frame.  Promoted theorems:
  `sinkhornOrbit01_identifies_of_probeEnergy_eq_zero`,
  `sinkhornOrbit01_coefficientStability` (abstract normalizer bound `B`), and
  the exact rescaling identity
  `inducedInteractionVector_sinkhornOrbit01_eq_smul` (per-probe form).
- **L4** one-step balanced corollary: instantiate `u = (sqrt (r x))⁻¹`,
  `v = (sqrt (g y))⁻¹` with the positivity lemmas already available
  (`algorithm2ColumnKernelMass_pos`; row-mass analogue is a two-term positive
  sum).  Also state the mass-uniformity remark as documentation only.
- **Audit**: no new axioms.  All promoted names registered in
  `scripts/AxiomAudit.ps1`.  `scripts/Check.ps1` green.  `#print axioms` on
  the identifiability theorem must show only the reviewed equation-11/31/
  antisymmetry axioms; the scaling lemmas must be axiom-free.

### Python (`SinkhornImplementation/`)

- **P1** `sinkhorn_drift.py`: library.
  - `balanced_affinity(logit, iters)`: geometric-mean balancing;
    `iters = 1` MUST reproduce Algorithm 2's `sqrt(A_row * A_col)` exactly
    (assert against `numerics/driftlab.py::compute_v_paper`).
  - `compute_v_sinkhorn(x, ypos, yneg, T, iters, self_mask)`: drift with
    balanced affinities and the paper's `W_pos = A_pos * A_neg.sum` recipe.
  - `beta_drift(...)`: the `k / g(y)^beta` one-parameter family
    (`beta = 0` bare, `= 1/2` paper) for the population-target comparison.
  - Diagnostics: row/column mass coefficient of variation, per-anchor ESS,
    empirical `dmin`, mass-product spread.
- **P2** `run_sinkhorn.py` writing `SinkhornImplementation/RESULTS.md`:
  - **S1** estimator dispersion vs balancing depth `t` in {0 (row-only), 1
    (paper), 2, 3, 5, 10} at `tau in {0.02, 0.05, 0.2}`, `N = 64`: Monte-Carlo
    variance of the drift/centroid statistic on the two-atom model; predicted:
    monotone decrease, largest gain at small `tau`.
  - **S2** mass uniformization: row/col mass CV and min/mean ratio vs `t`
    (the certified `dmin` dial); predicted: CV -> 0 geometrically.
  - **S3** particle-descent toy (the paper's Figure-3 methodology, 2-D
    bimodal target, three initializations incl. collapsed): particles moved by
    the estimator field directly (no network); compare `t = 1` vs `t >= 2` on
    mode-mass error vs iterations and on wall-clock per step; also `beta`
    sweep.  This is the "match/exceed at toy scale" test.
  - **S4** certified constants along the orbit: for the realized balanced
    matrices, extract `u, v` scalings, apply the L2 transfer formula, compare
    with directly computed `c_cert` of the balanced interaction matrix
    (transfer must lower-bound direct).
- Style: numpy only, deterministic seeds, ASCII output, run via
  `uv run --with numpy python SinkhornImplementation/run_sinkhorn.py`.

### Documentation

- `SinkhornImplementation/README.md`: what this is, NOT-the-paper marker, how
  to run, crosswalk of new Lean names.
- `DriftingIdentifiability/ResearchStatus.md`: new top-level section
  "Extension track: Sinkhorn-balanced drifting" (clearly outside Objectives
  1-7), summarizing theorems + numerics findings.
- Memory update.

## Checklist (tick as completed)

- [x] PLAN.md committed
- [x] L1 probe-scaling transfer proved
- [x] L2 biscaling transfer proved
- [x] L3 orbit setup + identifiability + stability + rescaling identity
- [x] L4 one-step balanced corollary
- [x] AxiomAudit entries + Check.ps1 green + #print axioms clean
- [x] P1 library (+ agreement assert vs driftlab at iters = 1)
- [x] P2 S1-S4 experiments + RESULTS.md
- [ ] README + ResearchStatus extension section + memory
- [ ] Final commit(s)

## Resumption notes

- Reuse targets (existing, verified): `interactionFrameBound_of_strictPairScaling`
  (FiniteStability), `interactionFrameBound_two`, `basisInteraction_empirical2`,
  `empirical2`/`empirical01Basis`/`empirical01Basis_integrable_prod`
  (EmpiricalFrameBound), `kernelNormalizer_empirical01`/`_pos`/`_le_one` and
  `algorithm2Kernel_le_one` (ColumnReweightedTwoAtom, namespace
  `PaperFiniteIdentifiability`), `algorithm2ColumnKernelMass(_pos)`
  (Algorithm2SNIS, namespace `Algorithm2`), `PopulationMeanShiftFiniteSetup`
  (PopulationIdentifiability).
- The two-atom integrands are handled by measurability (atomic reference), not
  continuity: mirror `columnReweighted01Setup`'s `basisInteractionIntegrable`
  and `interactionIntegrable` fields with `Measurable u/v` hypotheses.
- Python must not import torch; keep it numpy-only like `numerics/`.
- Mark every new artifact "extension — not part of the paper" in its header.
