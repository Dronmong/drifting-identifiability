# Coherent-transport implementation audit

Date: 2026-07-24

Scope:

- `est_plan.py`
- `metrics.py`
- `targets.py`
- `partial_controller.py`
- `run_coherent_particle_screen.py`
- `run_partial_controller_development.py`
- `Stage1A_Results.md`
- `Stage1B_Results.md`

This audit checks the current effort against
[`../AnchoredCoherentTransportResearchPlan.md`](../AnchoredCoherentTransportResearchPlan.md).

## Verdict

The implementation contains one correct and reusable mathematical core:

> `est_coupling(Pi, N)` is the average of sliced permutation couplings and has
> exact empirical row and column marginals \(1/N\).

The two reported roadmap gates are not established.

- Stage 1A used a row-wise modal route as its winning "coherent" arm. That
  route chooses one target identity per source but is not balanced and can
  badly distort the target marginal.
- The balanced-bijection numbers in the Stage 1B report have no source file,
  registry, stdout, or raw results artifact.
- The implemented Stage 1B controller is a nearest-neighbor deficit filler,
  not the registered persistent-quantile surplus/deficit controller.
- Planning and controller data are not split.
- The geometry guard evaluates direct moves using the same planning targets
  that receive those moves.

The correct checkpoint is therefore:

1. EST coupling primitive: verified;
2. barycentric EST failure control: useful;
3. row-wise modal routing: diagnostic only;
4. balanced EST route: now implemented and tested;
5. Stage 1 promotion: open;
6. Stage 1B partial controller: open;
7. neural amortization: not authorized by a passed particle gate.

## Findings

### A1. `est_coupling` is correct

For every direction, rank matching returns a permutation. The normalized
permutation matrix has row and column masses \(1/N\), and averaging these
matrices preserves both marginals.

The unit tests reproduce this exactly.

Severity: none.

Disposition: retain.

### A2. Row-wise modal routing was misclassified as a balanced plan

`hard_consensus` independently selected the most frequent target index in
every source row. This prevents barycentric averaging of target identities,
but it allows several sources to select one target and leaves other targets
unused.

Reproduced example at \(N=256,L=32\):

- unique targets used: `171 / 256`;
- maximum sources assigned to one target: `5`;
- target-marginal L1 error: `0.6640625`.

Severity: critical for the Stage 1A interpretation.

Repair:

- renamed semantically to `modal_route`;
- kept `hard_consensus` only as a documented compatibility alias;
- added `target_marginal_l1`;
- added regression tests showing that modal routing may be unbalanced.

### A3. The balanced-bijection result was not reproducible

`Stage1B_Results.md` reports a dense \(O(N^3)\) balanced assignment but no
implementation or raw artifact exists under `numerics/coherent_transport`.

Severity: critical for the claimed strongest particle teacher.

Repair:

- implemented `balanced_est_assignment`;
- dense mode uses Hungarian assignment;
- sparse mode solves full bipartite matching only on the union of EST edges;
- both maximize total sliced agreement;
- both are tested to return a permutation supported by EST;
- both attain the same consensus objective in the property test.

### A4. Full-bijection endpoint quality is partly tautological

With equal source and planning-target counts, any permutation assignment has
the planning target empirical measure as its exact endpoint. Therefore:

- endpoint ED2/support quality cannot by itself show that EST found a better
  route;
- a random bijection and an optimal Euclidean bijection have the same endpoint
  empirical distribution;
- the relevant mechanism comparisons are route length, sliced agreement,
  assignment stability, sparsity, solve cost, and quality during a bounded
  update or after neural amortization.

Severity: critical for experiment design, not a code error.

Repair:

`run_balanced_plan_audit.py` includes random-bijection and
minimum-Euclidean-cost controls and treats endpoint metrics as secondary.

### A5. The checkerboard leakage metric ignored the domain

The previous metric checked only parity of `floor(x)`. Points outside the
declared checkerboard domain could land in an even-parity cell and be counted
as valid.

Example:

```text
[-0.5, -0.5] and [4.5, 4.5]
```

were not counted as leakage for a four-cell checkerboard.

Severity: high for geometry claims.

Repair:

- leakage now requires both in-domain membership and ON-cell parity;
- an out-of-domain regression test was added.

### A6. The support threshold was not independently calibrated

`precision_recall` estimates local radii from the same evaluation target
sample. This is useful as a standard diagnostic, but it does not implement the
roadmap's independent target-versus-target threshold rule.

Severity: medium for confirmation; acceptable as a clearly labeled auxiliary
metric.

Repair:

- added `calibrated_support_radius`;
- added `calibrated_precision_coverage`;
- the corrected route audit uses two disjoint calibration pools and a third
  evaluation pool.

### A7. Stage 1B did not test its registered hypothesis

The roadmap calls for directional CDF deficits derived from persistent target
quantiles/KLL state. The current controller instead:

1. finds uncovered planning samples with a k-NN radius;
2. ranks generated particles by nearest-target distance;
3. moves particles directly to uncovered planning points.

It therefore does not test whether persistent sliced deficits can select the
right transported mass.

Severity: critical for accepting or rejecting RQ2/H2.

Disposition:

- retain `partial_controller.py` only as a nearest-neighbor diagnostic;
- do not call Gate 1B passed or failed from the current output;
- implement the registered controller only after the corrected route gate.

### A8. Controller and planning pools were reused

Stage 1B uses one target set `Y` for the local drift, deficit construction,
destinations, and support guard. A move directly onto `Y` is almost guaranteed
to look safe under a support metric centered on `Y`.

Severity: critical for the geometry guard.

Repair requirement:

- planning targets propose routes;
- an independent controller pool accepts/rejects them;
- a final evaluation pool is never read during the update.

The corrected route audit already creates disjoint planning,
calibration-A, calibration-B, and evaluation pools.

### A9. The rare-mixture probabilities were not the documented values

The previous constructor set raw weights `0.02` and `0.05` beside seven raw
weights of `1.0`, then normalized. The resulting probabilities were about
`0.25%` and `0.62%`, not `2%` and `5%`.

Severity: high for interpreting rare-mode results.

Repair:

- the constructor now assigns exact probabilities `2%` and `5%`;
- the remaining mass is divided across other modes.

Prior `sep_modes_rare` numbers must be interpreted under the old, much rarer
distribution and are not directly comparable to new runs.

### A10. The structured suite omitted Swiss roll

`swiss_roll` existed but was not returned by `suite()`.

Severity: medium, because Swiss roll was one of the motivating geometry
failures.

Repair: added it to the suite.

### A11. The paper arm is not the official paper demonstration

The screen uses free particles, a fixed planning set, `tau=0.2`, 200 direct
updates, and hand-declared gains. It is a local Algorithm-2 field control, not
the official neural checkerboard protocol.

Severity: critical for any paper-performance claim.

Disposition:

- label it `free-particle local drift`, not "official paper";
- use the official-style neural baseline only in the later fair model
  comparison;
- do not use the Stage 1 paper column to claim superiority over the paper.

### A12. The Sinkhorn arm is only an untuned local reference

It uses a fixed epsilon, 60 scaling iterations, and a direct barycentric map.
It is not a faithful implementation of W-Flow, Sinkhorn-Drifting, or a tuned
dense OT baseline.

Severity: medium if correctly labeled; critical if attributed to published
methods.

Disposition: retain only as an untuned dense reference.

## Tests run

The following pass after repair:

```text
uv run --with numpy --with scipy python numerics/coherent_transport/est_plan.py
uv run --with numpy --with scipy python numerics/coherent_transport/metrics.py
uv run --with numpy --with scipy python numerics/coherent_transport/partial_controller.py
```

The EST suite now checks:

- sliced permutations;
- exact averaged-plan marginals;
- identity behavior;
- deterministic ties;
- modal-route imbalance;
- dense/sparse balanced target marginals;
- EST-support membership;
- equal dense/sparse consensus objective.

## Correct next checkpoint

Run the corrected balanced route audit and decide whether sparse EST assignment
provides a useful route relative to:

- minimum-Euclidean-cost assignment;
- random bijection;
- row-wise modal routing;
- EST barycenter.

The audit must not promote a method merely because its full-bijection endpoint
copies the planning empirical measure. If sparse EST has acceptable route
cost, exact marginals, substantially higher sliced agreement, and lower solve
cost than dense Euclidean assignment, it becomes a plausible neural teacher.
The next meaningful gate is then neural route retention, not another
free-particle endpoint victory.
