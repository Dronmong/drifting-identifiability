# Balanced-affinity sampling theory (open items 1+2): plan

**STATUS: resumable specification.  If the session ends mid-implementation,
continue from the checklist; every lemma below has its full proof sketch.**

Extension track (not a paper claim).  Parent plan: `PLAN.md`.  Goal: the
statistical theorem that was left open for Sinkhorn depth `t >= 2` — the
balanced affinity's weights depend on the whole batch, so the fixed-weight
SNIS theorems do not apply directly.

## Structural insight that makes this tractable

Fix anchors `x_1..x_M` and iid samples `Y_1..Y_N`.  Write `k_j(y) :=
algorithm2Kernel tau (x_j) y`, `g(y) := sum_j k_j(y)` (column mass over
anchors — a *fixed* function), and `r_j := sum_s k_j(Y_s)` (row masses —
random sums over samples).

After two geometric-mean balancing steps, the affinity at anchor `i`, sample
`s` factors as `(per-anchor factor) * W_s`, where the per-anchor factor
cancels in the self-normalized centroid and

```text
W_s = k_i(Y_s) / ( q(Y_s) * sqrt( h(Y_s, r) ) ),
q(y)    := sqrt( sqrt( g(y) ) ),
h(y, r) := sum_j k_j(y) / sqrt(r_j).
```

**The entire batch-coupling of the `t = 2` weights is mediated by the
M-dimensional random vector `r` of row masses** — and each `r_j` is a sum of
bounded independent kernel values, exactly the object the existing
`weightSum` machinery concentrates.  Replacing `r` by its (caller-supplied)
mean vector `Mbar` defines the *reference weight* `Wbar(y)` — a fixed
measurable function to which the existing indexed SNIS deviation theorem
applies verbatim.

So the theorem decomposes: (concentration of `r`) + (deterministic relative
perturbation `r -> weights -> centroid`) + (existing SNIS theorem at `Wbar`).

## Lemma chain (all in `DriftingIdentifiability/BalancedSampling.lean`)

### B0. Two-sided weight-sum deviation (namespace `SelfNormalized`)

`weightSum_deviation_prob_le`: same hypotheses as `weightSum_lower_tail_prob_le`
(DenominatorTail.lean), conclusion

```text
P{ |sum_l w_l(Y_l) - sum_l mu_w l| > t } <= N * sigma_w^2 / t^2.
```

Proof: `D - M = sum_l W'_l` exactly (centered weights), so the event *equals*
`{ t < |sum W'| }`; then the sample-mean axiom + `meas_gt_le_meanSquare_div`,
identical to the lower-tail proof (event inclusion step becomes an equality;
reuse the file's scaffolding wholesale).

### B1. Deterministic centroid perturbation (namespace `SelfNormalized`)

`selfNormalizedCentroid_relative_perturbation`: for finite `iota` (Nonempty),
weights `W Wbar : iota -> R` with `Wbar s > 0`, relative error
`|W s - Wbar s| <= eta * Wbar s` with `0 <= eta < 1`, and points `y : iota -> E`
with pairwise diameter `D` (`forall s s', ||y s - y s'|| <= D`):

```text
|| c(W) - c(Wbar) || <= eta * D / (1 - eta),
c(W) := (sum W)^{-1} . sum W_s . y_s .
```

Proof sketch (all elementary):
1. sub-lemma `centroid_mem_diameter`: `||y s - c(Wbar)|| <= D` because the
   centroid is a convex combination: `y s - c(Wbar) = (sum Wbar)^{-1} .
   sum_t Wbar_t . (y s - y t)`, triangle inequality.
2. `c(W) - c(Wbar) = (sum W)^{-1} . sum_s W_s . (y_s - c(Wbar))` (same
   algebra as `hpoint` in the SNIS proofs).
3. numerator: `sum_s Wbar_s . (y_s - c(Wbar)) = 0` (definition of centroid),
   so it equals `sum_s (W_s - Wbar_s) . (y_s - c(Wbar))`, norm
   `<= eta * (sum Wbar) * D`.
4. denominator: `sum W >= (1 - eta) * sum Wbar > 0`.

### B2. Relative-error propagation toolkit (namespace `SelfNormalized`)

- B2a `abs_inv_sqrt_sub_inv_sqrt_le`: if `abar > 0`, `|a - abar| <= eta*abar`,
  `eta <= 1/2`, then `|1/sqrt a - 1/sqrt abar| <= 2*eta/sqrt abar`.
  Proof: `|sqrt abar - sqrt a| <= |abar - a|/(sqrt a + sqrt abar) <=
  eta*abar/sqrt abar = eta*sqrt abar`; then
  `|1/sqrt a - 1/sqrt abar| = |sqrt abar - sqrt a|/(sqrt a*sqrt abar)
  <= eta*sqrt abar/(sqrt a*sqrt abar) = eta/sqrt a`, and
  `sqrt a >= sqrt((1-eta)abar) >= sqrt(abar/2) = sqrt abar/sqrt 2`, so the
  bound is `eta*sqrt 2/sqrt abar <= 2*eta/sqrt abar`.  (Constants explicit,
  not optimized — document.)
- B2b `abs_sum_sub_sum_le_of_rel`: positive coefficients `k_j > 0`, values
  with per-index relative error `eta` give the weighted sum relative error
  `eta` (triangle inequality; here applied with values `1/sqrt r_j` vs
  `1/sqrt Mbar_j`, i.e. `|h(y,r) - hbar(y)| <= 2*eta*hbar(y)` from B2a).
- B2c `twoStepWeight_rel_of_rowMass_rel`: if `|r_j - Mbar_j| <= delta*Mbar_j`
  for all `j` with `delta <= 1/4`, then for every `y`
  `|W(y,r) - Wbar(y)| <= 4*delta*Wbar(y)`.
  Proof: B2a+B2b give `h` within `2*delta` relative; B2a again (at
  `eta = 2*delta <= 1/2`) gives `1/sqrt h` within `4*delta` relative; the
  factors `k_i(y)/q(y)` are common and positive.

### B3. Definitions + the headline theorem (namespace `Algorithm2`)

Defs (all finite, explicit):

```text
balancedRefLevelMass anchors tau Mbar y := sum_j k_j(y) * (sqrt (Mbar j))⁻¹     -- hbar / h
twoStepWeight anchors tau i Mbar y :=
  k_i(y) / ( sqrt (sqrt (g y)) * sqrt (balancedRefLevelMass anchors tau Mbar y) )
-- realized weights = twoStepWeight at the random row-mass vector
--   r(omega) j := sum_s k_j(Y_s omega);
-- reference weights = twoStepWeight at the caller-supplied mean vector Mbar.
```

Headline `balancedTwoStepCentroid_deviation_prob_le` (hypotheses:
probability space, iid-free per-slot structure as in the existing indexed
theorems; anchors; analysis anchor `i`; kernel positivity is automatic;
sample radius `R` (diameter `D := 2R`); `delta <= 1/4` with
`hMbar : forall j, 0 < Mbar j`; per-slot row-kernel means
`hrow : forall j s, integral of k_j(Y_s) = mu_{j,s}` with
`Mbar j = sum_s mu_{j,s}` and per-slot variances `<= sigma_row^2`;
SNIS data for the *reference* weight `Wbar := twoStepWeight ... Mbar`
exactly as in `selfNormalizedIndexed_deviation_prob_le`
(means `mu`, bias `b`, second moments `sigma`, weight-mean data
`mu_w, sigma_w`, split `t_w`)):

```text
P{ eps + 8*delta*D  <  || twoStepCentroid(omega) - c || }
  <=  [SNIS deviation bound for Wbar at eps]                (existing theorem)
   +  sum_j  N * sigma_row^2 / (delta * Mbar j)^2           (B0, union bound)
```

Proof skeleton:
1. good event `G := forall j, |r_j(omega) - Mbar j| <= delta * Mbar j`;
   `P(G^c) <= sum_j ...` by B0 + `measure_biUnion_finset_le`.
2. on `G`: B2c gives realized weights within `4*delta` of `Wbar(Y_s)`; B1 at
   `eta = 4*delta <= 1/2` gives
   `||c_realized - c_ref|| <= 4*delta*D/(1-4*delta) <= 8*delta*D`.
3. event inclusion:
   `{eps + 8 delta D < ||c_realized - c||} ⊆
    {eps < ||c_ref - c||} ∪ G^c`   (triangle inequality on `G`).
4. subadditivity + the existing `selfNormalizedIndexed_deviation_prob_le`
   for the first event.

The realized two-step centroid is *defined* in weight form
(`(sum_s W_s)^{-1} . sum_s W_s . Y_s` with the realized `r(omega)`); the
per-anchor balancing factor `1/sqrt(r_i * r1_i)` cancels by
`selfNormalizedCentroid_eq_of_common_scale`, exactly as at `t = 1`.

### B4 (optional, if budget remains). Full-matrix reconciliation

Define one `balancedStep` of a finite anchor-by-sample matrix and prove the
`t = 2` affinity centroid equals the weight-form centroid (entrywise
factorization `K2[i,s] = (1/sqrt(r_i r1_i)) * W_s`, positivity of all masses,
then common-scale cancellation).  Numerics S0/S4 already validate this
algebra; B4 upgrades it to a theorem.

Implemented in `BalancedSampling.lean` as:

- `twoStepBalancedMatrixAffinity_eq_commonScale_mul_weight`: the literal
  two-step matrix affinity is a per-row common factor times `twoStepWeight`;
- `twoStepBalancedMatrixCentroid_eq_weightCentroid`: the per-row common factor
  cancels, so the full matrix-form centroid equals the weight-form centroid
  used by the B3 sampling theorem.

### B5. Fixed `t = 3` unrolling core

Implemented the next finite unrolling layer without claiming a uniform
all-depth theorem:

- `twoStepLevelMass`, `secondBalancedColumnProfile`, and `threeStepWeight`
  express the row-cancelled `t = 3` centroid weights.
- `twoStepLevelMass_rel_of_rowMass_rel`,
  `secondBalancedColumnProfile_rel_of_rowMass_rel`, and
  `threeStepWeight_rel_of_rowMass_rel` prove deterministic relative-error
  propagation through the extra balancing level (`32*delta`, deliberately
  loose).
- `balancedThreeStepCentroid_deviation_prob_le_of_mass_tails` gives the
  high-probability bridge for the realized `t = 3` centroid, assuming explicit
  tails for both raw row masses and first-balanced row masses.
- `balancedThreeStepNormalizedDrift_deviation_prob_le_of_centroids` assembles
  positive/negative `t = 3` centroid bounds into a normalized drift bound.

Honest residual: the primitive iid proof of the first-balanced row-mass tails
is still open.  This is now isolated as a concentration lemma rather than
entangled with the `t = 3` algebra.

## Audit and docs

- No new axioms anywhere (B0 uses the reviewed sample-mean axiom via the
  existing scaffolding; B1/B2 are axiom-free; B3 composes existing theorems).
- Register promoted names in `scripts/AxiomAudit.ps1`:
  `SelfNormalized.weightSum_deviation_prob_le`,
  `SelfNormalized.selfNormalizedCentroid_relative_perturbation`,
  `SelfNormalized.twoStepWeight_rel_of_rowMass_rel` (name may live in
  `Algorithm2`),
  `Algorithm2.twoStepBalancedMatrixAffinity_eq_commonScale_mul_weight`,
  `Algorithm2.twoStepBalancedMatrixCentroid_eq_weightCentroid`,
  `Algorithm2.threeStepWeight_rel_of_rowMass_rel`,
  `Algorithm2.balancedThreeStepCentroid_deviation_prob_le_of_mass_tails`,
  `Algorithm2.balancedTwoStepCentroid_deviation_prob_le`.
- `scripts/Check.ps1` green; `#print axioms`: B1/B2 foundations-only, B0/B3
  foundations + `sampleMean_meanSquare_le`.
- Numerics (cheap, optional): S6 in `run_sinkhorn.py` — Monte-Carlo check
  that realized `t = 2` weights lie within `4*delta` of reference weights
  when row masses lie within `delta` (validates the B2c constant), and that
  the centroid gap obeys `8*delta*D`.
- Docs: update `PLAN.md` checklist (pointer here), README "open" list,
  ResearchStatus extension section, memory.

## Honest scope statements (repeat in all write-ups)

- This is fully closed at `t = 2`.  The fixed `t = 3` algebra/probability
  bridge is formalized conditional on first-balanced row-mass tail bounds;
  deriving those tails from primitive iid hypotheses remains open.
- Fixed anchors, one centroid branch (negative); the positive branch and the
  two-branch drift assembly compose exactly as in
  `DeletedEstimatorConsistency` / `meanSquare_sub_sub_le_two_add`.
- Constants (the `2`, `4`, `8` factors) are explicit but deliberately loose;
  sharpening them is numerics work, not theorem work.

## Checklist

- [x] PLAN_BALANCED_STATS.md committed
- [x] B0 two-sided weight-sum deviation
- [x] B1 centroid relative perturbation (+ in-hull sub-lemma)
- [x] B2a/B2b/B2c relative-error toolkit
- [x] B3 defs + balancedTwoStepCentroid_deviation_prob_le
- [x] Two-branch normalized drift assembly
- [x] B4 full-matrix reconciliation (optional)
- [x] B5 fixed t=3 unrolling core with explicit level-1 mass-tail hypotheses
- [x] Audit entries + Check.ps1 + #print axioms
- [x] S6 numeric validation of constants
- [x] Docs updated
- [x] Final commit
