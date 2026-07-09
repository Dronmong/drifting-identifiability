# Laplacian-kernel converse for Gaussian targets: implementation plan

**Status: COMPLETE, AXIOM-FREE, PROMOTED (2026-07-09).** All parts A–F compile
in `LaplacianGaussianConverse.lean`, imported by the default root and
registered in the promoted axiom audit (176 declarations; Check.ps1 green:
36 files, 15 paper + 5 conditional axioms — this route adds none and uses
none). `#print axioms` on `laplaceGaussianMeanShiftDrift_identifiesAtZero`,
`gaussianRadialLimit_zero_imp_parameters_eq`,
`multivariateGaussian_laplaceMeanShiftDrift_radial_tendsto`, and
`laplaceGaussianCandidate_isLegitimate` reports only
`propext, Classical.choice, Quot.sound`. Both reviewer-rebuttal converses are
now machine-checked (argument 1 in `GaussianScoreRecovery.lean`, argument 2
here).

## Objective and exact scope

Formalize the second identifiability result described in the authors'
rebuttal. For the paper's Laplace kernel

```text
kτ(x,y) = exp (-‖x-y‖ / τ),    τ > 0,
```

prove that pointwise zero raw mean-shift drift identifies two multivariate
Gaussian laws:

```text
ZeroDrift (meanShiftDrift (laplaceKernel τ))
  (multivariateGaussian μp Sp)
  (multivariateGaussian μq Sq)
→
multivariateGaussian μp Sp = multivariateGaussian μq Sq.
```

The intended Lean setting is
`E := EuclideanSpace ℝ ι`, with `[Fintype ι] [DecidableEq ι]`,
`Sp.PosSemidef`, `Sq.PosSemidef`, and positive bandwidth
`ValidBandwidth τ`.

This theorem is:

- exact and pointwise, not an asymptotic-identifiability theorem;
- for arbitrary means and positive-semidefinite covariances, including
  degenerate Gaussian laws;
- for Gaussian target/model laws only, not arbitrary distributions;
- about the ideal population mean-shift field, not Algorithm 2 or minibatches.

## Mathematical mechanism

For a unit direction `u`, put `x = r • u`. After cancelling the common
`exp (-r/τ)` factor, define

```text
w(r,u,y) := exp ((r - ‖r • u - y‖) / τ).
```

The radial geometry gives

```text
r - ‖r • u - y‖ → ⟪u,y⟫
```

as `r → +∞`. Therefore the normalized Laplace centroid converges to the
exponentially tilted centroid

```text
TiltCentroid(P,u)
  := (∫ exp (⟪u,y⟫/τ) • y dP)
       / (∫ exp (⟪u,y⟫/τ) dP).
```

For `P = N(μ,Σ)`, Gaussian exponential tilting gives

```text
TiltCentroid(P,u) = μ + (1/τ) • (Σ u).
```

Since

```text
meanShift k P x = kernelCentroid k P x - x,
```

the `-r • u` terms cancel in the drift, yielding

```text
meanShiftDrift kτ P Q (r • u)
  → (μp - μq) + (1/τ) • ((Sp - Sq) u).
```

If the drift is zero everywhere, this limit is zero for every unit `u`.
Applying the identity to `u` and `-u` gives `μp = μq`; the remaining
identity gives `(Sp - Sq)u = 0` for every unit direction, hence `Sp = Sq`.
The two `multivariateGaussian` measures are then definitionally equal after
substitution.

## Proposed Lean architecture

Create `DriftingIdentifiability/LaplacianGaussianConverse.lean`. Keep the
analytic helpers in this module until their interfaces stabilize.

### A. Deterministic radial geometry

Introduce:

```lean
noncomputable def laplaceCompensatedWeight
    (τ r : ℝ) (u y : E) : ℝ :=
  Real.exp ((r - ‖r • u - y‖) / τ)

noncomputable def exponentialTiltWeight
    (τ : ℝ) (u y : E) : ℝ :=
  Real.exp (⟪u, y⟫ / τ)
```

Prove:

1. `radial_norm_expansion`:
   for `‖u‖ = 1`,
   `Tendsto (fun r => r - ‖r • u - y‖) atTop (𝓝 ⟪u,y⟫)`.
   Use the rationalized identity

   ```text
   r - ‖ru-y‖
     = (2r⟪u,y⟫ - ‖y‖²) / (r + ‖ru-y‖)
   ```

   eventually for `r > 0`, then divide numerator and denominator by `r`.

2. `laplaceCompensatedWeight_tendsto`:
   continuity of division and `Real.exp` transports the previous limit.

3. `laplaceCompensatedWeight_le`:
   for `0 ≤ r` and `‖u‖ = 1`,

   ```text
   laplaceCompensatedWeight τ r u y ≤ exp (‖y‖ / τ).
   ```

   This follows from the reverse triangle inequality
   `r - ‖r • u-y‖ ≤ ‖y‖`.

4. The corresponding norm bound for the vector integrand:

   ```text
   ‖w(r,u,y) • y‖ ≤ exp (‖y‖/τ) * ‖y‖.
   ```

These lemmas contain no probability assumptions and should be proved first.

### B. Generic compensated-integral limit

Define the kernel centroid separately from mean shift:

```lean
noncomputable def kernelCentroid
    (k : E → E → ℝ) (P : Measure E) (x : E) : E :=
  (kernelNormalizer k P x)⁻¹ • ∫ y, k x y • y ∂P
```

Prove the algebraic identity:

```text
meanShift k P x = kernelCentroid k P x - x
```

under probability normalization and the visible integrability assumptions.

For a probability measure `P` satisfying

```text
Integrable (fun y => exp (‖y‖/τ) * (1 + ‖y‖)) P,
```

use `tendsto_integral_filter_of_dominated_convergence` to prove:

1. compensated normalizer convergence:

   ```text
   ∫ w(r,u,y) dP → ∫ exp (⟪u,y⟫/τ) dP;
   ```

2. compensated first-moment convergence:

   ```text
   ∫ w(r,u,y) • y dP
     → ∫ exp (⟪u,y⟫/τ) • y dP;
   ```

3. positivity of the limiting denominator;

4. ratio convergence:

   ```text
   kernelCentroid kτ P (r • u) → TiltCentroid(P,u).
   ```

Keeping this layer generic separates the Laplace asymptotic from Gaussian
moment calculations and makes every domination assumption explicit.

### C. Gaussian exponential integrability

For `P := multivariateGaussian μ S`, discharge the generic domination
hypothesis using Mathlib's Gaussian infrastructure:

- `IsGaussian.exists_integrable_exp_sq` (Fernique);
- `IsGaussian.memLp_id`;
- elementary bounds absorbing
  `exp (‖y‖/τ) * (1 + ‖y‖)` into
  `C * exp (a * ‖y‖²)` for a sufficiently small `a > 0`.

This avoids imposing an unnecessary bounded-support or finite-moment
hypothesis: Gaussian exponential-square integrability is enough.

### D. Gaussian tilted-centroid identity

Prove, rather than axiomatize:

```text
gaussian_exponentialTilt_centroid
  (hS : S.PosSemidef) (hτ : 0 < τ) (u : E) :
  TiltCentroid (multivariateGaussian μ S) u
    = μ + (1 / τ) • (S *ᵥ u).
```

Mathlib does not currently expose this exact multivariate MGF theorem, but it
does provide:

- `integral_id_multivariateGaussian`;
- `covarianceBilin_multivariateGaussian`;
- `charFun_multivariateGaussian`;
- `IsGaussian.map_eq_gaussianReal`;
- `mgf_gaussianReal`.

Derivation strategy:

1. For any `a : E`, map the Gaussian by `y ↦ ⟪a,y⟫`.
   `IsGaussian.map_eq_gaussianReal` identifies the scalar law.

2. Use `mgf_gaussianReal` to prove

   ```text
   ∫ exp (⟪a,y⟫) dP
     = exp (⟪a,μ⟫ + 1/2 * a ⬝ᵥ S *ᵥ a).
   ```

3. To identify the vector numerator, test against an arbitrary `v : E`.
   Differentiate at `s = 0` the scalar identity for
   `a = u/τ + s • v`.

4. Justify differentiation under the integral using the same Fernique
   domination developed in part C.

5. Extensionality over inner products yields the vector first-moment formula,
   and division by the positive scalar MGF gives the tilted centroid.

Fallback allowed by project policy: if the differentiation/MGF bridge proves
unreasonably expensive in Lean, isolate only the standard exponential-tilt
first-moment identity as a proposed external theorem, document a precise
literature source, and request user approval before adding it. Do not
axiomatize the radial limit, the final zero-drift implication, or a theorem
whose conclusion is equality of the two Gaussian laws.

### E. Assemble the radial drift limit

Prove:

```lean
theorem tendsto_laplaceMeanShiftDrift_gaussian_radial
    (hτ : ValidBandwidth τ)
    (hSp : Sp.PosSemidef) (hSq : Sq.PosSemidef)
    (hu : ‖u‖ = 1) :
    Tendsto
      (fun r =>
        meanShiftDrift (laplaceKernel τ)
          (multivariateGaussian μp Sp)
          (multivariateGaussian μq Sq) (r • u))
      atTop
      (𝓝 ((μp - μq) + (1 / τ) • ((Sp - Sq) *ᵥ u)))
```

The proof should visibly:

- rewrite each mean shift as `kernelCentroid - r • u`;
- cancel the shared `-r • u`;
- apply the two Gaussian centroid limits;
- simplify matrix-vector subtraction.

### F. Recover Gaussian parameters

Prove a purely finite-dimensional algebra lemma:

```text
(∀ u, ‖u‖ = 1 →
  (μp-μq) + (1/τ) • ((Sp-Sq) *ᵥ u) = 0)
→ μp = μq ∧ Sp = Sq.
```

Implementation:

1. Handle the empty index type separately by subsingleton elimination.
2. For nonempty `ι`, choose a coordinate unit vector.
3. Compare the equations at `u` and `-u` to obtain `μp-μq = 0`.
4. Use every coordinate basis vector to show each column of `Sp-Sq` is zero.
5. Apply matrix extensionality.

Then establish the final theorem:

```lean
theorem laplaceGaussianMeanShiftDrift_identifiesAtZero ...
```

For each unit `u`, the zero-drift hypothesis makes the radial function
identically zero, so its limit is zero. Compare that with the explicit limit,
recover the parameters, substitute them, and conclude equality of measures.

## Candidate legitimacy and failure testing

Add a real `CandidateSpec` for the nonempty-dimensional Gaussian family.
Prove legitimacy with two distinct Gaussians, for example equal covariance and
different means. This witness is chosen before any zero-drift assumption.

Record and test the following boundaries:

- arbitrary non-Gaussian targets are not covered;
- `τ ≤ 0` is excluded;
- zero drift at finitely many probes is insufficient for the radial argument;
- almost-everywhere zero requires a separate continuity/full-support upgrade;
- empty dimension is mathematically harmless but cannot witness candidate
  legitimacy;
- equality of means alone is insufficient; both `u` and `-u` and all coordinate
  directions are required to recover covariance.

## Trust and module architecture

- No new axiom, constant, opaque declaration, `sorry`, or hidden injectivity
  condition.
- Initially import the module only through a dedicated development target.
- Once axiom-free and audited, add it to the default root and promoted theorem
  list.
- If the optional external Gaussian-tilt identity is needed, keep the module
  conditional until that assumption is separately reviewed or discharged.
- Do not place the desired zero-drift converse, the radial limit, or Gaussian
  parameter injectivity in `Paperaxioms.lean`.

## Verification and acceptance criteria

Completion requires:

1. the exact paper Laplace kernel, not a smooth surrogate;
2. explicit positive bandwidth and positive-semidefinite covariance premises;
3. an internally proved radial limit for denominator and vector numerator;
4. visible denominator positivity and all Bochner integrability assumptions;
5. recovery of both means and covariance matrices;
6. a legitimate Gaussian-family candidate admitting a distinct pair;
7. `#print axioms` with no conditional identifiability dependency;
8. clean `scripts/Check.ps1`, trust audit, and warning-as-error build;
9. documentation that labels the theorem Gaussian-family-only and
   population/pointwise-only.

## Recommended implementation order

1. Part A: radial geometry.
2. Part C: reusable Gaussian exponential domination.
3. Part B: generic dominated-convergence centroid limit.
4. Part D: Gaussian tilted-centroid identity.
5. Part E: radial drift limit.
6. Part F: parameter recovery and final theorem.
7. Candidate legitimacy, counterexample ledger, documentation, and audits.

The first checkpoint should be the compensated scalar and vector integral
limits. The highest-risk checkpoint is part D, not the final identifiability
algebra.

## Implementation checkpoint (2026-07-09)

Completed in `LaplacianGaussianConverse.lean`:

- [x] exact rationalized radial identity and
      `r - ‖r • u - y‖ → ⟪u,y⟫`;
- [x] pointwise convergence of the compensated Laplace weight;
- [x] sharp domination by `exp (‖y‖ / τ)`, including the vector first moment;
- [x] generic dominated-convergence theorems for compensated mass and moment;
- [x] positivity of the limiting tilt mass and convergence of the normalized
      centroid;
- [x] exact invariance of the centroid under the compensating
      `exp (r / τ)` factor;
- [x] `meanShift = kernelCentroid - x` under explicit integrability and
      nonzero-normalizer hypotheses;
- [x] Fernique-based Gaussian integrability of both required dominators;
- [x] Laplace-kernel mass, first-moment, and normalizer regularity for Gaussian
      laws;
- [x] the generic Gaussian radial raw-drift limit to the difference of
      exponential-tilt centroids;
- [x] exact Gaussian exponential-tilt calculus via mathlib's scalar Gaussian
      MGF and tilted-measure API:
      `multivariateGaussian_exponentialTiltCentroid`;
- [x] explicit Laplacian/Gaussian radial limit:
      `multivariateGaussian_laplaceMeanShiftDrift_radial_tendsto`.

Completed in the Part F pass (2026-07-09):

- [x] parameter recovery (`gaussianRadialLimit_zero_imp_parameters_eq`):
      empty index type by extensionality; directions `u` and `-u` isolate the
      mean difference (`2 • (μp - μq) = 0` in a real vector space); positive
      scaling extends covariance-action vanishing from unit vectors to all
      vectors; `Matrix.toEuclideanCLM` is injective, so `Sp = Sq`;
- [x] final converse `laplaceGaussianMeanShiftDrift_identifiesAtZero`:
      zero drift makes each radial function identically zero, so
      `tendsto_nhds_unique` forces the explicit limit to vanish in every unit
      direction; parameter recovery closes the measure equality;
- [x] pair condition `BothMultivariateGaussian` (PSD covariances explicit),
      registered candidate `laplaceGaussianCandidate`, and
      `laplaceGaussianCandidate_identifiesAtZero`;
- [x] legitimacy `bothMultivariateGaussian_allowsDistinctPair` /
      `laplaceGaussianCandidate_isLegitimate` for every nonempty dimension:
      two unit-covariance Gaussians with different means, distinguished by the
      exact first moment (`integral_id_multivariateGaussian`), chosen before
      any zero-drift assumption;
- [x] registered in the default root and `scripts/AxiomAudit.ps1`
      (7 new promoted declarations; audit reports 176, no conditional
      dependencies); full `scripts/Check.ps1` green;
- [x] `#print axioms`: Lean foundations only, on every headline declaration —
      acceptance criteria 1–9 of this plan are all met.
