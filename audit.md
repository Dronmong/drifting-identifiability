# Comprehensive audit report

## Executive verdict

The core Lean mathematics is sound, meaningful, and substantially stronger than the paper’s Appendix C.1 heuristic. I found no proof escape, fabricated theorem, circular coefficient argument, or kernel-visible dependency on the conditional RKHS/Gaussian axioms in the promoted exact results.

However, the repository currently overstates its practical completion. In particular:

> The exact finite-population theorem is genuinely verified. The claimed end-to-end finite-sample theorem for the paper’s actual Algorithm 2 is not yet complete.

The central missing bridge is that the statistical theorems freeze the anchor batch, while the paper reuses generated negatives as those anchors, `x = y_neg`. Several documents call Objective 4 “complete” despite this mismatch.

The supplied [paper](</C:/Users/Dronm/Documents/drifting_identifiability/papers/2602.04770v2.pdf>) is itself careful: Appendix C.1 calls its argument a heuristic and the conclusion explicitly remains an open question in Section 6. The project’s exact finite theorem is therefore an original restricted theorem inspired by the paper—not a formalization of a theorem already proved by the paper.

## What is genuinely established

| Layer | Audit result |
|---|---|
| Anti-symmetry: `p=q ⇒ V=0` | Correctly proved |
| Finite coefficient-minor algebra | Correct and axiom-free |
| Finite population mean-shift theorem under a frame bound | Correct |
| Genuine probability measures via density mixtures | Correct |
| Probe-energy zero ⇒ equality in the specified finite setup | Correct |
| General-`m` atomic Gaussian construction | Correct, but uses a Gaussian kernel rather than the paper’s Laplace kernel |
| Paper Laplace kernel | Concrete theorem only for two atoms and one probe |
| Non-atomic smooth model | Correct for two ordered bump components with a Gaussian kernel |
| Stability bounds | Correct, with potentially extremely poor frame constants |
| Feature-law lifting | Logically correct but abstract/uninstantiated for the actual encoder |
| CFG | Correctly treated as affine/signed coefficients, not automatically probabilities |
| Fixed-anchor SNIS estimator | Correct |
| Actual reused-negative Algorithm 2 | Not yet covered by the statistical theorem |
| Raw-drift asymptotic identifiability | Still open |
| Optimization/SGD convergence to zero drift | Not addressed |

The strongest promoted population theorem depends only on:

- `equation_11_bilinear_mean_shift`
- `equation_31_bilinear_expansion`
- `antisymmetric_kernel_induces_basis_antisymmetry`

plus Lean’s foundational axioms. The targeted axiom prints confirmed this.

## High-severity findings

### 1. Fixed anchors are not the paper’s reused-negative estimator

The statistical chain in [Algorithm2SNIS.lean](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/Algorithm2SNIS.lean:8) explicitly assumes fixed anchors. Its probability theorems receive:

```lean
anchors : Fin Nx → F
Yneg    : Fin Nneg → Ω → F
```

as separate objects.

But Algorithm 2 sets:

```text
x = generated samples
y_neg = x
```

so the anchors themselves are the random negative samples. Consequently, the column mass and every negative weight depend jointly on the entire random negative batch. One cannot instantiate the current theorem by setting `anchors = Yneg ω`, because that turns the supposedly fixed weight function into a random, batch-coupled function.

The same issue remains in the deleted/self-mask theorem: [DeletedEstimatorConsistency.lean](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/DeletedEstimatorConsistency.lean:315) also takes deterministic `anchors` separately from random `Yneg`.

The numerical consistency experiment likewise fixes anchors such as `[0, 0.3, 1]`; it does not analyze `x=y_neg`.

Therefore these claims are too strong:

- “Objective 4 … complete” in [ResearchStatus.md](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/ResearchStatus.md:394)
- “complete the chain: sampled no-mask Algorithm-2 centroids…” in [ResearchStatus.md](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/ResearchStatus.md:502)
- the analogous statement in [WrittenProof.md](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/WrittenProof.md:434)

What is complete is a fixed-anchor or sample-split variant.

### 2. The selected probes are not generally probes observed by the paper’s loss

The general Gaussian theorem evaluates the field at structured integer probes. In its atomic model, generated samples lie on the support points `zᵢ`; those integer probes need not lie in the generated support.

Thus the statement that this “matches the finite quantity a training loss can observe” in [PopulationIdentifiability.lean](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/PopulationIdentifiability.lean:367) is generally false for the paper’s loss, which evaluates drift at generated anchors `x∼q`.

The probe-local theorem is mathematically valid, but it is a query-access theorem: it assumes the ability to evaluate drift at chosen probes. That is not automatically the paper’s training objective.

The two-atom `{0,1}` specializations partly avoid this because the probe can be an atom. The general structured-probe theorem does not.

### 3. The practical normalized, multi-temperature loss is not connected to identifiability

The project defines:

- feature normalization,
- drift RMS normalization,
- multiple-temperature aggregation,

in [Paperaxioms.lean](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/Paperaxioms.lean:491), but no promoted theorem consumes these definitions.

This matters because the practical paper does not simply minimize each raw field independently. It normalizes each drift and then sums multiple temperature fields before evaluating the feature loss:

```text
Ṽ_j = Στ Ṽ_{j,τ}
```

Zero of that sum does not imply zero of each temperature field; cancellation is possible. Likewise, “small normalized loss” is not the same statement as “small raw population drift.”

The exact zero-set arguments for one fixed kernel therefore do not yet establish identifiability of the actual multi-temperature training objective.

### 4. The registered candidate does not satisfy the project’s own final workflow

The registered condition [FiniteBasisFamily](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/PopulationCandidate.lean:26) says only that `p` and `q` belong to the same finite mixture family. It does not include the kernel, probes, regularity, or frame condition.

There is no theorem proving:

```lean
IdentifiesAtZero finiteBasisCandidate.condition V
```

for a fixed `V`.

Instead, the successful theorem takes a pair-specific `PopulationMeanShiftFiniteSetup` containing `a`, `b`, regularity, probes, and the frame certificate. This is mathematically fine, but it means the repository has not completed the precise `CandidateSpec → IsLegitimate → IdentifiesAtZero` workflow required by its own [AGENTS.md](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/AGENTS.md:95).

### 5. The original asymptotic goal remains open

The project has finite coefficient stability, but no promoted `AsymptoticallyIdentifies` theorem for the paper’s mean-shift field.

The only actual theorem using `AsymptoticallyIdentifies` is the conditional Gaussian-MMD result in [CharacteristicIdentifiability.lean](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/CharacteristicIdentifiability.lean:84), and that uses MMD discrepancy—not vanishing raw drift.

This matches the failure log’s correct observation that a small gradient/drift does not generally imply a small discrepancy. The project should state plainly:

> Exact restricted population identifiability is solved; the paper’s raw-drift asymptotic problem is not.

## Trusted-boundary assessment

The promoted core’s three analytic axioms appear mathematically valid under their stated integrability hypotheses. They are also plausible candidates for direct Mathlib proofs; none expresses identifiability.

Nevertheless, the trusted boundary is larger than necessary:

- Several “paper axioms” are elementary algebra or definitional identities.
- `equation_6_loss_value`, the CFG identities, and multiple MMD rearrangements should not need axiomatization.
- `sampleMean_meanSquare_le` should explicitly require measurability or integrability of the vector-valued summands, rather than relying on downstream callers to supply stronger conditions.
- The conditional `characteristic_gradientEmbedding_injective` is substantively the desired identifiability result for that MMD field. The repository correctly excludes it from promoted claims.

The SHA-256 manifest matches the current [Paperaxioms.lean](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability/Paperaxioms.lean:1), but a repository-local hash guards against accidental changes, not against simultaneous modification of both files.

## Assumption quality

The strongest assumptions are honestly exposed:

- Positive frame bounds are genuine, independently checkable finite linear algebra.
- Concrete atomic and two-bump examples discharge them without circularity.
- Normalizer nonvanishing is correctly required.
- Feature-law equality is not promoted to source-law equality without an embedding or stability certificate.
- CFG nonnegativity is correctly separated from affine algebra.

Important limitations remain:

- The general Gaussian construction is not the paper’s Laplace kernel.
- The actual Laplace result is only two atoms.
- The smooth result is only two components and uses Gaussian interactions.
- The learned encoder is not proved injective or measure determining.
- `FeatureStabilityCertificate` and `MeasureDetermining` are abstract interfaces, not results about the paper’s encoder.
- Positive temperature is sometimes omitted from exact theorem signatures. Lean totalizes division by zero, so a theorem may hold for the artificial `τ=0` function even though that is not the paper’s kernel.
- A raw-drift stability statement needs a nontrivial lower bound on the affinity mass product. Otherwise a small raw drift can result from vanishing gain rather than matched centroids.

## Paper-alignment assessment

The project improves several weak points in Appendix C.1:

- It replaces the paper’s informal “full support effectively gives all points” claim with a correct continuity/full-support theorem.
- It proves the minor grouping rather than assuming it.
- It makes normalization explicit.
- It identifies the necessary dimension bound.
- It distinguishes exact nonsingularity from useful conditioning.
- It does not repeat the paper’s unjustified final approximation step from finite bases to arbitrary distributions.

But the following paper-to-project bridges remain unproved:

1. Generated-anchor sampling versus arbitrary fixed probes.
2. Reused negatives versus fixed/sample-split anchors.
3. The eye-mask estimator with random anchors.
4. Batch feature normalization.
5. Drift RMS normalization.
6. Multi-temperature aggregation.
7. Multiple learned feature maps.
8. CFG mixed negatives in the actual estimator.
9. Generator parameterization and SGD dynamics.
10. Vanishing training loss or gradient implying the required population zero.

## Numerical audit

The numerical code is useful and generally candid, but some labels should be weakened.

The “certified sample counts” in `numerics/RESULTS.md` are theorem-informed calculations for the fixed-anchor two-atom model. They are not certificates for the paper’s reused-negative estimator.

The repository correctly admits that:

- no real paper encoder features have been evaluated;
- the feature geometry experiments are synthetic;
- the paper operates at `N=64`, while the formal bounds are vastly larger;
- large-`m` frame conditioning collapses badly.

The CFG “generic case” conclusion is based on a uniform Dirichlet experiment, so it should be described as generic under that chosen prior—not generic for real class-conditional ImageNet distributions.

The Sinkhorn extension is clearly labeled as outside the paper, which is good. Some summary numbers in its README/ResearchStatus no longer agree with the current generated results, indicating documentation drift.

## Documentation and architecture issues

The main README is stale:

- It links to nonexistent `AGENT.md` rather than `AGENTS.md` at [README.md](/C:/Users/Dronm/Documents/drifting_identifiability/README.md:68).
- It describes `GaussianNondegeneracy.lean` too positively despite that module being conditional and synthetic at [README.md](/C:/Users/Dronm/Documents/drifting_identifiability/README.md:48).
- It describes `EmpiricalFrameBound` as an `m=2` result although it now contains the general-`m` construction at [README.md](/C:/Users/Dronm/Documents/drifting_identifiability/README.md:56).
- It does not map most of the estimator, CFG, feature, or Sinkhorn architecture.
- `ResearchStatus.md` contains a duplicate Objective 5 heading and contradictory “complete” versus “remaining” descriptions.

The default root module also imports the Sinkhorn extension at [DriftingIdentifiability.lean](/C:/Users/Dronm/Documents/drifting_identifiability/DriftingIdentifiability.lean:15). A cleaner architecture would keep the paper formalization, promoted extensions, and conditional research in separate roots.

## Tooling results

The complete local check passed:

```text
Trust audit passed: 32 Lean files checked
Build completed successfully: 8609 jobs
Promoted axiom audit passed: 160 declarations
Conditional modules compiled
```

No files were changed, and the worktree remains clean.

One CI weakness remains: [lean_action_ci.yml](/C:/Users/Dronm/Documents/drifting_identifiability/.github/workflows/lean_action_ci.yml:20) runs the trust audit and ordinary Lean action, but not `scripts/Check.ps1`. Therefore CI does not explicitly run the promoted axiom audit or conditional-module compilation that the project’s protocol requires.

## Recommended priorities

1. Relabel Objective 4 as “fixed-anchor/sample-split estimator complete; reused-negative estimator open.”
2. Formalize the actual coupled estimator with `anchors(ω)=Yneg(ω)`, likely using leave-one-out, conditional, U-statistic, or batch-functional concentration.
3. Formalize the practical normalized multi-temperature loss and test cancellation counterexamples.
4. Replace arbitrary-probe observability claims with an explicit probe-access assumption, or derive probe zero from generated-anchor loss.
5. Package the finite candidate’s frame/regularity assumptions into a genuine condition and prove an `IdentifiesAtZero` theorem.
6. Prove the three promoted analytic axioms directly and tighten the sample-mean axiom.
7. Add a finite-mixture distribution-distance theorem, e.g. coefficient `ℓ¹` controlling total variation, while clearly retaining the fixed-family assumption.
8. Make CI invoke the full `Check.ps1`.
9. Refresh README, ResearchStatus, WrittenProof, and numerical wording.

Bottom line: this is a strong restricted-population formalization with real mathematical content. Its core exact theorem deserves confidence. Its present claim of having completed the paper’s practical finite-sample Algorithm-2 chain does not.

## Adopted repair program

Recorded on 2026-07-08. Repairs are ordered by correctness impact and
dependency, not by implementation convenience.

1. **Correct the public scope immediately.** Relabel Objective 4 as complete
   only for fixed-anchor/sample-split estimators; distinguish selected
   query-access probes from generated training anchors; state explicitly that
   exact restricted population identifiability is proved while raw-drift
   asymptotic identifiability remains open.
2. **Repair documentation and automation.** Refresh the root README,
   `ResearchStatus.md`, `WrittenProof.md`, presentation/numerical wording, and
   stale Sinkhorn summaries. Make CI execute the full trust, build,
   conditional-module, and promoted-axiom audit.
3. **Complete the candidate workflow.** Package finite-family membership,
   kernel, probes, regularity, integrability, and the positive frame
   certificate into a genuine pair condition; prove both legitimacy and an
   `IdentifiesAtZero` theorem for a fixed mean-shift field.
4. **Harden the statistical trusted boundary.** Add explicit
   measurability/integrability assumptions to `sampleMean_meanSquare_le`, or
   replace it with a Mathlib proof. This changes `Paperaxioms.lean` and its
   manifest and therefore remains blocked pending explicit user approval.
5. **Close the reused-negative estimator gap.** Formalize the coupled random
   estimator with `anchors(ω) = Yneg(ω)`, including its eye-mask/leave-one-out
   dependence, using a conditional, leave-one-out, U-statistic, or
   batch-functional concentration argument.
6. **Connect selected probes to training observations.** Either assume
   explicit query access, use the existing continuity/full-support energy
   route, or prove a generated-anchor sampling theorem appropriate to the
   model class.
7. **Analyze the practical normalized objective.** Formalize batch feature
   normalization, drift RMS normalization, and multi-temperature aggregation;
   first record cancellation counterexamples, then state sufficient
   non-cancellation conditions.
8. **Pursue the asymptotic theorem separately.** Specify the drift norm and
   distribution topology and prove a genuine `AsymptoticallyIdentifies`
   theorem only where uniform frame, normalizer, tightness, and
   approximation-control assumptions support it.

Implementation status: items 1–3 are authorized to proceed without modifying
the trusted boundary; item 4 is approval-gated; items 5–8 are substantive
research work rather than documentation cleanup.

Progress (2026-07-08):
- **Items 1–3 DONE** (commit `2fa14e9`): scope relabel, docs/CI/architecture
  (CI now runs the full `Check.ps1`; Sinkhorn extension moved to the opt-in
  `Extensions.lean` root), and the completed candidate workflow
  (`finitePopulationMeanShiftCandidate_identifiesAtZero` + `_isLegitimate`).
  A start on item 7's cancellation record also landed
  (`aggregateTemperatureDrift_cancels`, `_zero_with_nonzero_component`).
- **Item 4 DONE** (user-approved; commit pending): `sampleMean_meanSquare_le`
  now requires `hmeas : ∀ i, Measurable (Z i)` for the vector summands. On a
  second-countable Borel space this is strong measurability, so with `hint2`
  each `Z i` is genuinely `L¹` and the mean-zero premise `hmean` is a real
  Bochner integral rather than the vacuous totalized `0` a non-integrable
  function would give — closing the soundness gap the audit flagged. All six
  call sites already carried the measurability fact and were threaded through;
  the caller-less `sampleMean_concentration` gained the matching hypothesis.
  Manifest regenerated; `Check.ps1` green.
- **Beyond the audit** (user-approved; commit `cc465fe`): the raw-field Gaussian
  converse `gaussianMeanShiftDrift_identifiesAtZero` (opt-in Conditional module,
  one new conditional axiom `gaussianMeanShift_injective`) — the rigorous form
  of the reviewer rebuttal's argument 1; see `DriftingIdentifiability/RawFieldConverse.md`.
- **Still open:** items 5 (reused-negative estimator — the main practical gap),
  6, 7 (beyond the cancellation record), 8 (asymptotic).
