# NCJ low-dimensional improvement: results (2026-07-20)

*Final report for `IdentifiabilityDrivenImprovementPlan.md`.
Frozen protocol: `IdentifiabilityImprovementProtocol.md` (registries and gate
pre-registered before validation).  Frozen generator protocol:
`IdentifiabilityGeneratorProtocol.md`.  Field library: `identifiability_drift.py`
(9 invariant tests pass, paper path bitwise-identical to the audited historical
estimator).  Certified theorems: `DriftingIdentifiability/NCJIdentifiability.lean`
(T1--T4, trust-audited).  Runs with manifests in `identifiability_runs/`.
Everything here is measured synthetic evidence on the empirical-particle model;
nothing benchmarks the paper's trained image models.*

## Verdict up front

**The pre-registered particle gate (E4) PASSED decisively; the pre-registered
learned-generator gate (E5) FAILED.**

Per the plan's own outcome table (§9), the earned claim is exactly:

> **An empirical-particle estimator improvement over the exact paper
> implementation, under matched architecture and compute, that does NOT
> transfer through the tested learned generator.**

No general learned-generator claim is made, and nothing here supports an
ImageNet or real-feature superiority claim.

The winning mechanism is **not** jitter. Validation selected `sigma = 0`; the
gain came entirely from (1) dropping the identifiability-inert `P*Q` gain and
(2) removing the finite self-pair by cross-fitting. This is why **T5 (jitter /
fission) is deliberately not formalized**: the experiment gave it no support
(jitter monotonically *worsened* validation ED2), so there is no empirical
mechanism for T5 to certify.

## The frozen policy

`ncj_policy_frozen.json` (committed before the test): winner
`ncj__eta-0.0525__sigma-0.0`, i.e.

```text
gain      = constant (V_i = Cpos_i - Cneg_i, no P*Q)
reference = independent cross-fit resample, no eye mask
jitter    = sigma/tau = 0  (none)
eta       = 0.0525,  tau = 0.35,  norm clip = 2.0
```

Because `sigma = 0`, this frozen arm is identical to `normalized-crossfit`.

## Validation mechanism decomposition

Geometric-mean ED2 over validation cells (lower is better),
from `ncj_policy_frozen.json`:

| Arm | Mechanism | Aggregate ED2 | vs paper |
|---|---|---:|---:|
| paper | `P*Q`, reused, masked | 0.03944 | 1.00 |
| crossfit-only | `P*Q`, independent, unmasked | 0.03147 | 0.80 |
| jitter-only (σ/τ=0.10) | `P*Q`, reused, masked, jitter | 0.03997 | 1.01 |
| jitter-only (σ/τ=0.50) | — | 0.04533 | 1.15 |
| normalized-only (η=0.0525) | constant, reused, masked | 0.01215 | 0.31 |
| **normalized-crossfit (η=0.0525)** | **constant, independent, unmasked** | **0.00349** | **0.088** |
| ncj (η=0.0525, σ/τ=0.10) | + jitter | 0.00354 | 0.090 |
| ncj (η=0.0525, σ/τ=0.50) | + more jitter | 0.00497 | 0.126 |

Reading:

1. **Dropping the `P*Q` gain is the dominant effect** (0.31× on its own).
   This is the mechanism T1 certifies as zero-set-preserving and T4 certifies
   as removing an exponential off-support stall.
2. **Cross-fitting compounds it** (normalized-only 0.31× → normalized-crossfit
   0.088×). On its own with the paper gain, cross-fitting is a modest 0.80×.
3. **Jitter never helps.** Every positive `sigma` raises ED2 monotonically,
   both with the paper gain (jitter-only) and on top of NCJ. The selection rule
   therefore froze `sigma = 0` (ties broken toward smaller jitter were not even
   needed; σ=0 strictly won).

## E4 — frozen particle test (16 targets × 4 inits × 20 seeds, 6 arms)

Run: `identifiability_runs/20260720-011000-NCJ-test-standard/`
(7680 rows, ~87 min wall, clean-tree, `e4_gate.json`).

**All eight pre-registered criteria pass; `PASS=true`.**

| Criterion | Requirement | Realized | Verdict |
|---|---|---:|---|
| 1 geo-mean ratio | ≤ 0.80 | **0.100** | ✅ |
| 2 hierarchical 95% CI upper | < 1 | [0.050, **0.195**] | ✅ |
| 3 winning cells | ≥ 60% | **93.75%** | ✅ |
| 4 worst family median | ≤ 1.10 | **0.892** (student) | ✅ |
| 5 Gaussian-mixture subgroup CI upper | < 1 | [0.026, **0.279**] | ✅ |
| 6 non-Gaussian subgroup CI upper | < 1 | [0.044, **0.242**] | ✅ |
| 7 missing-mode KM recovery | ≥ paper | paper 114, NCJ **60** | ✅ |
| 8 divergence + equal cost | ≤ +2pp, equal pairs | 0.0 vs 0.0, single pair count | ✅ |

Per-family median ratios (all < 1): helix 0.29, gauss_mixture 0.34, sine 0.35,
ring 0.37, moons 0.40, circles 0.42, sphere_shell 0.44, spiral 0.45,
contaminated 0.47, grid_mixture 0.50, banana 0.52, gaussian 0.52, student 0.89.

The two cells with ratio > 1 are both heavy-tailed 5-d stress starts
(shell-5d/concentrated 1.45, student-5d/missing 1.25); every `far` init is a
very large win (ratios ~1e-3), reflecting the no-freeze mechanism T4 describes.

## E5 — frozen learned-generator transfer (FAILED)

Run: `identifiability_runs/20260720-024712-NCJ-generator-standard/`
(`e5_gate.json`, `PASS=false`). Fixed small MLP generator, identical
architecture / init / latent batches / optimizer / update count across paired
arms; exact paper stop-gradient semantics; equal generator-forward budget
(1602 vs 1602) and kernel-pair budget.

| Criterion | Requirement | Realized | Verdict |
|---|---|---:|---|
| 1 ratio vs paper | ≤ 0.80 | **1.072** | ❌ |
| 2 CI vs paper upper | < 1 | [0.946, **1.208**] | ❌ |
| 3 CI vs compute-matched upper | < 1 | [0.948, **1.205**] | ❌ |
| 4 winning cells | ≥ 60% | **28.1%** | ❌ |
| 5 subgroup CIs upper | < 1 | gmix [0.99, 1.32], nonG [0.89, 1.26] | ❌ |
| 6 coverage + divergence | non-inferior | equal coverage, 0 divergence | ✅ |
| 7 equal compute | equal forwards | 1602 = 1602 | ✅ |
| 8 zero-σ regression guards | bitwise | paper=matched, ncj=norm-crossfit | ✅ |

The particle-level improvement **does not survive** the learned-generator
optimization. The constant-gain field, excellent for moving free particles,
does not translate into a better regression target through the MLP under the
paper's training loop; `far` inits invert from huge particle wins to large
generator losses (banana/far 2.23, gmix-2d/far 1.59). This is a clean,
pre-registered negative that the plan explicitly provisioned for.

## Certified layer (T1--T4, trust-audited)

`DriftingIdentifiability/NCJIdentifiability.lean`, `Check.ps1` fully green;
`#print axioms` shows only `propext / Classical.choice / Quot.sound` for
T1/T2/T4, plus the single allowlisted `Paper.sampleMean_meanSquare_le` for T3.

* **T1** positive-gain zero-set preservation — the constant gain has exactly
  the paper field's zero set, so replacing `P*Q` by `1` cannot change what
  `V=0` identifies.
* **T2** jittered identifiability composition — even with symmetric Gaussian
  jitter, zero Laplace drift between the smoothed laws forces `p=q` (kept as a
  guarantee although jitter was not selected).
* **T3** cross-fitted centroid-difference consistency — the cross-fit estimator
  is the audited fixed-anchor self-normalized object; MSE bound with explicit
  batch size and weight floor.
* **T4** no-freeze vs exponential attenuation — the normalized field has a
  positive speed floor `gmin·c`, while the paper field inherits any certified
  exponential bound on `P*Q`. This is the certified shadow of the validation
  finding that dropping `P*Q` is the dominant mechanism.

* **T5 (jitter / fission): intentionally not formalized.** Validation selected
  `sigma = 0` and jitter degraded ED2; formalizing a jitter-splitting theorem
  would assert a mechanism the data contradict.

## Contemporary comparison controls

### Analytical Bias Correction (ABC): does a cheaper single-batch correction
### recover the cross-fit gain?

`ncj_abc_comparison.py`, run
`identifiability_runs/20260720-145203-NCJ-abc-comparison-abc-context/`
(`abc_comparison.json`). A **post-gate** mechanism-attribution comparison — not
a gate, and it changes neither frozen verdict. Same 400-step dynamics and
`N=48/B=64` as the frozen study, 6 paired seeds, on the frozen test registry.
The ABC field applies the standard first-order self-normalized ratio-bias
correction `Ĉ_ABC = Ĉ + Σ_j a_j²(y_j − Ĉ)` to the reused/masked batch — the
cheap single-batch alternative to cross-fitting (no second forward pass).
(This is the textbook SNIS correction, independently derived; the plan's cited
arXiv:2604.27239 post-dates this assistant's knowledge cutoff and is not
independently verified here.)

| Arm | Mechanism | Ratio vs paper | Hier. 95% CI | Winning cells |
|---|---|---:|---|---:|
| normalized-only | drop `P*Q`, reused+masked | 0.315 | [0.151, 0.636] | 36% |
| **abc** | + analytic self-pair correction | **0.326** | [0.156, 0.653] | 38% |
| normalized-crossfit | + independent cross-fit | **0.101** | [0.050, 0.196] | 95% |

**Finding: ABC does not recover the cross-fit gain.** The analytical
single-batch correction (0.326) is statistically indistinguishable from simply
dropping the gain (0.315) and is ~3× worse than cross-fitting (0.101), with an
essentially disjoint confidence interval. So the cross-fit mechanism is **not**
merely a leading-order bias correction: replacing the reused/masked self-pair by
a genuinely *independent* reference batch matters beyond the `O(1/N)` term that
ABC removes. (The context run's cross-fit ratio 0.101 reproduces the full E4
gate value 0.100, validating the reduced-seed profile.) This strengthens the
E4 attribution: cross-fitting is a load-bearing, not cosmetic, mechanism.

**Sinkhorn Drifting and Kernel-Gradient Drifting (context only).** The plan
(§7) cites arXiv:2603.12366 (Sinkhorn, two-sided balanced couplings) and
arXiv:2605.10727 (kernel-gradient, conservative smoothed-score field) as
heavier estimator changes to report as context rather than reimplement. Both
arXiv identifiers post-date this assistant's knowledge cutoff and cannot be
independently verified here; they are recorded as the plan's cited comparators,
not as validated reproductions. Structurally they are orthogonal to the NCJ
result: NCJ keeps the paper's one-sided softmax and changes only the gain and
the reference batch, whereas Sinkhorn replaces the one-sided normalization and
kernel-gradient replaces the displacement field. A matched implementation of
either is deferred; it is a separate estimator-design question and is not
required to state the particle-level claim earned above.

## What this buys the research program

1. **A pre-registered, honestly-passed particle gate** attributing a large
   low-dimensional gain to two formally-motivated mechanisms (drop `P*Q`,
   cross-fit the reference), with the certified T1/T3/T4 backing.
2. **A pre-registered generator-transfer negative.** The particle-estimator
   win does not pass through the tested MLP; the flagship general claim is
   *not* earned, and this is recorded rather than worked around.
3. **A falsified sub-hypothesis (jitter).** Symmetric jitter, though certified
   identifiability-safe (T2), does not help these low-dimensional dynamics and
   was correctly frozen out.

## Scope

Synthetic 1-d–5-d targets, empirical-particle and small-MLP models, exact
bi-softmax estimator, matched kernel-pair and generator-forward compute,
`N=48`, 400-update horizon. Not tested: learned encoder features, images,
large `N`, long horizons. The claim is bounded to the empirical-particle
estimator; the generator gate is a recorded failure, not a success.
