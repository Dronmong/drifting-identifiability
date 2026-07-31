# Encoder-Independent Kernel Drifting — Phase 0 results

*Implements `EncoderIndependentKernelDriftingResearchPlan.md` section 9,
Phase 0. Code: `numerics/encoder_independent_drifting/`. Immutable artifact:
`numerics/encoder_independent_drifting/phase0_gate.json` (+ `.sha256`).*

**Status: Phase-0 exit gate PASSED.** This is a mechanism and implementation
result on synthetic 16×16 structured images. It is not a generation-quality
result, not an ImageNet result, and not evidence that fixed kernels can
replace a pretrained encoder at scale. Phase 1 is the first test of that.

## What was built

A new isolated package, importing no flagship runner, no coherent-transport
route planner, no PQST controller and no neural transport teacher (plan
section 8). No pretrained encoder appears in any training objective.

| Module | Plan section | Contents |
|---|---|---|
| `spectral_anchor.py` | 6.1 | multiband random-feature anchor, orthogonal direction blocks, target-only projected-scale calibration, refresh schedule, independent audit bank, analytic gradients |
| `fixed_features.py` | 6.2 | Laplacian pyramid, fixed oriented Gabor bank, smooth modulus, first/second-order scattering, wavelet covariances, fixed random conv (NNGP-style), orthonormal Haar control |
| `kernels.py` | 6.2 | per-block positive-definite kernels, two combination rules, target-only bandwidth calibration, unbiased MMD, numeric PD check |
| `kernel_gradient.py` | 6.3 | one field API with `standard` and `kernel_gradient` modes, cross-fit drift SNR, kernel-health and drift-spectrum diagnostics |
| `adaptive_mixture.py` | 6.4 | floored simplex projection, EMA softmax controller, structurally enforced cross-fitting roles |
| `objectives.py` | 6.5 | separately nonnegative losses, branch gradient shares and cancellation cosines |
| `datasets.py`, `collision_suite.py`, `metrics.py`, `diagnostics.py`, `models.py`, `train.py`, `evaluate.py` | 9 | the nine structured targets, six source-collision pairs, the frozen metric set, the one-step generator, the arm registry |
| `reference_encoder.py` | — | the A8 stand-in, explicitly **not** the paper's pretrained encoder |

64 unit tests across the six modules named in the plan; all pass.

## Gate results

| Condition | Verdict |
|---|---|
| **G0.1** every mathematical unit test passes | **PASS** (64/64) |
| **G0.2** the anchor detects every synthetic collision pair | **PASS** (6/6) |
| **G0.3** ≥ 1 fixed geometry branch healthier than raw pixel drifting | **PASS** (12/15) |
| **G0.4** kernel-gradient and standard modes differ as predicted | **PASS** |

### G0.2 — anchor collision detection

Permutation two-sample tests, 192 samples per side, 199 permutations,
α = 0.05, using a 512-feature bank over three bands. Every pair is detected
at the smallest attainable p-value.

| Collision | What it defeats | p |
|---|---|---:|
| `patch_layout_permutation` | globally pooled patch statistics | .005 |
| `phase_scramble` | power-spectrum-only statistics | .005 |
| `color_swap` | luminance-only statistics | .005 |
| `rare_mode_drop` | bulk statistics (5% mode present vs absent) | .005 |
| `high_frequency_removal` | coarse-scale statistics | .005 |
| `translation_orbit` | fully translation-invariant statistics | .005 |

### G0.2b — geometry blindness (context, not a gate)

The same tests run against each fixed geometry's own unbiased MMD. **The
fixed geometries are measurably not measure-determining**, which is exactly
why the plan forbids them from being the correctness authority:

| Geometry | Detected | Blind to |
|---|---:|---|
| raw pixel | 6/6 | — |
| fixed wavelet | 5/6 | `color_swap` (p = .74) |
| fixed random conv | 5/6 | `rare_mode_drop` (p = .08) |

This is the empirical counterpart of
`DriftingIdentifiability/FeatureSpaceIdentifiability.lean`: a non-injective
feature map admits distinct source laws with equal feature laws. Here the
wavelet modulus discards the channel assignment that distinguishes the two
colour layouts, and the random convolutional kernel cannot resolve a 5% mass
difference at this sample size.

### G0.3 — kernel health

Median over 9 targets × 3 seeds, batch 64, kernel-gradient mode, on a
deliberately imperfect generated cloud. `ESS` is the mean row effective
sample size as a fraction of the batch: **1.0 means the kernel is flat and
carries no geometry**. `SNR` is the cross-fit drift signal-to-noise ratio
(disjoint half-batches, so no example estimates its own error bar).

| family::branch | ESS ↓ | drift SNR ↑ |
|---|---:|---:|
| `randconv::randconv` | 0.7343 | 36.03 |
| `wavelet::wavelet_s0` | 0.4885 | 29.14 |
| `wavelet::wavelet_s1` | 0.7109 | 19.14 |
| `wavelet::wavelet_s2` | 0.7836 | 16.56 |
| `pyramid::pyramid_l2` | 0.5611 | 10.41 |
| `pyramid::pyramid_l0` | 0.6324 | 9.15 |
| `pyramid::pyramid_l1` | 0.5325 | 8.76 |
| *(pyramid_global, all families)* | 0.8738 | 9.67 |
| **`haar_control::haar`** | **0.9325** | **4.778** |
| **`raw::raw`** | **0.9325** | **4.777** |

**The Haar control validates the measurement.** A full orthonormal wavelet
transform reproduces raw pixel health to three decimal places in ESS and to
four significant figures in SNR, because an orthonormal basis change leaves
every Euclidean distance untouched. The plan predicted exactly this
(section 15.1); had the control shown a gain, the apparatus would have been
wrong. The gains above it are therefore attributable to the modulus,
pooling and multiscale structure, not to "using wavelets".

### G0.4 — the two movement rules

| Kernel | cosine(standard, kernel-gradient) | spectral profile L1 |
|---|---:|---:|
| raw **Gaussian** (positive control) | **1.0000** | — |
| raw smooth-Laplace | 0.8462 | — |
| fixed wavelet | −0.0024 | 1.176 |
| fixed random conv | 0.1972 | 0.304 |

For a raw Gaussian kernel the kernel-gradient field is provably
`mean-shift / tau²`, so the cosine must be exactly 1 — measured 1.0000003,
which validates the field implementation independently of finite
differences. Under a structured kernel the two rules become **orthogonal**
(wavelet cosine ≈ 0). This is the load-bearing claim of plan section 6.3
turned into a measurement: with a structured kernel, the geometry controls
the update direction and not merely the interaction weight.

## Two implementation findings that changed the design

**1. The plan's declared sum form beats the product alternative.** Both
`sum_b w_b kappa_b` and `prod_b kappa_b^{w_b}` are positive definite (the
latter because these base kernels are infinitely divisible, plus the Schur
product theorem). Both were measured:

| | ESS | SNR |
|---|---:|---:|
| `wavelet_s0`, sum (plan's form) | 0.4885 | 29.14 |
| `wavelet_s0`, product | 0.9853 | 11.19 |

Under the product rule the block distances accumulate in one exponent, so
far from the target every affinity underflows together and the row-normalized
weights go flat — the paper's own reported failure mode, reproduced. The sum
keeps the best-matching block's discrimination. The plan's conservative
choice was right and is now the frozen default; the product rule is retained
as a registered, measured negative.

**2. A bare median heuristic leaves the kernel nearly flat.** With the plain
median-distance bandwidth every branch sat at ESS ≈ 0.98 — flat kernels,
barely distinguishable from raw. The repair (plan risk register, "kernel
affinities flatten") is a declared *target-only* selectivity calibration:
bisect one global bandwidth factor so the median row ESS on
target-vs-target data equals a declared 0.5. This reads target samples only
— never generated output, an evaluation metric or a gate — so it is a design
rule, not selection on an outcome.

## Corrections made during implementation

Four defects were found by the tests and fixed rather than papered over:

1. **`color_swap` was not a collision.** The colour target is a 50/50 mixture
   of a layout and its own channel swap, so the swap maps that distribution
   to itself — a symmetry, `p = q`. The pair now uses a single component.
2. **Denominator reporting was inconsistent between modes**, and the
   `standard` path reported the *floored* denominator, which would have
   hidden kernel collapse behind the floor. Both modes now report the raw
   denominator plus an explicit `denominator_floor_fraction`.
3. **The unbiased U-statistic can be negative**, so it cannot carry the
   plan's `total loss = 0 ⟹ anchor loss = 0` argument. Training uses the
   manifestly nonnegative biased V-statistic; the U-statistic is a
   diagnostic. `objectives.total_objective` refuses the unbiased estimator.
4. **Antithetic frequencies were deliberately not implemented.** The anchor
   summand is even in ω, so a ±ω pair contributes an identical value and the
   "variance reduction" would be an accounting illusion. Structured
   orthogonal direction blocks are used instead, with the Haar sign
   correction that keeps each row marginally uniform on the sphere.

## Scope and honesty boundary

- The exact-zero statement `L_anchor(p,q) = 0 ⟹ p = q` is a property of the
  **ideal expectation** over a full-support spectral measure. Every number
  here comes from a **finite random-feature approximation**, which is an
  unbiased estimator of that expectation and is *not itself* a
  measure-determining family. The training bank is refreshed on a declared
  schedule so no particular finite bank can be mistaken for the identifying
  object, and an independent audit bank — seeded off `AUDIT_SEED`, provably
  unreachable from any training seed — is used for reporting.
- No claim is made that wavelet, scattering or patch features are injective.
  G0.2b measures the opposite.
- Kernel health is not generation quality. G0.3 says a structured kernel is
  better conditioned than a raw pixel kernel on these targets; whether that
  converts into a better generator is precisely the Phase-1 question.
- Cost was recorded, not assumed. "No pretrained encoder" is not a synonym
  for "cheap": the fixed wavelet and kernel banks are inside the training
  ledger (`WorkLedger`), and the geometry arms are 4–20× the wall-clock of
  raw pixel drifting per step at this scale.

## Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.tests.run_all

uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.run_phase0_gate
```

Wall clock for the gate: 274 s excluding the unit tests.
