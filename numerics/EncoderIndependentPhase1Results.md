# Encoder-Independent Kernel Drifting — Phase 1 results

*Executes `EncoderIndependentPhase1Protocol.md` (frozen before the run).
Sealed artifact: `numerics/encoder_independent_drifting/phase1_screen.json`
(+ `.sha256`), 243 rows over 27 cells, 4.00 h. Regenerate every table below
with `analyze_phase1_screen.py`. Mechanism diagnosis:
`EncoderIndependentAnchorGradientDiagnosis.md`.*

> **Superseded in part by `EncoderIndependentPhase1Diagnosis.md` (deep
> investigation).** The gate verdict and the arm ranking below stand. Two
> interpretive claims do not:
>
> * §"Failed: fixed compositional geometry as an image prior" asserts that A4
>   "has converged to a law that is not the target". **Withdrawn** — the field
>   plateaus at ≈3× the `q=p` floor and is still being driven (diagnosis D2).
> * The screen ran at a budget where **no** objective succeeds: a
>   sliced-Wasserstein oracle on the same generator scores worse than A1 at
>   300 steps and never reaches target-level precision (diagnosis D7). The
>   conclusion is therefore about relative degradation, not about image
>   priors.
>
> Read the diagnosis before quoting any mechanism claim from this document.

## Verdict

**Phase-1 exit gate: FAIL.** The program does not proceed to Phase 2
(CIFAR-10). Two of five conditions failed, and the failing one is the
central hypothesis.

| ID | Condition | Result | Verdict |
|---|---|---|---|
| G1.1 | A4 materially beats A1 | ratio **4.507** [3.285, 6.348], **0/27** wins | **FAIL** |
| G1.2 | A5 within 10% of A4 | ratio **0.701** [0.616, 0.792], 24/27 wins | **PASS** |
| G1.3 | A5 passes collisions A4 fails | anchor 6/6 in all seeds; A4 geometry blind in all seeds | **PASS** |
| G1.4 | anchor practically present | gradient share 0.311, present 27/27 cells | **PASS** |
| G1.5 | gains hold across seeds/targets | A4 loses on 9/9 targets and 3/3 seeds | **FAIL** |

The headline is unambiguous: **fixed wavelet geometry with kernel-gradient
movement is 4.5× worse than raw pixel drifting on the pre-registered
normalized geometry score, and loses every one of 27 paired cells.** Per plan
section 9, "if A4 fails, fixed wavelet geometry is not supplying the needed
image prior."

## Arm results

Median over 9 targets × 3 seeds. Score 1.0 = indistinguishable from a fresh
real sample under the frozen metric set; lower is better.

| arm | geometry | direction | score | ED² | precision | coverage | wall s | kernel pairs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A0 | raw pixel | standard | 9.74 | 0.353 | 0.004 | 0.186 | 13.8 | 2.5e6 |
| **A1** | raw pixel | kernel-gradient | **8.37** | 0.186 | 0.199 | 0.945 | 14.6 | 2.5e6 |
| A2 | none (anchor only) | anchor gradient | 47.32 | 2.384 | 0.000 | 0.000 | 14.1 | 0 |
| A3 | wavelet | standard | 21.57 | 1.744 | 0.000 | 0.000 | 34.9 | 1.2e8 |
| A4 | wavelet | kernel-gradient | 50.86 | 5.333 | 0.000 | 0.000 | 65.5 | 1.2e8 |
| A5 | anchor + wavelet | kernel-gradient | 39.06 | 3.683 | 0.000 | 0.000 | 66.1 | 1.2e8 |
| A6 | anchor + randconv | kernel-gradient | 37.01 | 4.097 | 0.000 | 0.000 | 46.7 | 8.3e7 |
| A7 | anchor + dictionary | kernel-gradient, adaptive | 25.80 | 2.133 | 0.000 | 0.000 | 256.1 | 4.3e8 |
| A8 | *local encoder stand-in* | standard | 11.86 | 0.900 | 0.051 | 0.477 | 20.0 | 4.0e7 |

Paired geometric-mean ratios with 95% paired-bootstrap intervals:

| comparison | ratio | 95% CI | wins |
|---|---:|---|---:|
| A1 / A0 — kernel-gradient vs standard, **raw** kernel | 0.981 | [0.715, 1.442] | 22/27 |
| A4 / A3 — kernel-gradient vs standard, **wavelet** kernel | **1.862** | [1.045, 3.234] | 13/27 |
| A3 / A1 — best wavelet arm vs raw | 2.420 | [1.402, 4.171] | 4/27 |
| A5 / A4 — adding the anchor to wavelet | **0.701** | [0.616, 0.792] | 24/27 |
| A6 / A4 — random conv + anchor | 0.658 | [0.584, 0.735] | 26/27 |
| A7 / A4 — adaptive dictionary + anchor | 0.549 | [0.415, 0.729] | 24/27 |
| A5 / A2 — adding geometry to the anchor | 0.534 | [0.358, 0.713] | 23/27 |
| A7 / A1 — best structured arm vs raw | 2.475 | [1.858, 3.377] | 2/27 |
| A8 / A1 — learned encoder stand-in vs raw | 1.322 | [0.872, 1.862] | 4/27 |

## What failed, and what did not

### Failed: fixed compositional geometry as an image prior

Every structured-geometry arm is worse than raw pixel drifting, and the best
of them (A7, the adaptive dictionary) is still 2.48× worse with only 2/27
paired wins. This is the plan's central hypothesis and it is not supported at
mechanism scale.

A4 at 4× the step budget gets *slightly worse*, not better, so more of the
same training does not rescue it.

**Corrected mechanism (see `EncoderIndependentPhase1Diagnosis.md` §2).** The
first pass read that plateau as convergence to `V = 0` at a
non-measure-determining kernel. Direct measurement refutes this: A4's field
plateaus at ≈3× the `q = p` finite-sample floor while A1's reaches 1.26× it,
so A4 is still being driven, not sitting at a stationary point. The accurate
statement is stronger and stranger — **A4 achieves a *lower* wavelet-field
residual than A1 (1.56 vs 2.16) and is 4.5× worse.** Each arm descends its own
discrepancy successfully; reducing the wavelet discrepancy does not reduce the
source discrepancy. Diagnosis §3 supplies the mechanism: kernel-gradient
movement through a non-injective feature map exploits the map's null
directions, leaving the output 2–3.4× further off the data manifold than the
raw arms.

### Falsified: the kernel-gradient prediction of plan section 6.3

The plan predicted that kernel-gradient movement is "more compatible with the
structured kernel than raw Euclidean displacement". The required ablation
reverses it:

- with the **raw** kernel, kernel-gradient is mildly better (A1/A0 = 0.981,
  22/27 paired wins, though the interval crosses 1);
- with the **wavelet** kernel, kernel-gradient is **1.86× worse** than
  standard displacement (A4/A3, interval excludes 1).

So the direction rule that helps in pixel space actively hurts in feature
space, which is the opposite of the design rationale. Phase 0 already showed
the two fields become orthogonal under a structured kernel (cosine −0.002);
Phase 1 shows which of the two orthogonal directions is the useful one, and
it is not the kernel-gradient. Keeping `standard_displacement` as a required
ablation is what caught this.

### Held: every claim made for the anchor

The anchor is the one part of the architecture that behaved exactly as
designed, and all three of its conditions passed:

- **it helps, consistently.** Adding it to fixed wavelet geometry improves
  the score by 30% (A5/A4 = 0.701, 24/27 wins), and the same effect appears
  for random-convolutional geometry (A6/A4 = 0.658) and the dictionary
  (A7/A4 = 0.549). It is not a correctness tax paid for in quality.
- **it is practically present, not rhetorical** (G1.4). Median gradient
  share 0.311, above the frozen 0.05 threshold in 27/27 cells for A5, A6
  and A7.
- **it covers geometry blindness** (G1.3). Across all three seeds, A5's
  independent *audit* bank detects 6/6 source collisions, while A4's wavelet
  branches repeatedly miss `color_swap` and `rare_mode_drop`:

  | seed | A4 wavelet branches | A5 anchor |
  |---|---|---|
  | 20260724 | 4–5 / 6, missing `color_swap`, `rare_mode_drop` | 6/6 |
  | 20260725 | 4–5 / 6, missing `color_swap`, `rare_mode_drop` | 6/6 |
  | 20260726 | 4–5 / 6, missing `color_swap`, `rare_mode_drop` | 6/6 |

Geometry and anchor also help each other: A5/A2 = 0.534 says the geometry
branch substantially improves the anchor-only arm too. The two-branch
architecture is doing what the plan wanted — it is the *fixed geometry* that
is not competitive with raw pixels, not the pairing.

## The caveat that limits every conclusion

*(The deep investigation found a second, larger one: at this budget a
sliced-Wasserstein oracle on the same generator also fails — scoring worse
than A1 at 300 steps and never reaching target-level precision on any target.
See `EncoderIndependentPhase1Diagnosis.md` §1. The screen measured which arm
degrades most gracefully, not which geometry supplies an image prior.)*

**The locally trained encoder stand-in also fails to beat raw pixels**
(A8/A1 = 1.322, interval [0.872, 1.862] crossing 1, 4/27 wins). At 16×16
with these targets, raw pixel distance is already a good geometry, so this
testbed **cannot separate "fixed compositional geometry is inadequate" from
"any feature geometry is unnecessary at this scale"**.

This matters because the paper's motivation for a feature encoder is a
high-dimensional phenomenon — distance concentration, vanishing affinities,
collapsing effective sample size. A 768-dimensional testbed on which the raw
kernel achieves 0.945 coverage is not exhibiting that regime. The screen
therefore falsifies the mechanism *as a way to beat raw pixels at 16×16*; it
does not establish that fixed geometry would fail where an encoder is
actually needed.

A8 is in any case a weak stand-in — a small autoencoder trained from scratch
on the target family — and is **not** the paper's pretrained encoder. It is
context, and was excluded from every gate.

## Cost

"No pretrained encoder" is not a synonym for cheap. At matched target-example
consumption (19,200 field examples for every arm):

| arm | kernel pairs | × A1 | wall s | × A1 | anchor feature evals |
|---|---:|---:|---:|---:|---:|
| A1 | 2.5e6 | 1× | 14.6 | 1× | 0 |
| A3 | 1.2e8 | 49× | 34.9 | 2.4× | 0 |
| A4 | 1.2e8 | 49× | 65.5 | 4.5× | 0 |
| A5 | 1.2e8 | 49× | 66.1 | 4.5× | 9.8e6 |
| A7 | 4.3e8 | 169× | 256.1 | 17.5× | 9.8e6 |

A4 buys 49× the kernel work and 4.5× the wall clock for 4.5× *worse* quality.
The plan's stop condition "fixed feature cost approaches or exceeds the
avoided pretrained encoder without a correctness or robustness benefit"
(section 10.4) is met on the cost side; the correctness benefit that does
exist comes from the anchor, which is nearly free (9.8e6 feature evaluations,
no measurable wall-clock share).

## Per-target detail

Median geometry score (lower better):

| target | A0 | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkerboard | 6.85 | 6.31 | 126.31 | 106.49 | 68.56 | 49.56 | 50.66 | **10.64** | 8.70 |
| color_layout | 10.56 | 6.23 | 19.55 | 117.99 | 21.93 | **7.80** | 9.42 | 9.99 | 8.69 |
| deformed | 31.19 | **24.84** | 46.50 | 62.48 | 50.86 | 45.70 | 41.48 | 43.72 | 71.44 |
| patch_layout | 6.81 | 6.12 | 133.58 | **7.14** | 110.04 | 104.06 | 89.98 | 77.51 | 7.65 |
| phase_structured | 19.03 | **13.16** | 47.47 | 19.32 | 20.18 | 15.58 | 18.63 | 56.70 | 30.94 |
| pinwheel | 88.01 | 25.61 | 37.84 | 90.18 | 43.82 | 37.10 | 30.67 | **25.80** | 87.69 |
| rare_object | **0.50** | 6.40 | 34.41 | 0.58 | 15.49 | 6.94 | 5.60 | 7.16 | 0.59 |
| rings_islands | **10.44** | 9.56 | 65.52 | 20.54 | 84.51 | 38.17 | 58.93 | 19.29 | 22.77 |
| texture_blocks | 8.08 | **7.89** | 47.32 | 14.72 | 57.42 | 60.67 | 37.01 | 42.45 | 10.60 |

Three targets defeat every arm — `deformed`, `pinwheel`, `rings_islands` and
`texture_blocks` all sit at zero coverage across the board. These are the
continuous-manifold and fine-texture families, and no arm in this screen
learns them at 300 steps. A0's score of 0.50 on `rare_object` is not a
success either: that target is 95% background, so an arm that produces only
background scores well on the composite while missing the rare mode entirely.

## Declared next steps

The plan's own failure branches apply, in this order:

1. **Do not proceed to Phase 2.** The gate is explicit and it failed.
2. **The convolutional kernel was already tested** (plan: "test the
   convolutional kernel before considering a learned encoder"). A6 beats A4
   but is still 4.4× worse than raw. That branch is exhausted at this scale.
3. **Fix the anchor's gradient before anything else.** The diagnosis document
   shows the anchor loss barely descends with the declared three-band bank
   (6% over 600 steps) because the gradient is proportional to ‖ω‖ and the
   high band contributes the largest gradient norm while carrying pure noise.
   A coarse-only bank descends 96%. The indicated repair is a coarse-to-fine
   frequency schedule with a full-width audit bank — motivated by the plan's
   own section 5.1 citation and untested.
4. **Rebuild the testbed before re-running the geometry question.** The A8
   result shows this testbed cannot reward feature geometry at all. Any
   successor screen needs targets where the raw pixel kernel demonstrably
   degrades — higher resolution, or explicit distance concentration — or it
   will keep answering a question nobody asked.
5. **Retain the anchor.** It is the one component that passed every condition
   asked of it, it is nearly free, and it is the only part of the
   architecture with a source-space correctness story.

## Scope

This screen establishes nothing about CIFAR-10, ImageNet, FID, natural
images, or the paper's trained model. It is 16×16 synthetic structured
targets at 300 steps with one small generator. The anchor's exact-zero
property remains an ideal-expectation statement; every number here comes from
a finite random-feature approximation. No pretrained encoder entered any
training objective, controller decision, or metric. A8 is a locally trained
stand-in and is excluded from every gate. Development results, not
confirmation — no fresh sealed registry was opened, because the candidate did
not earn one.
