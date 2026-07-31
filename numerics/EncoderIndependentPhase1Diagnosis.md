# Encoder-Independent Kernel Drifting — Phase 1 failure diagnosis

*Deep post-hoc investigation of the failed Phase-1 exit gate
(`EncoderIndependentPhase1Results.md`). Code:
`numerics/encoder_independent_drifting/diagnose_phase1.py`. Sealed
artifacts: `phase1_diagnosis.json` (D1–D6, 0.69 h) and
`phase1_capacity.json` (D7–D8, 0.28 h), each with a `.sha256` sidecar.*

**None of these experiments feeds a gate.** They exist to explain a result
that already happened, and several of them overturn claims made in the first
pass. Where that happens it is stated explicitly.

---

## 0. Executive summary

The Phase-1 screen concluded that fixed wavelet geometry with kernel-gradient
movement is 4.5× worse than raw pixel drifting. That **ranking is real and
robust**. The *explanation* offered for it in the first pass was substantially
wrong, and the most important defect was in the screen itself, not in the
method under test.

Eight measurements, in order of how much they change the picture:

| # | Question | Answer | Status |
|---|---|---|---|
| **D7** | Can *any* objective solve this testbed at the frozen budget? | **No.** A sliced-Wasserstein oracle scores 14–30 where A1 scores 8.4, and never reaches target-level precision | **new; reframes everything** |
| **D2** | Did A4 converge to `V = 0`, as first claimed? | **No.** It converges to ≈3× the `q=p` floor — but it *does* beat A1 on its own field while scoring 4.5× worse | **corrects the first pass** |
| **D1** | Is the geometry loss informative? | **No.** It is pinned at exactly η² = 0.25 in all 243 rows, by construction | **new; defect** |
| **D4** | Where does each rule put its output? | Structured kernel-gradient arms sit 2–3.4× further off the data manifold than raw arms; the anchor pulls them back | **new; mechanism** |
| **D5** | Is the bandwidth calibration the cause? | **No.** A 64× bandwidth sweep changes nothing | **new; rules out** |
| **D6** | Is the field normalization the cause? | **No** for quality; **yes** for loss semantics | **new; rules out** |
| **D3** | Was the Phase-0 health probe representative? | Partly — and it exposed an ESS diagnostic that lied in the collapsed regime | **new; defect, fixed** |
| **D8** | What do the failures look like? | Target-dependent: over-dispersion, collapse, or mild shrinkage | **new** |

The revised bottom line: **the Phase-1 screen was run in a regime where no
method reaches the target, so it measured which arm degrades most gracefully
rather than which geometry supplies an image prior.** The two defensible
mechanism findings that survive are D2 (the wavelet objective is optimized
successfully and is the wrong objective) and D4 (kernel-gradient movement
through a non-injective feature map drives the output off the data manifold).

---

## 1. D7 — the testbed cannot be solved at the frozen budget

This is the finding that should have existed before the screen ran, and its
absence is the largest methodological defect in the whole program.

The Phase-1 screen compared nine arms of which the best, A1, achieved median
calibrated precision **0.199** against a target-null level of **0.98**. Every
structured arm achieved **0.000**. A comparison in which no arm approaches the
reference level is a comparison between degrees of failure, and it is only
interpretable if something *can* succeed under the same conditions.

Nothing can. Training the **same generator** with a **sliced-Wasserstein**
objective — chosen precisely because it shares none of the suspected defects
(no kernel, no bandwidth, no feature map, no field normalization) — gives:

| target | steps | score | coverage | precision |
|---|---:|---:|---:|---:|
| checkerboard | 300 | 20.13 | 0.064 | 0.029 |
| checkerboard | 1200 | 13.21 | 1.000 | 0.393 |
| texture_blocks | 300 | 29.89 | 0.000 | 0.000 |
| texture_blocks | 1200 | 8.95 | 0.000 | 0.000 |
| rings_islands | 300 | 14.02 | 0.000 | 0.000 |
| rings_islands | 1200 | 11.00 | 0.000 | 0.000 |

*Reference: Phase-1 median A1 score 8.37; target-null precision 0.98.*

Three consequences:

1. **At the frozen 300-step budget the oracle is worse than A1** (20.1 vs
   ≈5–6 on checkerboard). A1 is not a weak baseline; it is a strong one. But
   it also means the budget is far below what a well-posed objective needs.
2. **Two of the three probed targets are unsolvable even at 4× the budget.**
   `texture_blocks` and `rings_islands` never leave precision 0.000. In the
   full screen, four of nine targets had zero coverage for *every* arm. Those
   cells contributed noise, not signal, to all 27 paired comparisons.
3. **No configuration tested anywhere in this program reaches the null
   precision of 0.98.** The best result in the entire investigation is
   coverage 1.000 / precision 0.393 (SW at 1200 steps on checkerboard).

The Phase-1 gate's *direction* survives this — A4 loses to A1 in 27/27 cells,
under every metric subset (ratios 4.1–12.2; see §6). What does not survive is
the *interpretation*. "Fixed wavelet geometry is not supplying the needed
image prior" asserts something about image priors; what was measured is that
it degrades faster than raw pixels in a regime where the reference objective
also fails.

---

## 2. D2 — the first pass's central mechanism claim was wrong

The first diagnosis said A4 "has converged — to a law that is not the target",
and attributed this to the stop-gradient regression driving the field to
`V = 0` at a non-measure-determining kernel. **The field does not go to zero.**

Raw (unnormalized) field magnitude, measured on identical probes:

| branch | `q = p` floor | trained A1 | trained A3 | trained A4 | random init |
|---|---:|---:|---:|---:|---:|
| raw pixel | 0.0635 | **0.0801** | 0.0820 | 0.1209 | 0.2527 |
| wavelet_s0 | 0.5346 | 2.1604 | 2.2014 | **1.5645** | 8.8979 |
| wavelet_s1 | 0.6389 | 2.6006 | 2.7400 | **2.0897** | 7.8472 |
| wavelet_s2 | 0.3136 | 0.5748 | 0.5766 | 0.5965 | 1.0009 |
| pyramid_global | 0.1266 | 0.1939 | 0.2137 | 0.1780 | 0.6104 |

Read the diagonal, and the picture becomes clean and much more damning than
the original one:

- **A1 genuinely converges.** Its raw-kernel field reaches 0.0801 against a
  finite-sample floor of 0.0635 — a ratio of 1.26. It has run out of signal.
- **A4 does not.** Its wavelet field reaches 1.56 against a floor of 0.53 — a
  ratio of 2.9. It descends a long way (8.90 → 1.56) and then plateaus above
  the floor.
- **A4 beats A1 at A4's own objective.** A4's wavelet_s0 residual (1.56) is
  *lower* than A1's (2.16). The arm optimizing the wavelet discrepancy does
  reduce the wavelet discrepancy more than the arm that ignores it.
- **And A4 is 4.5× worse.** Lower wavelet-field residual, worse law.

An independent measurement confirms it. With normalization disabled (D6) the
loss becomes a real quantity, and the median training loss is:

| arm | median loss, unnormalized |
|---|---:|
| A1 (raw) | **0.0020** |
| A4 (wavelet) | 1.2492 |
| A3 (wavelet, standard) | 31.9287 |

A1 drives its objective to ~0.002; A4 stalls at 1.25.

**The corrected statement:** each arm successfully descends its own
discrepancy, and reducing the wavelet discrepancy does not reduce the source
discrepancy. This is consistent with the Phase-0 collision suite, which
independently measured this geometry as *not* measure-determining (blind to
`color_swap` at p = 0.74). It is the empirical face of
`DriftingIdentifiability/FeatureSpaceIdentifiability.lean` — but via a
partially-descended field, not a vanished one, and the difference matters:
the arm is not sitting at a stationary point of a bad objective, it is still
being driven by one.

---

## 3. D4 — the mechanism: kernel-gradient movement goes off-manifold

This explains *why* the wavelet objective can be reduced without improving
the law, and why the plan's §6.3 prediction reversed.

Distance from generated output to the **nearest real target sample**, in units
of the target's own nearest-neighbour scale (1.0 = as close as a fresh real
sample typically is):

| arm | direction rule | geometry | median distance |
|---|---|---|---:|
| A0 | standard | raw | 1.67 |
| A1 | kernel-gradient | raw | **1.48** |
| A3 | standard | wavelet | 3.22 |
| A4 | kernel-gradient | wavelet | **3.42** |
| A5 | kernel-gradient | wavelet + anchor | 2.91 |

The structural asymmetry is in the two update rules:

- **Standard displacement** moves toward `Σ_j w_j Y_j` — a convex combination
  of *real target images*. The update direction is confined to the span of
  differences to actual data, so the output is pulled toward the data's convex
  hull no matter what the kernel measures.
- **Kernel-gradient** moves along `∇_x log Z_p(x) − ∇_x log Z_q(x)`, an
  arbitrary direction in pixel space. It is free to find pixel-space
  perturbations that reduce the *feature* discrepancy without producing
  feature-plausible images.

When the feature map is the identity (raw kernel), those two coincide up to
scale and there is no off-manifold direction to exploit — which is exactly why
A1 is fine and marginally better than A0. When the feature map is
non-injective, the kernel-gradient rule behaves like **an adversarial attack on
the fixed feature map**: it exploits the map's null directions to lower the
loss without matching the law. That is the mechanism behind the §6.3 reversal
and behind D2's "lower field residual, worse law".

**The anchor counteracts this.** A5 sits at 2.91 versus A4's 3.42 — adding the
spectral anchor pulls the output measurably back toward the data manifold.
That is a coherent explanation for the otherwise surprising Phase-1 result
that the anchor improves every geometry arm by 30–45%: it is the only term in
the objective that constrains the pixel-space law directly, so it partially
closes the feature map's null directions.

---

## 4. D1 — the geometry loss is a constant, by construction

In all **243** Phase-1 rows, every geometry loss — total, per-branch, median
and final — takes exactly one value: **0.25**.

The reason is algebraic. The paper-style stop-gradient regression is

```
L = mean_i |f_i − sg(f_i + η V_i)|² = η² · mean_i |V_i|²
```

and RMS normalization sets `mean_i |V_i|² = 1` identically. So `L = η² = 0.25`
always, for every arm, on every target, at every step. Verified directly
(`D1_loss_constancy`): the RMS-normalized loss equals η² to 1e-4 at
η ∈ {0.25, 0.5, 1.0}, while the unnormalized loss on the same tensors is ≈48.

Three consequences, of decreasing severity:

1. **The plan's exact-zero argument is vacuous for the geometry branch as
   implemented.** Plan §6.5 argues `L_total = 0 ⟹ L_anchor = 0 ⟹ p = q`. But
   `L_geom ≡ η² > 0`, so `L_total = 0` is unreachable and the implication is
   never triggered. The *population* argument is untouched; the implemented
   objective does not instantiate it.
2. **There is no convergence signal.** No stopping rule, no plateau detection
   and no divergence check can read this loss. The screen ran blind.
3. **The generator is pushed with constant force forever.** A unit-RMS field
   with a fixed step is a stochastic-approximation scheme with a
   non-decaying step, so it has a noise floor rather than a fixed point.

**But this is not the cause of the quality gap.** D6 disabled normalization —
restoring an informative, decaying loss — and quality did not improve:

| arm | rms (loss ≡ 0.25) | unnormalized (loss real) |
|---|---:|---:|
| A1 | 6.62 | 7.99 |
| A3 | 13.82 | 35.22 |
| A4 | 36.88 | 37.82 |

So D1 is a defect of **argument and instrumentation**, not of quality. It must
be fixed because the plan's correctness story depends on it, not because it
would have changed the ranking.

---

## 5. D5, D6, D3 — what is *not* the cause

Three candidate explanations were tested and rejected. Recording negatives
matters here: each one was plausible enough to have absorbed a redesign.

**D5 — bandwidth calibration is not the cause.** The declared calibration
solves target-only for median ESS = 0.5, and training visits distances far
outside that scale. A sweep across a **64× bandwidth range** and four
calibration rules changes nothing:

| variant | checkerboard | texture_blocks | rings_islands |
|---|---:|---:|---:|
| declared (ESS 0.5) | 36.88 | 50.06 | 17.93 |
| ESS 0.8 | 40.06 | 49.31 | 18.75 |
| ESS 0.95 | 39.78 | 50.94 | 19.76 |
| median heuristic | 36.23 | 51.33 | 21.38 |
| median × 4 | 39.18 | 49.04 | 17.11 |
| median × 16 | 39.79 | 48.35 | 20.46 |
| median × 64 | 40.12 | 48.54 | 21.30 |

Coverage is 0.000 in all 21 runs. The paper's flat-kernel story is *not* what
is happening here.

**D6 — field normalization is not the cause** (table in §4).

**D3 — the Phase-0 health probe was only partly representative,** and checking
it exposed a real diagnostic bug. Distance in units of the kernel bandwidth:

| cloud | wavelet_s0 | interpretation |
|---|---:|---|
| target vs target | 2.75 | the calibration point |
| Phase-0 G0.3 probe | 20.11 | far out in the tail |
| A4 at initialization | 28.82 | nearly dead (affinity ≈ 1e-4) |
| A4 trained | **2.72** | healthy, same as real data |

So the kernel is nearly dead at initialization and *recovers* to a healthy
regime by convergence. The Phase-0 probe sat near the initialization end, so
G0.3 certified health in the early regime and not along the trajectory. This
is a real methodological weakness but, given D5, not the cause of the failure.

**Defect found and fixed:** in the collapsed regime the ESS diagnostic
reported `ESS_fraction = 1.5e10`. Normalizing an all-underflow row against the
denominator floor gives zero weights and `1/0` — reported as spectacular
health in exactly the regime the plan most wants flagged. `kernel_health` now
excludes collapsed rows, reports `collapsed_row_fraction`, and returns NaN
rather than a fabricated number; `tests/test_kernel_gradients.py` gained a
regression test (65 tests, all passing).

---

## 6. The composite score is ill-conditioned — but the verdict survives

The pre-registered composite divides each metric by its target-vs-target null.
Two of the five components are near-degenerate:

| component | null level | A1 ratio | A4 ratio |
|---|---:|---:|---:|
| ed2 | 0.0194 | 9.9 | 90.3 |
| sw1 | 0.0141 | 3.2 | 9.3 |
| patch_ed2 | 0.0299 | 2.0 | 7.8 |
| **spectral_l1** | **0.00063** | 48.1 | 346.1 |
| **off_support** | 0.0195 | 32.0 | 51.2 |

`spectral_l1`'s null is ≈6e-4, so its ratio explodes and dominates the
geometric mean. `off_support` saturates: once an arm is fully off support its
ratio is pinned at `1/0.0195 = 51.2` and stops discriminating.

The verdict is nevertheless robust, because A4 is worse than A1 on **every
individual component**:

| composite variant | A4/A1 ratio | 95% CI | wins |
|---|---:|---|---:|
| all five (pre-registered) | 4.507 | [3.285, 6.348] | 0/27 |
| drop `spectral_l1` | 4.149 | [3.139, 5.548] | 0/27 |
| drop `off_support` | 5.681 | [3.788, 8.639] | 0/27 |
| drop both | 5.497 | [3.786, 8.057] | 0/27 |
| `ed2` alone | 12.165 | [7.334, 20.412] | 0/27 |

---

## 7. D8 — the failures are not all the same failure

`rms_ratio` (1.0 = correct scale) and `diversity_ratio` (1.0 = correct spread):

| target | A0 | A1 | A3 | A4 | A5 | A7 |
|---|---|---|---|---|---|---|
| checkerboard | 0.97/4.82 | 0.97/7.37 | **0.08/0.57** | 0.94/7.96 | 0.96/8.26 | 0.99/7.56 |
| texture_blocks | 1.08/0.95 | 1.00/0.94 | 0.93/0.74 | 0.76/0.66 | 0.74/0.64 | 0.79/0.70 |
| rings_islands | 1.01/0.84 | 1.01/0.82 | 1.02/0.84 | 0.99/0.80 | 0.99/0.78 | 1.00/0.78 |

Three distinct failure modes, which a single composite score conflates:

- **Over-dispersion** on checkerboard: nearly every arm has the right scale
  but **5–8× the target's spread**. Combined with coverage 1.0 / precision
  ≈0, this is "cloud smeared across the whole region" — consistent with the
  non-decaying constant-magnitude push of D1.
- **Collapse**: A3 on checkerboard sits at `rms_ratio = 0.08` — its output has
  imploded toward zero. Its Phase-1 checkerboard score of 106 is a collapse,
  not a geometry result.
- **Mild shrinkage** on the other two targets (0.74–1.08 scale, 0.64–0.95
  spread) — ordinary under-training, consistent with D7.

---

## 8. Revised causal account

Ordered from the largest effect to the smallest:

1. **The screen was run below the budget at which anything works** (D7). This
   dominates everything else and makes all mechanism conclusions provisional.
2. **Kernel-gradient movement through a non-injective feature map is an
   adversarial channel** (D4, D2). The generator reduces the feature
   discrepancy by exploiting the map's null directions, ending 2–3.4× further
   off-manifold than raw arms. This is a genuine mechanism finding and it is
   consistent across every measurement.
3. **The anchor partially closes that channel** (D4: 3.42 → 2.91), which
   explains why it improves every geometry arm despite barely descending its
   own loss.
4. **Instrumentation was blind** (D1, and the ESS bug in D3). The loss carried
   no information and one health metric inverted its meaning under collapse.
5. **Not causes:** bandwidth (D5), field normalization for quality (D6),
   finite-bank overfitting (established in the earlier anchor diagnosis).

---

## 9. Reforms

Ordered by expected value. R1 and R2 are prerequisites: no further mechanism
claim from this program is interpretable until they are done.

### R1 — every screen must carry an oracle/skyline arm *(blocking)*

Add a well-posed reference objective (sliced Wasserstein is adequate and
cheap) to the frozen protocol, and require:

- the oracle must reach a declared fraction of the target-null precision on a
  target, or **that target is excluded** from the screen;
- the training budget is raised until the oracle clears that bar, and the
  budget is frozen from the *oracle*, not from the baseline arm;
- the oracle row is reported alongside every arm.

Had this been in the Phase-1 protocol, the screen would not have been run in
its current form. The budget was frozen from A1's *coverage*, which reached
1.0 while precision was still 0.199 — coverage saturates early and was the
wrong quantity to freeze on.

### R2 — make the objective's value mean something *(blocking for the theory)*

- Report the **unnormalized** loss always; if RMS normalization is kept for
  step-size control, apply it to the update and keep the loss honest.
- Add a Phase-0 unit test asserting that the geometry loss is a non-constant
  function of the field along a real trajectory — the existing test used a
  literal zero drift and so passed against a defect that is invisible only in
  the pipeline.
- Re-state plan §6.5: the exact-zero argument holds for the population
  objective, and the implemented objective must be shown to instantiate it.

### R3 — replace or constrain the kernel-gradient rule *(highest mechanism value)*

D4 identifies the defect precisely, and it suggests a repair that keeps the
geometry's *weighting* while restoring the raw rule's *support constraint*:

- **R3a (cheapest):** use standard displacement with structured kernels. A3
  already beats A4 by 1.86×, and the plan's §6.3 rationale is falsified.
- **R3b (the interesting one):** project the kernel-gradient direction onto
  the span of `{Y_j − x}` — the subspace the standard rule is confined to.
  This keeps the structured kernel deciding *which* targets matter and *how
  much*, while forbidding off-manifold directions. It is a one-line change to
  `kernel_gradient.field` and is directly motivated by a measurement.
- **R3c:** keep the anchor in every structured arm; D4 shows it is already
  acting as a partial manifold constraint.

### R4 — re-specify the composite score

- Floor or drop `spectral_l1` (null ≈6e-4 makes its ratio meaningless).
- Replace the saturating `off_support` with the graded D4 statistic (median
  nearest-real distance in target-NN units), which keeps discriminating after
  an arm leaves the support.
- Require the verdict to hold on a majority of components individually, as
  §6 does post-hoc, rather than trusting the aggregate.

### R5 — Phase 0 must test the zero-set, not just kernel health

The single cheapest predictor of the Phase-1 failure is D2, and it costs one
short training run per geometry: **minimize each candidate field from a random
initialization and check whether the residual reaches the `q = p` floor.** A
geometry whose field plateaus at 3× the floor while a raw kernel reaches 1.3×
should never have been promoted to a full screen. Add as a Phase-0 condition,
alongside:

- kernel health measured **along the optimization trajectory** (init / mid /
  converged), not on a synthetic probe (D3);
- the collapsed-row fraction reported for every branch (now implemented).

### R6 — anchor frequency schedule *(carried forward, still untested)*

Unchanged from `EncoderIndependentAnchorGradientDiagnosis.md`: the anchor
gradient scales with ‖ω‖, so the high band contributes the largest gradient
while carrying pure noise. Begin coarse and anneal toward high frequencies,
with the audit bank held at full width. D4 adds a new reason to want this: the
anchor is the component that keeps the generator on-manifold, so making it
actually descend is worth more than the first pass suggested.

### R7 — rebuild the target suite

Four of nine targets had zero coverage for every arm, and D7 shows two of
three probed targets are unsolvable even at 4× budget. Either raise the budget
under R1 until the oracle clears them, or replace them. Separately, if the
program's motivation is the *high-dimensional* regime where a raw kernel
degrades, then a 768-dimensional testbed on which raw pixel drifting is the
best arm cannot answer the question — the successor needs targets where the
raw kernel demonstrably fails.

---

## 10. What the program may now claim

**Survives:**
- The Phase-0 implementation results (Haar control, Gaussian field control,
  anchor 6/6 collision detection, measured geometry blindness).
- The Phase-1 *ranking*: A4 loses to A1 in 27/27 cells under every metric
  subset. Fixed wavelet geometry with kernel-gradient movement is worse than
  raw pixel drifting on this testbed.
- The §6.3 falsification: kernel-gradient is worse than standard displacement
  under a structured kernel, and D4 supplies the mechanism.
- The anchor's three Phase-1 conditions (helps, present, covers blindness),
  now with a mechanistic explanation in D4.

**Withdrawn or downgraded:**
- "A4 has converged to a law that is not the target" — **withdrawn**. It
  converges to ≈3× the finite-sample floor and is still being driven (D2).
- "Fixed wavelet geometry is not supplying the needed image prior" —
  **downgraded** to a statement about relative degradation in a regime where
  the reference objective also fails (D7).
- Any use of the geometry loss as evidence of convergence — **withdrawn** (D1).
- The Phase-0 G0.3 kernel-health certification as evidence about training
  conditions — **downgraded**; it characterized the early regime only (D3).

**Never claimed and still not claimed:** anything about CIFAR-10, ImageNet,
FID, natural images, the paper's model, or the exactness of the finite
random-feature anchor.
