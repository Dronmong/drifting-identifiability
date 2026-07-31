# Encoder-Independent Kernel Drifting — Phase 25 results

> **§3's inference and §4's closure are WRONG — corrected by
> `EncoderIndependentPhase26Results.md`.** The measurements below stand: teacher
> recall really is 0.000 at every bandwidth and the teacher really is
> bandwidth-invariant. But I inferred from them that this "bounds what the
> generator can learn", and that does not follow — the generator follows a flow,
> and a flow is bounded by its fixed point, not by one step's target at a bad
> cloud. Phase 26 put the cloud **at the real data** and iterated the actual
> training map 40 times: KID stayed at the floor (−0.00017 → +0.00041) and
> recall held at 0.72, while the map moved points 12.3% of their norm. **The
> data distribution is a stable attractor and the objective's fixed point is
> correct.** The zero-recall teacher describes a cloud in the wrong basin, not a
> limit of the method. The pixel-space line is *not* closed; the problem is
> basin of attraction.

## The mean-shift teacher is invariant to bandwidth. Recall is 0.000 at every setting.

*Plan: `EncoderIndependentPhase25Plan.md` (frozen before implementation).
Code: `kernels._row_ess_fraction`, `kernels.degenerate_row_fraction`,
`diagnose_phase25.py`, test 18. Artifact: `phase25_probe.json` (+ `.sha256`),
six teacher grids. 5 000-step cloud, 512 particles, 6 bandwidths × 8
independent positive batches.*

---

## Step 1 — the calibration correction, and why the default did *not* change

The self-paired Gram used by `calibrate_block_kernel` has a zero-distance entry
on every diagonal, so `target_ess_fraction` was met by that self-match rather
than by selectivity among distinct samples:

```
tau from target-only calibration               = 7.714
declared target, diagonal included             = 0.0500
same kernel, same data, diagonal removed       = 0.6019
the FIELD's actual cloud -> positives          = 0.7104
```

The label is wrong by ~12×. **But the plan's instruction to default to the
corrected measurement was wrong, and implementing it revealed why.**

Phase 22 measured the performance optimum in *realized* terms — 0.52–0.71, with
both ends sharply worse. The legacy path's mislabelled 0.05 therefore lands
almost exactly on the good operating point, while a corrected solve at the same
nominal 0.05 lands in the regime measured catastrophic at p = 0.0001. On the
checkerboard dataset it drove τ ~20× smaller and the field degenerate, breaking
a pre-existing test (8d) that pins the RMS-normalized loss at η².

So `exclude_self` defaults to **legacy**, all existing behaviour and artifact
reproducibility are preserved, and the correction is to the *reporting*. Pass
`exclude_self=True` to measure what the field does, and choose a target near
0.5–0.7 rather than 0.05 when calibrating that way.

### A latent numerical bug the correction exposed

Removing the diagonal made the ESS computation diverge: at small τ the surviving
entries are subnormal, the true row sum falls *below* the `clamp_min(1e-30)`
guard, and the clamp then rescales a legitimate row into garbage — **ESS read
1.6e28 at τ = 0.00197**, destroying monotonicity in τ so the bisection ran away
to zero. The self-term had always kept every row O(1), which is why this could
never fire before.

Fixed by normalizing each row by its own maximum before forming weights, which
is exact for scale-invariant weights and keeps every intermediate in [0, 1].
Legacy results are unchanged (with the diagonal present, `row_max ≈ 1`, so the
normalization is a no-op). `degenerate_row_fraction` now reports collapse
directly, and test 18 pins the two measurements apart in both directions.

---

## Step 2 — what the teacher actually is

Bandwidths solved so the **field** realizes each declared ESS, on the
rectangular cloud→positives Gram (no self-match to corrupt it):

| realized ESS | images averaged | τ | drift cosine | teacher variance | alpha | precision | **recall** | teacher KID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.900 | 57.6 | 20.698 | +0.3472 | 0.0337 | 4.383 | 0.600 | **0.000** | +0.19423 |
| 0.710 | 45.4 | 10.368 | +0.4621 | 0.0366 | 4.381 | 0.594 | **0.000** | +0.19313 |
| 0.520 | 33.3 | 6.781 | +0.6832 | 0.0263 | 4.384 | 0.596 | **0.000** | +0.19429 |
| 0.350 | 22.4 | 4.795 | +0.8928 | 0.0304 | 4.386 | 0.590 | **0.000** | +0.19384 |
| 0.240 | 15.4 | 3.709 | +0.8974 | 0.0303 | 4.384 | 0.576 | **0.000** | +0.19230 |
| 0.150 | 9.6 | 2.774 | +0.8777 | 0.0209 | 4.384 | 0.594 | **0.000** | +0.19365 |

*(real data alpha 3.592; no dead rows at any setting)*

**Over a 7.5× range of bandwidth the teacher does not change.** Alpha stays
within 4.381–4.386. KID stays within 0.1923–0.1943. Precision stays within
0.576–0.600. **Recall is exactly 0.000 at every single setting.**

All three candidate explanations are dead:

- **not variance** — the sharp/flat ratio is **0.729**; variance *falls* slightly
  as τ shrinks, the opposite of the hypothesis;
- **not consistency** — the drift-direction cosine *improves* from 0.347 to
  0.898 as τ shrinks (ratio 2.19). Sharpening makes the field **more**
  reproducible across batches, not less;
- **not sharpness** — alpha moves by 0.002 and precision by 0.012 across the
  whole range.

The declared third branch fires: **the mean-shift teacher form is the binding
constraint, and no kernel-family intervention can help.**

---

## 3. The finding, stated plainly

> The teacher is a kernel-weighted average of real samples. As a *set*, those
> averages have **zero recall** against the data distribution at every
> bandwidth, and precision ~0.59 — the hyper-typical signature. The generator is
> regressing onto targets that cover none of the data, and bandwidth does not
> change that. It is a property of the averaging, not of how sharply it is done.

This is why 22 phases of bandwidth, selectivity, mixture, positive-count and
geometry work moved KID between 0.131 and 0.260 and never produced a
recognizable object. Those knobs all reshape *which* average is taken. None of
them can make an average of real samples cover the distribution.

### What this does and does not explain

**Explains the ceiling.** Zero teacher recall at every setting bounds what the
generator can learn, independently of optimization, capacity or budget.

**Does not explain the cliff.** Nothing measured here distinguishes ESS 0.24
(KID 0.260 in Phase 22) from ESS 0.52 (KID 0.147) — the teacher is
indistinguishable at both. So the cliff lives in the training dynamics, not in
the target. It is now the only unexplained pixel-space phenomenon, and it is
also no longer worth chasing: both sides of it are bounded by zero recall.

**A side observation that inverts an assumption.** At the operating point Phase
22 measured as *best* (ESS 0.71) the drift direction reproduces across batches
at cosine 0.462 — barely. At the *worst* setting (0.24) it reproduces at 0.897.
Drift consistency is therefore anti-correlated with performance over this range,
so "make the field more reproducible" is not a route to improvement either.

---

## 4. What follows

**The pixel-space kernel-drifting line is closed.** Not by exhaustion but by
measurement: the target it regresses on has zero coverage at every reachable
bandwidth. The encoder-independence result from Phases 17–18 stands and remains
a statement about a method that does not generate images.

**Nothing warrants an overnight run.** Step 3 of the plan required Step 2 to
name a branch that justified one; it named the branch that forecloses it.

**The one direction with measured headroom** is the Phase-24 autoencoder
ceiling: d=512 reconstructs CIFAR at KID 0.031 with recognizable objects and
precision 0.736, above real data's 0.731. But Phase 24 also showed a plain
autoencoder's code averages decode *worse* than pixel averages (precision 0.352
vs 0.795), so the same zero-recall teacher problem applies there — a
distribution-matched latent space (VAE, or a normalizing-flow code) is the
minimum needed for averages to stay in-distribution. That is a genuine new
build, not a variation, and it should be costed before starting.

---

## 5. Scope

- CIFAR-10 32×32; comparable only between arms measured identically.
- One 5 000-step cloud, one seed. The invariance is measured across 6
  bandwidths × 8 independent positive batches on that cloud, so it is a strong
  statement about *this* cloud and not a seeded performance claim. The
  200-step smoke reproduced every column to within 0.01 on alpha and 0.003 on
  KID, so it is not an artifact of one training length.
- Teacher precision/recall use 512 teachers against 2 048 real samples.
- Bandwidths were solved against generated output, which is legitimate for
  characterization and must never become a declared design rule.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
