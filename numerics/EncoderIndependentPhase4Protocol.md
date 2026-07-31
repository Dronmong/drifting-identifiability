# Encoder-Independent Kernel Drifting — Phase 4 protocol

## Is the teacher contraction general, and is it mechanistic?

*Frozen pre-outcome design. Every threshold was fixed before any Phase-4 run.
Design evidence: `EncoderIndependentPhase4Design.md` (G1, G2). Results go to
`EncoderIndependentPhase4Results.md`.*

---

## 1. The claim under test

> Stop-gradient regression onto a mean-shift teacher is
> **variance-contracting**; the contraction compounds over training; matching
> the teacher's second moment to the data's removes it. This holds across
> batch sizes, resolutions and kernel geometries, at the paper's own declared
> operating point, and in an independently written harness — and it follows
> from regression attenuation, not from the field.

This is a claim about a **training recipe**, not about encoder independence.
Phases 0–3 asked whether fixed geometry helps (no, three times) and whether a
particular repair works (yes, confirmed). Phase 4 asks whether that repair
describes something general.

---

## 2. The predicted mechanism

The design investigation measured the contraction but did not explain it. The
protocol commits to an explanation in advance so that P4C can falsify it.

For a fixed latent `z`, the teacher `x + eta*V` depends on the random target
minibatch, so it fluctuates across steps. Least-squares regression converges
to the **conditional mean** of the teacher given `z`, whose variance is

```
Var(fit) = Var(teacher) - E[ Var(teacher | z) ]
```

so each step loses variance in proportion to the teacher's *minibatch noise*.
Iterated, this compounds into collapse. The prediction that makes it
falsifiable:

> **the contraction weakens as the batch grows**, because a larger batch
> lowers `Var(teacher | z)`.

The design data are already consistent — effective dimension 0.227 (batch 32),
0.232 (64), 0.301 (128) — but that was not a preregistered test and P4A/P4C
must confirm it on fresh seeds.

Note this locates the defect in the **regression**, not the field. In the
tractable Gaussian case the field is *expansive* when the cloud is too
narrow, which is further reason to expect the regression to be the culprit.

---

## 3. P4A — generality grid

One factor at a time around a declared base point, `R11 ∈ {off, on}`, three
fresh seeds (`MASTER_SEED + 3000..`), CIFAR-10.

Base point: raw geometry, batch 64, resolution 16, ESS-calibrated bandwidth,
self-mask off, paper Algorithm-2 field.

| axis | values |
|---|---|
| batch | 32, **64**, 128 |
| resolution | **16**, 32 |
| geometry | **raw**, wavelet |
| **temperature** | ESS-calibrated (base), τ = 0.02, 0.05, 0.2 |
| **self-mask** | **off**, on |

The temperature and self-mask axes have **never been tested in this program**
and are the paper's declared operating point (`numerics/README.md`, Table 8 /
A.6): τ in normalized units where the mean pairwise feature distance is 1, and
Algorithm 2's `eye(N)*1e6` self-mask. If the contraction survives them, the
claim is about the paper's recipe rather than about my parameter choices.

**Gate P4A:** in **every** cell, R11's paired ratio ≤ 0.60 **and** the
effective-dimension ratio rises above 0.60 with R11 while sitting below 0.50
without it.

## 4. P4B — cross-harness replication *(the pivot)*

Reproduce the contraction using **`numerics/lowdim_drift.py`** — this
repository's independently audited, verbatim Algorithm-2 harness, cross-checked
against `driftlab.compute_v_paper` and carrying its own Lean crosswalk.

`lowdim_drift.py` is **not modified**: its `drift_paper` field and its target
families are imported as-is. Only the generator and training loop are new,
because that harness is particle-based and has none.

Targets: `gauss_mixture` at dimensions 8 and 16 (room to collapse), plus
`ring_target` and `moons_target` at their native dimension.

**Gate P4B:** without the fix the effective-dimension ratio is < 0.50; with it
> 0.80; in a harness this program did not write, on ≥ 3 target families.

A failure here shrinks R11's scope to "a local fix to this package" and must
be reported that way.

## 5. P4C — analytic characterization

Three steps, each falsifiable:

1. **Measure the teacher's conditional noise** directly: hold the latent
   fixed, recompute the teacher against independent target batches, and
   estimate `Var(teacher | z) / Var(teacher)`.
2. **Predict** the per-step contraction from it, and the compounded fixed
   point over the training budget.
3. **Compare** the prediction with the measured effective-dimension ratio
   across batch sizes.

**Gate P4C:** the predicted and measured effective-dimension ratios agree in
**ordering across batch size**, and within a factor of 1.5 in magnitude, on at
least three batch sizes.

Ordering is the primary criterion. A mechanism that gets the direction and
scaling right but the constant wrong is still informative; one that predicts
the wrong ordering is refuted.

## 6. P4D — residual-gap diagnostic *(scoped: one experiment)*

After R11 the corrected baseline is ≈1.15–1.23× the sliced-Wasserstein
skyline. Compare the **covariance spectra** (not just the participation ratio)
of the corrected generator, the skyline generator, the free-particle cloud and
real data, to localize what remains.

No gate. This is exploratory and may return "estimator noise", which is an
acceptable outcome. It must not expand into a programme.

---

## 7. Declared failure branches

- **P4A fails in some cells** → the contraction is real but conditional;
  report which axis breaks it and scope R11 accordingly.
- **P4B fails** → R11 is a local fix to this package. Withdraw the general
  claim, keep the Phase-3 confirmation as an implementation result, and stop.
- **P4C fails** → the repair works but the regression-attenuation account is
  wrong. Report the effect, withdraw the mechanism, do not substitute a new
  story without evidence.
- **No retuning of R11 in any branch.** It has no free parameter; adding one
  would convert a mechanism into a knob.

## 8. What Phase 4 cannot conclude

- Nothing about fixed compositional geometry: three negatives, plus a
  re-check under R11. That thread stays closed.
- Nothing about the anchor beyond the ~3.5% it contributes on raw geometry;
  it appears in P4A as a reported secondary factor, not a subject.
- Nothing about ImageNet, FID, or the paper's trained model. Even a clean pass
  says *this training recipe contracts and can be corrected* — the paper
  trains with a feature encoder at a scale and batch this does not touch and
  may already avoid it by other means.
- A positive result is scoped to: *CIFAR-10 at 16–32 px and low-dimensional
  synthetic targets, one generator family per harness, three fresh seeds.*
