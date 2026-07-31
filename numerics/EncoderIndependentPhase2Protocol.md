# Encoder-Independent Kernel Drifting — Phase 2 protocol (CIFAR-10)

*Frozen pre-outcome design. Every threshold below was fixed before any Phase-2
arm was run. Supersedes `EncoderIndependentReformedScreenProtocol.md`, whose
synthetic target suite was retired by the regime search in
`EncoderIndependentSecondPassAudit.md` (§4). Results go to
`EncoderIndependentPhase2Results.md`.*

Entry evidence already on record:

| fact | source |
|---|---|
| pixel k-NN content accuracy .267 vs wavelet .390 on CIFAR-16 | second-pass audit A5 |
| a learned autoencoder reaches only .282 — barely above pixels | audit A5 |
| CIFAR-16 admissible at 300 steps, skyline precision 1.000 | audit A5 |
| no synthetic testbed makes pixel geometry fail | audit A4 |
| `wavelet::kernel_gradient` zero-set plateaus at 2.39; `wavelet::standard` reaches .68 | Phase-0 G0.5 |

---

## 1. The question

> On real images, where pixel geometry is measurably weak, does a **fixed
> compositional kernel** beat a **raw pixel kernel** for drifting — with no
> pretrained encoder anywhere in the training objective?

Phases 0–1 could not ask this: their synthetic targets had pixel geometry that
already worked, so fixed geometry had nothing to win. CIFAR-16 is the first
testbed in this program that is simultaneously **solvable** (skyline precision
1.000 at 300 steps) and **discriminating** (pixel k-NN .267 against wavelet
.390).

---

## 2. Data

CIFAR-10, downsampled to 16×16 by area interpolation, scaled to `[-1, 1]`.

**Disjoint splits, fixed once.** The 50,000 training images are partitioned by
index: `train = [0, 40000)`, `eval = [40000, 50000)`. Arms, calibration and the
skyline draw **only** from `train`; every evaluation, null and support-
calibration pool draws **only** from `eval`. Class labels are never used —
not by an objective, not by a controller, not by a metric. They exist in the
dataset and are ignored.

No pretrained network is loaded at any point.

---

## 3. Phase 2A — entry gate

Run before any arm. All four conditions must hold; the artifact records them
whether they pass or fail.

| ID | Condition | Threshold |
|---|---|---|
| **E1** | the testbed is solvable | skyline precision ≥ 0.5 × null precision |
| **E2** | pixel geometry is the bottleneck | wavelet or scattering k-NN ≥ 1.25 × pixel k-NN |
| **E3** | each arm's field can reach the target | G0.5 residual / (q=p floor) ≤ 2.0 for every `(geometry, direction)` pair that an arm uses |
| **E4** | no kernel is collapsed | `collapsed_row_fraction` = 0 for every admitted branch |

**E3 is the condition Phase 1 lacked.** An arm whose field provably plateaus
above the floor is *not run*. On synthetic targets this rejected
`wavelet::kernel_gradient` (2.39) while admitting `wavelet::standard` (0.68);
it must be re-measured on CIFAR because reachability is a property of the data
as much as of the kernel.

The **budget is derived from the skyline**, not from a baseline arm: the
smallest of `(300, 600, 1200)` at which E1 holds, then one step up for
headroom. Freezing on a baseline's *coverage* is what under-budgeted Phase 1.

---

## 4. Phase 2B — arms

Four arms and a skyline. Phase 1 ran nine and learned little, because eight
were variations on a mechanism G0.5 now rejects in seconds.

| ID | Anchor | Geometry | Direction | Purpose |
|---|---|---|---|---|
| **B0** | — | raw pixel | standard | the baseline; won Phase 1 outright |
| **B1** | — | wavelet | standard | the +46% k-NN candidate |
| **B2** | — | scattering | standard | the other +45% candidate |
| **B3** | scheduled | wavelet | standard | does the anchor still help? |
| **SKY** | — | — | sliced Wasserstein | skyline; **never in a gate** |

Every arm uses **standard displacement**. The plan's §6.3 kernel-gradient
hypothesis is *abandoned, not re-tested*: Phase 1 measured it 1.86× worse
under a structured kernel, G0.5 rejects its zero-set, and R3's data-span
projection does not repair it (2.21 vs 2.39).

**All arms use the same base kernel** (`smooth_laplace`). Phase 1 gave the raw
standard arm the paper's non-smooth `laplace` kernel and everything else the
smooth one, confounding geometry with kernel smoothness. Fixed here.

Matched across arms within a cell: generator initialization, latent stream,
target-minibatch stream, cross-fitting role split, calibration sample, budget,
batch sizes, and optimizer. Arms differ **only** by geometry and by whether
the anchor is present.

---

## 5. Measurement

- Composite: `normalized_geometry_score_v2` over `(ed2, sw1, patch_ed2,
  nearest_real)`, with `spectral_l1` and `off_support` reported but not
  scored (reform R4).
- Null: median over 5 independent draws, with `null_spread` reported
  (reform R8). The achievable floor is ≈1.2–1.7, so differences below that
  are not claimed.
- Per-component verdicts reported alongside the aggregate.
- Reported every run: unnormalized geometry loss (R2), full-band anchor loss
  (R6), collapsed-row fraction, ESS, drift SNR, and the full cost ledger.

## 6. Exit gate

| ID | Condition | Threshold |
|---|---|---|
| **P2.1** | a fixed-geometry arm beats raw pixels | best of B1/B2 vs B0: paired v2 ratio ≤ 0.90, bootstrap upper bound < 1 |
| **P2.2** | the win is not one metric's artifact | that arm wins a **majority of components** individually |
| **P2.3** | the win is not one seed | ratio < 1 on **every** seed |
| **P2.4** | the anchor is not destructive | B3 vs B1 paired ratio ≤ 1.25 |
| **P2.5** | the objective is not vacuous | unnormalized geometry loss falls ≥ 25% from its first logged value, in every arm |
| **P2.6** | the result is within reach of the skyline | winning arm's v2 score ≤ 2.5 × SKY's |

P2.5 and P2.6 are new. P2.5 is impossible to fail silently only because of
reform R2 — under Phase-1 instrumentation the loss was pinned at η² and could
not have been checked. P2.6 prevents declaring victory in a regime where every
arm is far from what a well-posed objective achieves, which is exactly how
Phase 1 became uninterpretable.

**Compute is reported, never claimed as an advantage.** The structured arms
cost 4–20× the wall-clock of raw pixel drifting at this scale and the ledger
says so.

---

## 7. Declared failure branches

- **If E2 fails on CIFAR** the regime evidence does not replicate and Phase 2B
  is not run.
- **If E3 rejects wavelet and scattering under standard displacement**, no
  fixed-geometry arm is runnable and the program's geometry thread is closed;
  the anchor and zero-set threads (second-pass audit §7.4) take over.
- **If P2.1 fails**, a 46% better neighbour ranking does not convert into
  better drifting. Given G0.5 and the Phase-1 diagnosis, that points at the
  objective rather than the geometry, and it must be reported as the third
  consecutive negative for fixed compositional geometry — not retuned.

## 8. What Phase 2 cannot conclude

- Nothing about ImageNet, FID, the paper's model, or 32×32 and above.
- Nothing about pretrained encoders: none is loaded, and the audit's
  supervised control was a small CNN on 2048 images, not a real encoder.
- Nothing about cost advantage.
- "The finite random-feature anchor is characteristic" remains false; it is an
  unbiased estimator of an ideal expectation.
- A positive result would be scoped to: *CIFAR-10 at 16×16, one generator, one
  budget, three seeds, encoder-free throughout.*
