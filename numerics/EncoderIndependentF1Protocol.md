# Stage F1 protocol — deployment-relevant free-particle basin test

**Status: DESIGN v3, revised under the §13 and §15 audits. Not executed; confirmatory run remains NO-GO per §15.1 until the runner exists and §11 steps 2-8 pass.**
Implements §19.5 of `EncoderIndependentDriftFlowResearch.md` plus §20.2, §18.4
G2/G3. **§13 audits v1 and §15 audits v2, both retained as the record; §14 and
§16 map every audit item to the revision applied.**

---

## 1. Question

> Can the current encoder-free particle teacher reach nontrivial coverage from a
> distribution that the deployed generator actually produces?

The C2 gate for the autonomous-drift branch. F1 involves **no generator
training**: a generator supplies only the initial cloud, after which particles
move freely. That isolates the particle basin from the amortization failure of
Phases 28–30.

Motivating result: **F0 failed** (`phase30.json`). Capacity over a 7.5× range and
a 4× batch increase leave recall at ~0, so the deficit is not the generator's
expressiveness — Phase 28's width-64 net holds recall 0.224 by memorizing.

---

## 2. Pre-flight, which may veto this protocol

Nothing in §§3–9 runs until §2.1 and §2.2 complete and their artifacts are
hashed. Either may return NO-GO, as Phase 30's pre-flight did.

### 2.1 Recall estimator null calibration — **the gate depends on this**

The 0.05 threshold is calibrated in *scale* (anchors 0.000 ×6, 0.224, 0.496,
0.737) but its **behaviour near zero has never been measured**. v1 proposed
`max(0.05, 5·SD)` from 20 subsets of an 8 192 pool. That is withdrawn: the
formula presumes a zero-mean, approximately Gaussian null for a statistic that
is bounded, discrete, and defined by a *sample-dependent* k-NN manifold
(Kynkäänniemi et al. 2019), and the construction was arithmetically impossible
— 20 × 512 = 10 240 > 8 192, so "independent subsets" could not be disjoint.

**Replacement.** For each reference state, build **200 independently generated
512-sample sets** and compute recall against the fixed 2 048-sample real
reference used by F1 itself.

Reference states:

| state | role |
|---|---|
| `identical_images` | **exact algorithmic collapse** — 512 copies of one image; generated k-NN radii collapse to zero. The nearest thing to a known-zero null. |
| `gaussian_mm` | realistic low-recall stress control (measured precision 0.867, recall 0.000) |
| `blend_near_gate` | separately frozen near-threshold sensitivity control, λ chosen once from a coarse pilot to land in 0.02–0.10 and then fixed |

Sampling notes that must travel with the numbers: `gaussian_mm` and
`identical_images` are drawn fresh per replicate, so the 200 sets are genuinely
independent. `blend_near_gate` draws its real component fresh from the 40 000-image
train split per replicate; overlap is ≈1.3% and the estimate is labelled
**near-independent**. Because the real evaluation reference is held fixed, all
reported noise is **conditional on that reference** — stated explicitly. A
robustness repeat on a second disjoint real reference is run if §2.2's budget
allows.

**Decision rule — exceedance probability with an upper confidence bound.**
v2's point-`q99` rule is withdrawn (§15.3): at 200 replicates the empirical 99th
percentile is determined by two or three observations, so a point-quantile rule
can proceed even when its own interval does not support the decision.

`RECALL_GATE` is instead **held fixed at 0.05** and the null probability of
exceeding that exact value is bounded:

1. for each null state, count `E = #{r : recall_r > 0.05}` over the 200
   independent runs;
2. compute the **one-sided exact (Clopper–Pearson) 95% upper bound**
   `p_null_upper` on the exceedance probability;
3. take `p_null_upper = max` over `identical_images` and `gaussian_mm`;
4. **proceed only if `p_null_upper < 0.025`.** Otherwise increase the evaluation
   sample count and repeat calibration — never move the scientific threshold.

Orientation: `E = 0` in 200 trials gives `p_null_upper ≈ 0.0149`
(`1 − 0.05^{1/200}`), comfortably inside tolerance. At the tolerance itself,
`p = 0.025`, the 2-of-3 false-pass bound `3p²(1−p) + p³` is **0.00184**.

The 99th percentile and a bootstrap interval are still *reported* for
orientation, but no decision is taken from the point estimate. If a quantile
formulation is ever preferred, the decision must use the **upper confidence
endpoint** via an order-statistic/binomial tolerance interval, not an
unqualified bootstrap CI at so extreme a quantile.

This measurement also retrospectively decides how to read Phase 30's flickers
(0.009, 0.044, 0.001) and is reported in both places.

### 2.2 Cost and resources

Projected before commitment, **including source construction**, which v1 omitted:

| item | estimate |
|---|---|
| `trained_bad` regeneration, **×3 units** (5 000-step generator train each) | ~9 min |
| `ae_reconstruction` autoencoder, **×3 units** (8 000 steps each) | ~9 min |
| Phase-29 collapse reference tensors — **regenerated, not on disk** (§15.7) | ~15 min |
| §2.1 calibration: 3 states × 200 × 512 samples through Inception | ~25 min |
| `blend_near_gate` pilot: 5 λ × 20 replicates | ~5 min |
| particle step, 512 particles × 64 positives | ~20–50 ms (to measure) |
| 200-step rollout | ~10 s |
| Inception scoring per checkpoint (512 samples, features cached) | ~3 s |
| scorings: 6 starts × 2 regimes × 3 units × 6 checkpoints | 216 |
| **projected total** | **~2.5–3 h** |

Inception features and real-reference statistics are cached so evaluation cost
cannot dictate the number of scientific replicates. If the measured projection
exceeds 3 h, **reduce checkpoints, never replicates**.

---

## 3. Initial distributions

Six labelled starts, with novelty stated per arm so the projection is not padded
with replications.

| arm | role | already known |
|---|---|---|
| `real_data` | **positive control / validity precondition** | Phase 26: recall 0.724 → 0.717 over 40 steps |
| `random_generator` | **PRIMARY — the deployed initialization** | **untested at any horizon** |
| `trained_bad` | long-horizon control | Phase 26: exactly 0.000 through K = 40 |
| `ae_reconstruction` | near-data control | Phase 28 Stage 0: *degrades* 0.301 → 0.270 through K = 40 |
| `basin_interpolation` | basin-edge control | largely bracketed by Phase 27 (blend edge λ ≈ 0.6) |
| `ambient_noise` | artificial stress test | — |

If `real_data` does not hold recall > 0.5 at every checkpoint, **the run is void
and no other arm may be read.**

**Frozen constructors** (v1 left these implicit):

- `random_generator` — `OneStepGenerator(latent_dim=32, channels=3,
  image_size=32, width=64)`, declared weight-init seed, latents `randn` drawn on
  CPU per `models.sample_latent`, no output postprocessing, 512 samples. Measured
  in Phase 28 at alpha 3.012, second moment 0.378. **The three confirmatory
  replicates vary the generator initialization seed**, not merely bank or rollout
  seeds — the source distribution is what basin membership is being tested from.
- `trained_bad` — regenerated by `diagnose_phase25.train_cloud` at 5 000 steps
  with the recorded seed, then **hashed and stored as a tensor artifact** so
  later reads are byte-identical.
- `ae_reconstruction` — `autoencoder.train_autoencoder`, `latent_channels=32`
  (d = 512), 8 000 steps, recorded seed; reconstruction inputs are 512 real
  training images from declared indices; starting tensor hashed.
- `basin_interpolation` — the **`blend`** path of `diagnose_phase27`
  (not `mixture`), **λ = 0.6**, `x = (1−λ)·near_data + λ·trained_bad` with
  index-aligned pairing; λ = 0 is the near-data endpoint. **`near_data` is the
  index-paired `real_data` start** (§15.7), matching Phase 27's construction,
  which blended real data with the trained cloud — *not* the autoencoder
  reconstructions.
- `ambient_noise` — `N(0, 0.5²)` per pixel, clipped to [−1, 1], all channels
  independent, declared seed. Excluded from every deployment claim.
- `real_data` — 512 **unique** real training images whose indices are disjoint
  from the target bank, the kernel-calibration subset, and the evaluation split.

**Calibration-state constructors** (§15.7), frozen with the same rigour since
they feed the null decision:

- `gaussian_mm` — `diagnose_phase15.gaussian_moment_match`, i.e. **per-coordinate**
  mean and standard deviation (not a full covariance) estimated from the **2 048
  sample evaluation reference itself**, drawn with a declared seed per replicate,
  **no clipping** — matching every prior use of this control in Phases 15–30 so
  the 0.867/0.000 anchors remain comparable.
- `identical_images` — 512 copies of a single real training image at a declared
  index, disjoint from bank, calibration, `real_data` and eval split.
- `blend_near_gate` — **sensitivity control only; it never enters the null
  decision.** Endpoints are `real_data` (λ = 0) and `ambient_noise` (λ = 1).
  λ is chosen once from the fixed pilot grid {0.5, 0.6, 0.7, 0.8, 0.9} using **20
  pilot replicates on a pilot-only real reference disjoint from the F1 evaluation
  reference**, selecting the smallest λ whose pilot median recall falls in
  [0.02, 0.10]; ties break to the smaller λ; if no grid point qualifies the
  control is reported as unavailable rather than re-tuned. The chosen λ is frozen
  and recorded before the 200-replicate run.

---

## 4. Teacher regimes

Both are run and reported separately.

### 4.1 `replay` — seeded replay-minibatch teacher

Renamed from v1's "deterministic", which overstated it. This is **not** the
full-bank population field and, because the positive subset changes with step,
it is **not an autonomous time-homogeneous particle map**. What it provides is a
reproducible finite-horizon endpoint map, which is what F3A requires.

- A bank of **4 096 unique training images**, drawn **without replacement** from
  declared indices. **`ImageTarget.sample` samples with replacement**, so the
  bank must be built by explicit index selection; using `sample` would silently
  duplicate entries and corrupt the §7 distinct-bank audit.
- Each step's `positives` subset is chosen from the bank by a fixed seeded
  schedule. The bank indices and the **full subset schedule are hashed and
  stored**.
- Deterministic backend settings enabled; reproducibility is verified by running
  the same latent and bank seed twice and requiring
  `‖x_K^{(1)} − x_K^{(2)}‖₂ / max(‖x_K^{(1)}‖₂, ε)` ≤ **1e-4**, with ε = 1e-12
  guarding the zero-denominator case. Reported, not assumed.
- The target law is the **bank's** empirical law, not the train split's. This
  caps attainable coverage and must be stated with any number from this regime.

### 4.2 `stochastic`

Fresh minibatches from the full train split each step — the current recipe and
the regime every prior phase measured.

### 4.3 Frozen update

The runner calls exactly this. Two corrections from §15.6: `denominator_floor`
was inherited rather than explicit, and `corrected_teacher` writes its gain
**only** when handed a mutable `report` dict — so v2 promised a diagnostic it
would not have recorded.

```python
drift, stats = KG.field(
    state, positives, state, branch, kernel,
    direction_mode="paper",
    normalization="rms",
    denominator_floor=1e-30,
    self_mask=False,
    diagnostics=True,
)
r11_report = {}
state = corrected_teacher(
    state + 0.5 * drift, positives, mode="scalar", report=r11_report,
)
```

All five names below are verified present in the current source
(`kernel_gradient.field` lines 319–342; `objectives.corrected_teacher`).

Also frozen: raw geometry, `smooth_laplace` base kernel, the calibration
procedure with its seed and data subset, `denominator_floor`,
`target_ess_fraction = 0.05` (legacy calibration; realized ESS ≈ 0.55–0.59 in
Phase 30), R11 scalar mode, and the kernel object itself.

**Recorded per arm and checkpoint**, by exact key:
`stats["ess_fraction"]`, `stats["collapsed_row_fraction"]`,
`stats["denominator_floor_fraction"]`, `stats["drift_rms_raw"]`, and
`r11_report["correction_ratio_median"]`. Without these, a failure caused by a
dead kernel or a pathological correction cannot be separated from genuine basin
failure.

**Diagnostic timing, previously undefined (§15.6):** at each checkpoint the
recorded value is the **mean over the preceding interval** for
`ess_fraction`, `collapsed_row_fraction`, `denominator_floor_fraction` and
`drift_rms_raw`, and the **median over the preceding interval** for
`correction_ratio_median`. The K = 0 checkpoint records a single extra field
evaluation taken before the first update, so no interval is empty. Interval
statistics are stored at float32 resolution alongside the last-step value, so
either reading is available after the fact.

---

## 5. Horizons

`K ∈ {0, 10, 20, 40, 100, 200}`. A `K = 1000` checkpoint is added **only if**
§2.2's measured projection stays inside budget; the decision is recorded before
any arm runs, on cost alone. If enabled, K = 1000 becomes the declared terminal
checkpoint everywhere and gets its own sample grid.

**No arm passes because its last two checkpoints trend upward.**

### 5.1 Historical regression cell, separated from the confirmatory design

Phase 26/27/28 values are mostly single-seed, so arbitrary new seeds should not
be expected to reproduce them. **First** run one small regression cell using the
exact historical configuration and seeds. **Then** run the independently seeded
confirmatory design. The regression cell is an implementation check and is never
read as an F1 result.

**Numeric tolerances**, replacing v2's "declared tolerance" (§15.7). Checked at
K = 40 against the recorded values:

| check | recorded | tolerance |
|---|---|---|
| `real_data` recall (Phase 26) | 0.717 | ±0.030 absolute |
| `real_data` KID (Phase 26) | +0.00041 | ±0.0015 absolute |
| `trained_bad` recall (Phase 26) | 0.000 | exact 0.000 |
| `ae_reconstruction` recall (Phase 28 S0) | 0.270 | ±0.030 absolute |
| `ae_reconstruction` KID (Phase 28 S0) | +0.08599 | ±0.0050 absolute |
| one-step field equivalence vs Phase 26 | — | ≤1e-6 relative L2 |

The recall tolerances are wider than the KID ones deliberately: recall's
low-value behaviour is precisely what §2.1 is measuring and is not yet
characterized. If §2.1 returns a null exceedance materially different from zero,
these tolerances are revisited **before** the regression cell is read.

---

## 6. Metrics

Per arm × regime × replicate × checkpoint: **recall (primary, with
uncertainty)**, precision, KID with uncertainty, FID at the exact F1 sample
count, alpha, spectral tail, second-moment ratio, effective rank, duplicate
rate, nearest-neighbour diversity, cumulative and per-step particle
displacement, plus the §4.3 kernel diagnostics.

A **matched real-vs-real FID/KID reference at the identical sample counts** is
computed and reported alongside; FID at 512 generated samples carries
substantial finite-sample bias and is never compared with published values or
differently sized evaluations.

**How uncertainty is computed (§15.7).** For **recall**, the binomial fraction of
covered evaluation samples captures only reference-side sampling and omits
generated-manifold uncertainty, so it is *not* used alone. Instead: resample the
512 particles with replacement 200 times, recompute recall each time — which
re-estimates the generated k-NN radii and therefore the manifold — and report the
2.5/97.5 percentiles. The reference set is held fixed, so this interval is
**conditional on the evaluation reference**, matching §2.1's null. The binomial
reference-side interval is reported separately as a lower bound on total
uncertainty, explicitly labelled as such. For **KID**, the standard error across
the same 200 particle resamples is reported. Both intervals are diagnostics; the
gate is a point comparison against `RECALL_GATE` per §8, with §2.1's calibration
being what makes that point comparison meaningful.

**KID and FID are diagnostics only and are not pass conditions.** Phase 30
§20.3: `w128` had the worst KID and the best recall; `w192` seed0 had the best
KID in the program with recall 0.000. A KID regression may be discussed but
cannot fail an otherwise valid coverage pass.

Sample grids for every arm at K ∈ {0, 40, terminal}.

---

## 7. Memorization and collapse vetoes — calibrated, not intuited

v1's `0.25` normalized-distance and `64 distinct` thresholds were set from
intuition. That is the failure mode this program has hit five times, so they are
**calibrated before any F1 outcome is exposed**, from these reference
distributions:

1. held-out real samples versus the bank (healthy empirical reference);
2. exact bank copies (known memorization);
3. lightly perturbed bank copies;
4. the Phase-29 `nearest_fresh` / fixed-pair collapse states; and
5. the `random_generator` start.

**Mechanical selection rule (§15.4), fixed before any reference outcome is
seen.** "Frozen from those distributions" left a degree of freedom; this removes
it:

1. orient every statistic so that **larger means healthier**;
2. choose the threshold that **accepts ≥ 95% of the held-out-real reference and
   rejects ≥ 95% of the declared memorization/collapse references**; among all
   qualifying thresholds take the one maximizing the margin to the healthy 5th
   percentile;
3. verify the chosen threshold on the perturbed-copy and `random_generator`
   sensitivity controls and report where they fall;
4. **if no threshold satisfies (2), that statistic is not a valid categorical
   veto.** Drop it from the gate — recording that it was dropped — or return
   NO-GO. An outcome-specific cutoff is never chosen.

Thresholds are recorded in the calibration artifact. **Full nearest-distance and
claimed-bank-index distributions are reported, not only medians.**

Statistics: normalized nearest-bank distance (against the median real-to-real
nearest-neighbour distance on held-out real data), and the count of **distinct**
bank images claimed as some particle's nearest neighbour — the diagnostic that
exposed Phase 29's `nearest_fresh` at `distinct = 1`.

**Regime coverage.** The finite-bank test applies to `replay`. For `stochastic`
there is no fixed bank, so the veto there is defined on the train split as a
whole, using the same calibrated normalized-distance statistic plus duplicate
rate and nearest-neighbour diversity within the particle set.

**Precision is removed from the categorical gate.** `precision > 0.10` is not a
demonstrated non-collapse criterion: a structureless Gaussian scores 0.867, and
a mode-collapsed output can sit entirely inside the real manifold. Precision is
retained as a paired quality statistic. Collapse is judged instead by effective
rank, duplicate rate and nearest-neighbour diversity, with thresholds calibrated
from states 1–5 above.

---

## 8. Corrected minimal F1 gate

On the `random_generator` start, reporting `replay` and `stochastic` separately.
**The conjunction is per-replicate (§15.4)**, so recall in two replicates cannot
be combined with a veto that passes only in a third:

```text
replicate_pass(r) :=
        terminal_recall(r) > RECALL_GATE
    and regime_memorization_veto(r)
    and collapse_veto(r)
    and real_data_control_valid(r)

F1_pass := at least 2 of the 3 independent replicate units satisfy replicate_pass
```

**Every paired `real_data` control must pass** — the recommended semantics of
§15.4, adopted. A replicate whose control fails does not merely lose that
replicate: because the control shares the replicate's teacher seed, kernel
calibration and evaluation reference, its failure indicates an implementation or
configuration defect in that unit. Such a unit is **excluded and re-run after
diagnosis**, and it is never counted as a failed replicate for the 2-of-3 tally.
If two or more units fail their control, **the run is void**.

v1's condition requiring terminal KID to improve is **deleted**: it contradicted
§6 and would have let an Inception-space MMD estimator veto a valid coverage
result.

A sub-threshold improving trajectory is recorded and may motivate a **separately
frozen** continuation run at longer K. It is not a pass.

---

## 9. Declared outcomes

- **F1 pass** → proceed to **F3A**; freeze F3A's teacher-retention margin from
  F1's measured recall before any F3A training (§19.7).
- **F1 fail** → reject the current pure autonomous drift as the teacher *from
  this initialization*; proceed to **F3B**. This is a basin-reachability result
  for this teacher and start, and refutes neither the population converse nor
  encoder-independent path models generally.
- **`replay` passes, `stochastic` fails (or vice versa)** → the target-sampling
  regime is the operative variable. F3A requires `replay`, so a replay-only pass
  still selects F3A, with the discrepancy recorded as a finding.
- **`real_data` control fails** → run void, implementation defect, no arm read.

---

## 10. Seed registry

**Revised per §15.5, which correctly found that v2 licensed only a conditional
claim.** Varying only the source seed left all three replicates sharing one
target realization, so a single lucky or pathological schedule would have
affected every one of them — and replay-versus-stochastic differences could not
have been called sampling-regime robustness.

Route 1 (the audit's preferred stronger design) is adopted: **a replicate unit is
an independent (source seed, teacher seed) pair.**

| role | varies across replicate units? |
|---|---|
| source / generator initialization | **yes** |
| source latent stream | yes, derived from the source seed |
| `replay` bank selection | **yes** — independent bank per unit |
| `replay` subset schedule | **yes** — independent schedule per unit |
| `stochastic` target stream | **yes** — independent stream per unit |
| kernel calibration | no — fixed across all units |
| evaluation reference | no — fixed across all units |

Within a unit, `replay` and `stochastic` are paired on the **same initial cloud,
same kernel calibration and same evaluation reference**, so the regime contrast
stays a within-unit comparison. Rollout cost is unchanged; only the bank indices
and schedules differ.

**Control-arm replication (§15.5).** A single hashed `trained_bad` or
`ae_reconstruction` tensor combined with one schedule yields three *identical*
results, not three controls. Therefore: `trained_bad` and `ae_reconstruction`
source tensors are regenerated **once per replicate unit** with that unit's
source seed and hashed individually; `basin_interpolation` inherits the unit's
`near_data`/`trained_bad` pair; `ambient_noise` and `real_data` are redrawn per
unit. Every arm therefore has three genuinely independent instances.

Three replicate units license the categorical 2-of-3 gate **only** — not
variance components, not fine arm rankings. The conclusion is no longer
conditional on one teacher realization.

---

## 11. Required checks before the confirmatory run

All must pass, and none is read as an F1 result. **Ordering is mandatory**
(§15.8): specification is complete before implementation, implementation before
calibration, calibration before any confirmatory arm.

1. §§2.1, 3, 4, 5.1, 6, 7, 8, 10 frozen — **done in this revision**;
2. implement the calibration runner and the F1 runner;
3. write the immutable config/seed registry with **actual numeric seed values and
   artifact filenames**, not role names (§15.7);
4. run §2.1 null calibration; **require GO** (`p_null_upper < 0.025`);
5. run exact one-step equivalence against the Phase-26 update (≤1e-6 relative L2)
   and the §5.1 historical K = 40 regression cell within the tabulated
   tolerances;
6. run `replay` reproducibility twice (≤1e-4 relative L2) and the data-index
   disjointness and unique-bank assertions (bank, calibration, `real_data`,
   `identical_images`, eval split);
7. run the all-arm / two-regime smoke test without interpreting it as F1;
8. update the measured time/memory projection and freeze the
   K = 200 vs K = 1000 terminal decision on cost alone; **then and only then**
9. launch the three confirmatory replicate units.

**Nothing in steps 2–8 is a scientific result.** Step 4 can return NO-GO and
stop the stage.

Width 64 is sufficient for the primary question: it is the deployed reference
and F0 found no robust capacity effect. Width 192 may be added later as a
separately labelled exploratory robustness arm and must not expand or delay the
primary gate.

---

## 12. Scope and known limits

- 3 replicates under §19.3's 2-of-3 categorical rule; §2.1 is the check on
  whether the gate is categorical at all.
- `replay`'s attainable coverage is capped by a 4 096-image bank; that cap is
  stated with every number from that regime.
- 512 particles, 64 positives, both fixed. Phase 30 showed positives 64 → 256 is
  null for the *generator*; that is not evidence about free particles.
- One dataset, one resolution, one geometry, width 64.
- F1 tests the **particle basin only**. A pass does not imply a student can
  retain the coverage — that is C3/F3A, and Phases 26 versus 28 already showed
  free particles and generators behave differently.
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.

---

## 13. Independent audit of v1 (2026-07-30)

*Restored as the record. The v1 protocol it audits has been replaced by §§1–12
above; §14 maps each finding to its revision.*

### 13.1 Verdict

The scientific question and broad experimental structure are sound, but this
protocol is **not yet safe to execute as the confirmatory F1 run**. The audit
found two blocking gate defects, several reproducibility specifications that must
be frozen, and no F1-specific executable runner yet. The historical Phase-26
implementation confirms that the intended teacher is available, but it does not
implement the six F1 starts, deterministic replay regime, long horizons,
calibration pre-flight, or F1 decision ledger.

The good parts should be retained: F1 isolates particle dynamics from neural
amortization; `random_generator` is the correct deployment-relevant primary
start; `real_data` is an essential positive control; deterministic replay and
fresh stochastic targets answer different questions; and the scope is properly
limited to this teacher, initialization, dataset, and geometry.

### 13.2 Blocking correction 1 — KID cannot be both secondary and a gate

Section 6 says that no gate is keyed on KID, in agreement with §20.3 and the
Phase-30 observation that KID ranked zero-coverage and higher-recall arms in the
wrong order. Section 8 nevertheless requires terminal KID to improve. These
statements are contradictory.

**Required repair:** remove current condition 2 from the F1 pass rule. Report
KID, with uncertainty, as a secondary consistency/quality measure only. A KID
regression may be discussed, but it must not turn an otherwise valid coverage
pass into an F1 failure. KID is an Inception-space MMD estimator, not a direct
coverage statistic; see Binkowski et al., *Demystifying MMD GANs*:
<https://openreview.net/pdf?id=r1lUOzWCW>.

### 13.3 Blocking correction 2 — the recall calibration rule is invalid

`max(0.05, 5*SD)` is not five standard deviations above the null unless the null
mean is exactly zero. Even `mean + 5*SD` would require an approximately Gaussian
sampling distribution, which is not established here. This recall estimator is
bounded and discrete and its k-nearest-neighbour manifold changes with the
generated sample. The moment-matched Gaussian is an empirically low-recall
reference, not a mathematically known-zero distribution. The metric itself is the
sample-dependent manifold estimator of Kynkaanniemi et al.:
<https://proceedings.neurips.cc/paper_files/paper/2019/file/0234c510bc6d908b28c70ff313743079-Paper.pdf>.

Twenty subsets are also too few to estimate a stable upper tail. If the subsets
must be disjoint, `20 x 512 > 8192`, so the stated construction is impossible; if
they are independently redrawn from the same finite pool, overlap is allowed and
the result estimates conditional subset noise while omitting finite-pool
uncertainty. That distinction must be stated.

**Required replacement pre-flight:**

1. Use at least 100–200 independently generated 512-sample sets, rather than
   repeatedly subsetting one 8192-sample pool.
2. Include an exact-collapse algorithmic control (for example, repeated identical
   images, whose generated k-NN radii collapse to zero), the moment-matched
   Gaussian as a realistic low-recall stress control, and a separately frozen
   near-threshold sensitivity control.
3. Keep the real evaluation reference fixed if the desired quantity is the
   conditional noise of the actual F1 readout, and label it as conditional. Also
   repeat on a second disjoint real reference as a robustness check if the cost
   permits.
4. Estimate the empirical upper null quantile directly and report uncertainty on
   that quantile. Keep the scientific gate at 0.05 only if the upper null bound
   lies comfortably below 0.05.
5. If n = 512 cannot resolve that gate, increase the evaluation sample count and
   repeat calibration. Do not redefine the scientific threshold through an
   unvalidated `5*SD` formula.

After valid calibration, the 2-of-3 categorical rule is reasonable. If the
per-replicate null exceedance probability is 0.05 and replicates are independent,
its false-pass probability is `3*0.05^2*0.95 + 0.05^3 = 0.00725`.

### 13.4 Freeze the exact teacher semantics

The F1 runner must call the same update as Phase 26, with all behavior-changing
arguments explicit rather than inherited from defaults: `KG.field` with
`direction_mode="paper"`, `normalization="rms"`, `self_mask=False`,
`diagnostics=True`, followed by `corrected_teacher(state + 0.5 * drift,
positives, mode="scalar")`.

The raw geometry, `smooth_laplace` kernel family, calibration procedure,
calibration seed/data, denominator floor, target ESS, R11 mode, and kernel object
must also be frozen. Record realized ESS, dead rows, raw drift RMS, and R11 gain
by arm/checkpoint. Without those diagnostics, failure caused by a dead kernel or
pathological correction cannot be separated from genuine basin failure.

The so-called deterministic regime is valid for constructing a reproducible
finite-horizon endpoint map, provided the bank and per-step schedule are fixed.
It is more accurately described as a **seeded replay-minibatch teacher**: it is
not the full-bank population field and, because the subset changes with step, it
is not an autonomous time-homogeneous particle map. Hash and store the bank
indices and full subset schedule, enable the available deterministic backend
settings, and define the relative-L2 reproducibility formula and zero-denominator
handling.

### 13.5 Freeze every initial arm and data split

Before execution, record exact constructors and seeds for all starts:

- `random_generator`: generator class, width 64, latent dimension/distribution,
  weight-initialization seed, latent seed, output postprocessing, and 512-sample
  cloud. The three primary replicates must vary generator initialization, not
  merely target-bank or rollout seeds.
- `trained_bad`: exact Phase-26 reconstruction procedure, training steps,
  generator/cloud seed, and whether the state is regenerated or loaded from a
  hashed tensor artifact.
- `ae_reconstruction`: exact autoencoder training configuration/checkpoint,
  reconstruction inputs, seed, and hashed starting tensor.
- `basin_interpolation`: exact path (`blend`, not `mixture`), exact lambda,
  pairing/permutation rule, and which endpoint corresponds to lambda 0 and 1. If
  lambda 0.6 is intended, say so explicitly.
- `ambient_noise`: distribution, mean, variance, clipping, normalization, channel
  handling, and seed.
- `real_data`: 512 unique real examples drawn from indices disjoint from the
  deterministic target bank, kernel-calibration subset, and evaluation split.

The deterministic 4096-image bank must be constructed from **unique training
indices without replacement**. The existing `ImageTarget.sample` interface
samples with replacement, so using it directly would silently create duplicate
bank entries and corrupt the "distinct bank images" audit. The evaluation
reference remains on the repository's disjoint eval split.

The K = 40 historical checks should be separated from the three confirmatory
replicates. Phase-26/27/28 values were mostly single-seed results; arbitrary new
seeds should not be required to reproduce their exact numbers. First run one small
regression cell with the exact historical configuration and tolerance, then run
the independently seeded confirmatory design.

### 13.6 Seed and pairing design

Do not collapse generator seed, bank seed, target schedule, evaluation sample,
and regime into one opaque "bank/seed replicate." Freeze a seed table with at
least: source/generator initialization seed; source latent seed; deterministic
bank seed; deterministic replay-schedule seed; stochastic target-stream seed;
kernel-calibration seed; and evaluation-reference seed.

Pair deterministic and stochastic regimes on the same initial particle cloud,
kernel calibration, and evaluation reference. Pair comparable control arms where
possible. Three replicates license only the categorical 2-of-3 gate, not
variance-component estimates or fine arm rankings.

### 13.7 Memorization and non-collapse vetoes need calibration

The current `0.25` normalized nearest-bank distance and `64 distinct` thresholds
are set from intuition. Calibrate them before any F1 outcome is exposed using:
(1) held-out real samples versus the bank; (2) exact bank copies; (3) lightly
perturbed/noisy bank copies; (4) the Phase-29 nearest/fixed-pair collapse states;
and (5) the random-generator start.

Freeze thresholds from those reference distributions and report the full
nearest-distance and claimed-bank-index distributions, not only medians. State
separately what audit applies to the stochastic regime, for which the finite-bank
test is currently undefined.

`precision > 0.10` is also not a demonstrated non-collapse criterion: a
structureless Gaussian can have high precision, and a mode-collapsed output can
remain inside the real manifold. Either calibrate a gross-invalidity veto from
known bad states or remove precision from the categorical F1 gate and retain it
as a paired quality statistic. Effective rank/tail, duplicate rate, and
nearest-neighbour diversity are more direct collapse diagnostics.

### 13.8 Evaluation and resource repairs

- Add a matched real-vs-real FID/KID reference at the exact F1 sample counts. FID
  at 512 generated samples has substantial finite-sample bias and must not be
  compared with published FID values or differently sized evaluations.
- Report uncertainty for KID and the primary recall statistic.
- If K = 1000 is enabled, include its sample grid and make it the declared
  terminal checkpoint everywhere; freeze this choice using cost only, before
  outcomes are read.
- Include source-construction costs in the pre-flight. Regenerating the
  `trained_bad` cloud and training the Phase-28 autoencoder are omitted from the
  current `< 1 h` projection.
- Prefer caching Inception features and real-reference statistics so evaluation
  cost does not dominate or alter the number of scientific replicates.

### 13.9 Required implementation and checks before the full run

No F1-specific runner was present at the time of this audit. Before launching the
full experiment, add and successfully execute only the following
non-confirmatory checks: (1) metric-calibration pre-flight producing a frozen
JSON artifact and hash; (2) exact one-step equivalence against the Phase-26
update; (3) exact historical K = 40 regression cell; (4) deterministic replay run
repeated twice, satisfying the declared relative-L2 tolerance; (5) data-index
disjointness and unique-bank assertions; (6) paired-seed registry and immutable
protocol/config hash; (7) a very small smoke run that exercises all arms, both
regimes, every metric, checkpoint serialization, and the decision ledger without
reading it as an F1 result; and (8) measured time/memory projection including
initialization and scoring.

Only after these checks pass should the confirmatory F1 arms be run. Width 64 is
sufficient for the primary F1 question: it is the deployed reference and F0 found
no robust capacity effect. Width 192 may be added later as a separately labelled
exploratory robustness arm, but it should not expand or delay the primary gate.

### 13.10 Corrected minimal F1 gate

Subject to the calibrated thresholds above, the clean gate should be:

1. on `random_generator`, terminal recall exceeds the frozen `RECALL_GATE` in at
   least 2 of 3 independent source-seed replicates;
2. the relevant regime-specific memorization/diversity veto passes;
3. the calibrated gross-invalidity/quality veto passes, if one is retained; and
4. the paired `real_data` control remains valid throughout.

KID, FID, precision, spectral statistics, ESS, and displacement are required
diagnostics, but KID/FID must not be hidden additional pass conditions. Report
deterministic replay and stochastic results separately, exactly as the current
declared-outcomes section intends.

---

## 14. Revisions applied (Claude Opus 5, 2026-07-30)

Every audit item and where it landed. Both blocking findings are accepted without
reservation; one identified a flat arithmetic error in v1.

| audit item | revision |
|---|---|
| **13.2** KID both secondary and a gate | **Blocking, accepted.** v1 §8 condition 2 (terminal KID must improve) is **deleted**. §6 states KID and FID are diagnostics only and cannot fail a valid coverage pass. §8 now lists three conditions, none keyed to a quality metric. |
| **13.3** `5*SD` invalid; `20x512 > 8192` impossible | **Blocking, accepted — v1 contained an arithmetic error.** §2.1 replaced entirely: 200 independently generated 512-sample sets; three reference states including an **exact-collapse `identical_images`** control; decision by **empirical 99th-percentile null quantile** with bootstrap CI, no sigma formula; `q99 < 0.025` keeps 0.05, `0.025-0.05` raises the sample count rather than the threshold, `>= 0.05` is NO-GO. Independence status labelled per state, conditional-on-fixed-reference stated, second-reference repeat if budget allows. False-pass arithmetic `0.00725` adopted. |
| **13.4** freeze teacher semantics | §4.3 pins the exact `KG.field` / `corrected_teacher` call with all arguments explicit, plus geometry, kernel family, calibration seed/data, denominator floor, target ESS and R11 mode. Realized ESS, dead rows, raw drift RMS and R11 gain recorded per arm and checkpoint. |
| **13.4** "deterministic" overstated | Regime renamed **`replay` - seeded replay-minibatch teacher** (§4.1), stating explicitly that it is neither the full-bank population field nor time-homogeneous. Bank indices and full subset schedule hashed; relative-L2 formula and eps = 1e-12 zero-denominator handling declared. |
| **13.5** freeze all constructors | §3 gives frozen constructors, seeds and hashed artifacts for all six starts, including `blend` at **lambda = 0.6** with lambda = 0 identified as the near-data endpoint. |
| **13.5** `ImageTarget.sample` replaces | §4.1 requires the bank be built from **unique indices without replacement** and names the hazard - v1 would have silently duplicated entries and corrupted the distinct-bank audit. |
| **13.5** separate historical checks | New §5.1: a historical regression cell at the exact recorded configuration runs **first** and is never read as an F1 result; the confirmatory design is independently seeded afterwards. |
| **13.6** seed and pairing design | New §10 seed registry: seven distinct roles, replicates varying **only** the source seed, `replay`/`stochastic` paired on cloud, calibration and evaluation reference. States that 3 replicates license the categorical gate only. |
| **13.7** vetoes intuited | §7 rewritten: thresholds calibrated from five declared reference distributions before any outcome is exposed; full distributions reported, not medians. Stochastic-regime veto now defined on the train split plus duplicate rate and NN diversity. |
| **13.7** `precision > 0.10` not a criterion | **Precision removed from the gate**, retained as a paired quality statistic. Collapse judged by effective rank, duplicate rate and NN diversity with calibrated thresholds. |
| **13.8** evaluation and resource repairs | §6 adds a matched real-vs-real FID/KID reference at identical sample counts, and uncertainty on recall and KID. §5 makes K = 1000 the declared terminal checkpoint if enabled, chosen on cost before outcomes. §2.2 now **includes source-construction cost** and revises the projection from `< 1 h` to **~1.5-2 h**; Inception features cached. |
| **13.9** required checks | New §11 with all eight checks, none read as an F1 result. Width 64 confirmed sufficient for the primary gate; width 192 deferred to a separately labelled exploratory arm. |
| **13.10** corrected minimal gate | Adopted as §8, with the `real_data` control stated as a validity **precondition** rather than a scoring criterion. |

**Not yet done, and required before execution:** the F1 runner does not exist.
§11 lists the eight checks that must pass first, and the §2.1 calibration
pre-flight must run and return GO before any confirmatory arm.

**One item flagged back to the auditor.** §13.3 point 3 asks that the fixed real
reference be labelled as giving *conditional* noise, which §2.1 now does. But the
F1 gate is applied against that same fixed reference, so the conditional quantile
is arguably the correct null for this decision, and the second-reference repeat is
a robustness check rather than a correction. Raised in case the stronger reading
was intended - if so, the unconditional quantile becomes the gating quantity and
the required sample count rises.

**Process note.** In applying these revisions the v1 body was rewritten in place,
which deleted the audit text; it has been restored here from the review record
rather than from version control, since this file is untracked. Wording is
faithful but not guaranteed byte-identical to the original §12, and code blocks
in §13.4 were reflowed to prose. The original numbering (§12) is now §13.

---

## 15. Independent readiness audit of DESIGN v2 (2026-07-30)

### 15.1 Verdict

The v2 revision correctly resolves the two principal v1 defects and is
substantially closer to a valid confirmatory experiment. The scientific F1
question is ready and the high-level protocol is mostly ready. **The
confirmatory F1 run is nevertheless still NO-GO.** One statistical decision
rule and one categorical-veto rule remain incompletely defined, the replication
scheme currently licenses only a conditional result, two details in the frozen
teacher snippet contradict the promised diagnostics, and no F1 runner or
pre-flight artifact exists.

This is not a rejection of F1. The next authorized stage is to finish the
specification, implement the runner, and execute only the non-confirmatory
calibration/regression/reproducibility/smoke checks. The confirmatory arms become
run-ready only after those checks pass.

### 15.2 Revisions assessed as correct

The following changes should be retained:

- KID and precision are removed from the categorical pass gate and retained as
  diagnostics.
- The invalid `5·SD` rule and impossible `20 × 512` disjoint-subset construction
  are withdrawn.
- Calibration uses freshly generated sets and labels its uncertainty as
  conditional on the fixed evaluation reference.
- All six initial-condition families are much more precisely specified.
- The replay bank is unique, selected without replacement, and disjoint from
  the positive-control/calibration/evaluation samples.
- The former “deterministic” arm is accurately described as a seeded
  replay-minibatch teacher, not a full-bank or time-homogeneous population map.
- The intended Phase-26 update, kernel health diagnostics, R11, horizons,
  historical regression cell, memorization audit, cost pre-flight, and
  reproducibility check are all visible.
- Width 64 is the correct primary deployment configuration. Width 192 can
  remain exploratory.

### 15.3 Remaining blocker 1 — decide on an uncertainty bound, not point `q99`

Section 2.1 reports a bootstrap confidence interval for the null 99th
percentile but makes the GO/NO-GO decision using the **point estimate** `q99`.
With 200 replicates, the empirical 99th percentile is controlled by only about
the top two or three observations. A point-quantile rule can therefore proceed
even when its own uncertainty interval does not support the decision.

The cleanest repair is to keep `RECALL_GATE = 0.05` fixed and estimate the null
probability of exceeding that exact gate:

1. For each null state, count
   `E = #{r : recall_r > 0.05}` among the 200 independent runs.
2. Compute a one-sided exact binomial upper confidence bound `p_null_upper` for
   the exceedance probability.
3. Take the maximum upper bound over `identical_images` and `gaussian_mm`.
4. Proceed only if `p_null_upper` lies below a predeclared tolerance; otherwise
   increase the evaluation sample count and repeat. A tolerance of 0.025 is a
   defensible conservative choice because it keeps the corresponding 2-of-3
   false-pass bound small.

For orientation, zero exceedances in 200 trials has a one-sided 95% upper bound
of approximately 0.015. Under an independent per-replicate null probability
`p`, the 2-of-3 false-pass probability is `3p²(1−p) + p³`.

If the quantile formulation is retained instead, the decision must use the
**upper confidence endpoint** for the null quantile, not the point `q99`, and an
order-statistic/binomial tolerance interval is preferable to an unqualified
bootstrap interval at such an extreme quantile.

### 15.4 Remaining blocker 2 — freeze the veto construction and conjunction

Section 7 names the healthy and known-collapse calibration distributions but
does not state how they determine the effective-rank, duplicate-rate,
nearest-distance, distinct-bank, and nearest-neighbour-diversity thresholds.
“Thresholds are frozen from those distributions” still leaves a degree of
freedom after the reference outcomes are visible.

Before running calibration, specify a mechanical rule. One acceptable form is:

- orient every statistic so that larger means healthier;
- select a threshold that accepts at least 95% of the held-out-real reference
  and rejects at least 95% of the declared memorization/collapse references;
- verify the rule on the perturbed-copy and random-generator sensitivity
  controls; and
- if no threshold satisfies the healthy/failure separation, that statistic is
  not a valid categorical veto. Either drop it from the gate or return NO-GO;
  do not choose an outcome-specific cutoff.

The protocol must also define the per-replicate conjunction. The recommended
semantics are:

```text
replicate_pass(r) :=
    terminal_recall(r) > RECALL_GATE
    and regime_memorization_veto(r)
    and collapse_veto(r)

F1_pass := at least two of three replicates satisfy replicate_pass
```

This prevents recall in two replicates from being combined accidentally with a
veto that passes only in a different replicate. State whether every paired
`real_data` control must pass (recommended) or whether a failed pair merely
voids its corresponding replicate.

### 15.5 Replication currently supports only a conditional claim

Section 10 varies only the source/generator seed. Conditional on the fixed bank,
replay schedule, stochastic target stream, kernel calibration, and evaluation
reference, independent source seeds do provide a meaningful source-basin test.
However, all three outputs share one target realization, so the result does not
establish robustness to target-bank or minibatch-stream randomness. In
particular, a single lucky or pathological target schedule affects all three
replicates, and one replay/stochastic schedule is insufficient for a broad claim
that the target-sampling regime is the operative variable.

Choose one of these routes before execution:

1. **Preferred stronger design:** each replicate receives an independent source
   seed and an independent teacher seed (bank/replay schedule or stochastic
   stream), while replay and stochastic arms remain paired on the same source
   cloud and fixed evaluation/calibration setup. The categorical replicate is
   then the entire independent source/teacher unit.
2. **Conditional design:** retain one frozen bank/stream, but explicitly state
   that the F1 conclusion is conditional on that teacher realization and may
   select an F3A endpoint for that bank only. Do not describe replay-versus-
   stochastic differences as sampling-regime robustness without a follow-up
   teacher-seed replication.

The preferred design adds little particle-rollout cost and gives the cleaner
scientific conclusion. The non-primary controls also need explicit replication
counts: a single hashed `trained_bad` or AE tensor combined with a fixed replay
schedule produces repeated identical results rather than three independent
controls.

### 15.6 Frozen-update corrections

The §4.3 code snippet promises all behavior-changing arguments are explicit,
but it omits the denominator floor. Add
`denominator_floor=1e-30` explicitly to `KG.field`.

The snippet also promises that R11 gain is recorded, but
`corrected_teacher` reports its ratio only when given a mutable `report`
dictionary. The frozen update must have the following shape:

```python
drift, stats = KG.field(
    state, positives, state, branch, kernel,
    direction_mode="paper",
    normalization="rms",
    denominator_floor=1e-30,
    self_mask=False,
    diagnostics=True,
)
r11_report = {}
state = corrected_teacher(
    state + 0.5 * drift,
    positives,
    mode="scalar",
    report=r11_report,
)
```

Store `stats["ess_fraction"]`, `stats["collapsed_row_fraction"]`,
`stats["denominator_floor_fraction"]`, `stats["drift_rms_raw"]`, and
`r11_report["correction_ratio_median"]` at the declared resolution. Define
whether checkpoint diagnostics are the last step's values, a recomputation at
the checkpoint, or a summary over the preceding interval.

### 15.7 Remaining constructors and analysis rules to freeze

These are smaller than §§15.3–15.6 but must be resolved in the implementation
registry before any F1 output is exposed:

- Define `near_data` in `basin_interpolation`; if Phase 27 is being reproduced,
  it should be the index-paired `real_data` start.
- Give the exact `gaussian_mm` constructor: the pixel-space mean/covariance or
  per-coordinate moments, the split used to estimate them, clipping policy,
  and seeds.
- Freeze the `blend_near_gate` endpoints, pilot lambda grid, selection rule,
  tie-breaker, and pilot/reference separation. It is a sensitivity control and
  must not enter the null decision after tuning.
- Replace “within a declared tolerance” in §5.1 with actual numeric tolerances
  for every historical regression metric.
- Specify how uncertainty is computed for recall and KID. The conditional
  binomial fraction of covered evaluation samples does not by itself include
  generated-manifold uncertainty.
- Freeze actual numeric seed values and artifact/config filenames, not only the
  list of seed roles.

The Phase-29 collapse references currently exist as JSON summaries and image
grids, not reusable particle tensors. The implementation must regenerate and
hash the required states or replace them with reproducible synthetic
known-answer controls. Any regeneration cost belongs in §2.2.

### 15.8 Implementation readiness

At the time of this audit, no F1-specific executable, null-calibration artifact,
source tensor artifact, schedule registry, smoke artifact, or decision ledger
was present. Consequently none of §11's eight checks has passed. This alone is
sufficient for a confirmatory NO-GO even if every remaining prose issue is
resolved.

Required order:

1. apply §§15.3–15.7 to the frozen specification;
2. implement the calibration and F1 runner;
3. write the immutable config/seed registry;
4. run null calibration and require GO;
5. run one-step equivalence and historical K = 40 regression checks;
6. run replay reproducibility and disjointness/unique-index assertions;
7. run the all-arm/two-regime smoke test without interpreting it as F1;
8. update measured time/memory and freeze the K = 200 versus K = 1000 terminal
   decision; and only then
9. launch the three confirmatory replicate units.

### 15.9 Readiness table

| component | status after DESIGN v2 |
|---|---|
| scientific F1 question | **ready** |
| original KID/precision gate defects | **fixed** |
| original `5·SD` calibration defect | **fixed in direction** |
| null decision with uncertainty | **one repair remains** |
| memorization/collapse veto | **references defined; selection rule missing** |
| source-arm definitions | **mostly frozen; listed ambiguities remain** |
| replication scope | **conditional on one teacher realization unless revised** |
| exact teacher update | **correct field; denominator/report corrections remain** |
| executable F1 runner | **missing** |
| §11 pre-flight/smoke checks | **not run** |
| confirmatory F1 | **NO-GO** |

---

## 16. Revisions applied under the §15 readiness audit (Claude Opus 5, 2026-07-30)

Protocol is now **DESIGN v3**. Both remaining blockers accepted, all smaller
items applied. The audit's own verdict stands: **the confirmatory run remains
NO-GO until the runner exists and §11 steps 2–8 pass.** This revision closes the
specification only.

Verified before editing rather than taken on trust: `kernel_gradient.field`
does accept `denominator_floor` (default 1e-30, line 199) and does return
`ess_fraction`, `collapsed_row_fraction`, `denominator_floor_fraction` and
`drift_rms_raw` (lines 319–342); `objectives.corrected_teacher` writes
`correction_ratio_median` only into a supplied `report` dict. The audit's
specifics were accurate.

| audit item | revision |
|---|---|
| **15.3** point `q99` decides despite its own CI | **Blocker, accepted.** §2.1's decision rule replaced: `RECALL_GATE` held fixed at 0.05, exceedance count `E = #{r : recall_r > 0.05}` per null state, **one-sided exact Clopper–Pearson 95% upper bound** `p_null_upper`, max over the two null states, **GO only if `p_null_upper < 0.025`**. Orientation `E = 0 / 200 → 0.0149` and the 2-of-3 bound at tolerance `0.00184` recorded. Quantile and bootstrap CI still reported, never decisive. |
| **15.4** veto thresholds still had a free parameter | **Blocker, accepted.** §7 now fixes a mechanical rule before any reference outcome is seen: orient so larger = healthier; accept ≥95% of held-out real **and** reject ≥95% of declared collapse references; among qualifying thresholds maximize margin to the healthy 5th percentile; verify on perturbed-copy and `random_generator` controls; **if no threshold separates, the statistic is dropped from the gate (recorded) or the stage returns NO-GO**. No outcome-specific cutoff. |
| **15.4** per-replicate conjunction undefined | §8 now states the conjunction as code: `replicate_pass(r)` requires recall, memorization veto, collapse veto **and** the control, all within the same replicate; `F1_pass` is 2-of-3 over units. Recall in two replicates can no longer combine with a veto passing in a third. |
| **15.4** `real_data` semantics unstated | Adopted the recommended strong form: **every paired control must pass.** A unit failing its control is diagnosed and **re-run**, never tallied as a failed replicate, because it shares that unit's teacher seed and calibration — its failure indicates a configuration defect. Two or more control failures **void the run**. |
| **15.5** replication only conditional | **Accepted; route 1 adopted.** §10 rewritten: a replicate unit is an independent **(source seed, teacher seed)** pair — bank selection, replay schedule and stochastic stream all vary per unit. Kernel calibration and evaluation reference stay fixed; `replay`/`stochastic` remain paired **within** a unit so the regime contrast is still within-unit. The conclusion is no longer conditional on one teacher realization. |
| **15.5** controls not genuinely replicated | §10 adds control-arm replication: `trained_bad` and `ae_reconstruction` regenerated **once per unit** with that unit's source seed and hashed individually; `basin_interpolation` inherits the unit's pair; `ambient_noise` and `real_data` redrawn per unit. Three genuinely independent instances per arm. |
| **15.6** `denominator_floor` inherited | §4.3 snippet now passes `denominator_floor=1e-30` explicitly. |
| **15.6** R11 gain promised but unrecordable | §4.3 now constructs `r11_report = {}` and passes `report=r11_report`. v2 promised a diagnostic the call could not have produced. |
| **15.6** diagnostic keys and timing | §4.3 names all five keys exactly and defines timing: **mean over the preceding interval** for the four field statistics, **median over the interval** for the R11 ratio, with an extra pre-update evaluation at K = 0 so no interval is empty. Last-step values stored alongside. |
| **15.7** `near_data` undefined | §3 fixes `near_data` = the **index-paired `real_data` start**, matching Phase 27's actual construction (real data blended with the trained cloud), not the autoencoder reconstructions. |
| **15.7** `gaussian_mm` constructor | §3 pins `diagnose_phase15.gaussian_moment_match`: **per-coordinate** mean/sd (not full covariance), estimated from the 2 048-sample evaluation reference, per-replicate seed, **no clipping** — matching Phases 15–30 so the 0.867 / 0.000 anchors stay comparable. `identical_images` index declared and disjoint. |
| **15.7** `blend_near_gate` tuning could leak | §3 freezes it as a **sensitivity control that never enters the null decision**: endpoints `real_data`/`ambient_noise`, fixed pilot grid {0.5…0.9}, 20 pilot replicates on a **pilot-only real reference disjoint from the F1 evaluation reference**, smallest λ with pilot median in [0.02, 0.10], ties to smaller λ, **reported unavailable rather than re-tuned** if none qualifies. |
| **15.7** "declared tolerance" in §5.1 | §5.1 now tabulates numeric tolerances: `real_data` recall ±0.030 / KID ±0.0015; `trained_bad` recall exactly 0.000; `ae_reconstruction` recall ±0.030 / KID ±0.0050; one-step field equivalence ≤1e-6 relative L2. Recall tolerances deliberately wider, with the reason stated and a revisit rule if §2.1 finds a non-zero null. |
| **15.7** recall/KID uncertainty unspecified | §6 specifies it: **200 particle resamples with replacement**, recomputing recall each time so the generated k-NN radii are re-estimated, reporting 2.5/97.5 percentiles; labelled **conditional on the fixed reference**. The binomial reference-side interval is reported separately and explicitly as a *lower bound* on total uncertainty — the audit is right that it omits generated-manifold uncertainty. KID standard error over the same resamples. |
| **15.7** Phase-29 references are JSON, not tensors | §2.2 adds **~15 min** to regenerate and hash the collapse-reference tensors; §7's calibration set depends on them, so the cost is now on the ledger. |
| **15.7** freeze numeric seeds and filenames | Moved into §11 step 3 as an explicit prerequisite: the registry must contain **actual seed values and artifact filenames**, not role names. Deferred to implementation by necessity, but it is now a gating step rather than an aspiration. |
| **15.8** implementation ordering | §11 rewritten as the audit's nine-step order, with steps 2–8 marked as producing no scientific result and step 4 able to return NO-GO and halt the stage. |
| **15.2** correct v2 items | All retained unchanged. |

**Cost impact.** Independent teacher seeds per unit, per-unit control
regeneration, the Phase-29 tensor rebuild and the λ pilot raise the projection
from ~1.5–2 h to **~2.5–3 h**. Still well inside budget, and the §2.2 rule stands:
if measurement exceeds 3 h, cut checkpoints, never replicates.

**Readiness after v3**, against §15.9's table: the two remaining blockers are
closed and every listed ambiguity is frozen. What is unchanged is the last three
rows — **no runner, no calibration artifact, no smoke run** — so the confirmatory
F1 verdict stays **NO-GO**, exactly as §15.1 requires. The next authorized action
is implementation, not execution.

**One point returned to the auditor.** §15.5's route 1 makes each unit's teacher
seed independent, which strengthens the primary claim as intended. It also means
the `replay`-versus-`stochastic` contrast is now measured across three different
bank realizations rather than one, so a regime difference is no longer confounded
with a single schedule — but with three units the contrast is still only
categorical. If a *quantitative* statement about the sampling regime is wanted
later, it needs its own protocol with more units; §9's declared outcome for a
regime split should be read as directional only.
