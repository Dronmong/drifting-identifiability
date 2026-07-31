# Encoder-Independent Kernel Drifting — Phase-6 follow-up

## The deficit is a bandwidth artifact, and the program has been operating at the wrong bandwidth

*Long research pass on what Phase 6 left unresolved. Code:
`diagnose_phase7.py`. Artifacts: `phase7_probe.json`, `phase7_cifar.json`
(+ `.sha256`), stdout alongside. Development seeds
(`MASTER_SEED + 11000..`). Nothing here feeds a gate. 118 unit tests
passing, including a new equivalence test anchoring the balancing axis.*

---

## 0. What this pass found

Phase 6C measured free particles at 0.594 of the data's variance, and I
wrote that the deficit "belongs to the kernel drifting dynamics themselves".
That reading was too strong. Six measurements later:

1. **6C's number is a genuine attractor** — at 4× the budget with a trace,
   0.600 with window growth −0.003. The caveat I flagged is resolved.
2. **But it is a property of the *bandwidth*, not of drifting.** Across
   admissible bandwidths the fixed point runs **0.304 → 0.856**.
3. **Quality tracks it monotonically**: ED² runs **1.669 → 0.071** over the
   same ordering, and the program has been sitting near the bad end since
   Phase 2.
4. **Free particles at the best bandwidth beat the R11-corrected generator
   by 2.3×** (ED² 0.071 against 0.160), without memorizing.
5. **Cloud size is a second axis** — and the generator trains in the bad
   part of it.
6. Two mechanism hypotheses of mine died: the analytic blur factor, and
   balancing depth.

---

## 1. Resolving the 6C caveat: it is an attractor

6C logged no trace, so it could not distinguish "settled at 0.6" from "still
growing at step 600". Run at 2400 steps with a trace every 40, comparing the
last quarter against the quarter before it:

| arm (512 particles) | 2nd moment | window growth |
|---|---:|---:|
| ess = 0.50 (6C's setting) | **0.600** | **−0.0034** |
| ess = 0.10 | 0.503 | +0.0012 |
| τ = 0.20 | 0.469 | +0.0022 |

Flat to three decimals over the last 1200 steps. **6C's 0.594 was a fixed
point, not an unfinished trajectory** — the caveat is discharged, and in the
direction that makes the mechanism question well posed.

---

## 2. The fixed-point scale is set by the bandwidth

### At CIFAR-16, ordered by the kernel's realized neighbour count

τ = 0.02 is excluded first: **R15 finds it inadmissible** — 100% collapsed
rows, median affinity 1.9e-23, ESS not finite. It posted the *best* second
moment in the whole sweep (1.092, in band, converged) and it is a
numerically dead kernel whose particles random-walk to roughly the right
scale by accident. Without the Phase-5 admissibility guard this pass would
have reported it as the answer. Reported, not silently dropped.

Among the admissible arms, at 512 particles:

| arm | realized ESS | 2nd moment | ED² | score v2 |
|---|---:|---:|---:|---:|
| τ = 0.05 | 0.146 | 0.304 | 1.6693 | 6.428 |
| τ = 0.20 | 0.743 | 0.469 | 0.6320 | 3.824 |
| ess = 0.10 | 0.777 | 0.503 | 0.5285 | 3.504 |
| ess = 0.25 | — | 0.536 | 0.4289 | 3.061 |
| **ess = 0.50** *(the program's setting)* | 0.903 | **0.600** | 0.3491 | 2.701 |
| ess = 0.90 | 0.971 | 0.754 | 0.1274 | 1.582 |
| **τ = 1.00** | 0.978 | **0.856** | **0.0708** | **1.070** |

**Perfectly monotone in both columns.** More effective neighbours → higher
second moment *and* better quality, with no crossings. The relationship the
last three phases have been treating as a fixed defect is a dial, and the
program has had it near the wrong end since Phase 2.

The gain is large: moving from ESS 0.5 to τ = 1.0 improves free-particle ED²
by **4.9×** (0.349 → 0.071) and takes the second moment from out of band to
in it.

### It is not memorization

A particle method that simply migrated onto training points would score well
on ED² and badly on nearest-real. Checked directly:

| arm | ED² ratio | sw1 | patch ED² | **nearest-real** |
|---|---:|---:|---:|---:|
| τ = 1.00 | 1.413 | 1.123 | 1.246 | **0.944** |
| ess = 0.50 | 5.936 | 2.511 | 7.608 | **0.739** |

τ = 1.00 is better on **every** component, and its nearest-real ratio is
*closer to 1* than the incumbent's — the particles sit about as far from the
eval set as a fresh real sample does. Every ratio is near 1, meaning nearly
indistinguishable from real data on this suite. The incumbent setting is the
one that looks more memorization-like.

### The direction is not universal

In the low-dimensional harness the dependence runs the **other way** in the
regime tested — second moment 8.9 (τ=.05), 1.95 (τ=.1), 1.27 (τ=.2), 1.16
(τ=.5), 1.18 (τ=1) — i.e. small bandwidth *explodes* the cloud rather than
collapsing it, and d=32 sits at 2.9 rather than below 1. CIFAR's τ is
normalized by the median pairwise distance and the low-dimensional τ is not,
so the axes are not directly comparable; what carries across is that
**bandwidth strongly controls the fixed-point scale in both**, not the sign.
No simple "blur contracts" law survives, which is the next section.

---

## 3. Two mechanism hypotheses of mine, both refuted

### The analytic blur factor (M5) — refuted

For a Gaussian target and Gaussian kernel the kernel-weighted mean contracts
by `s²/(s²+τ²)`, so I predicted the deficit would follow it. Measured
against prediction in the low-dimensional harness:

| τ | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|---:|
| measured | **8.945** | 1.945 | 1.270 | 1.164 | 1.183 |
| blur predicts | 0.931 | 0.762 | 0.400 | 0.047 | 0.004 |

The blur factor predicts contraction at every bandwidth and increasingly so;
the measurement shows *expansion*, most extreme exactly where the blur is
weakest. Wrong in magnitude and in sign. Algorithm 2 is not a blur — the
negative term is doing the work the derivation ignored.

### Balancing depth (M4) — refuted, in the opposite direction

Algorithm 2 forms its plan as `A = sqrt(row ⊙ col)`: **one** symmetrized
normalization pass, not a converged balancing. Since the repo already
contains a converged `sinkhorn_map`, the natural hypothesis was
under-balancing — an under-balanced plan lets a data point claimed by many
particles keep too much weight, pulling the barycentric image toward dense
regions.

Balancing further does not repair it. It detonates:

| depth (d=8) | 0 *(the paper)* | 1 | 2 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| 2nd moment | **1.270** | 248.0 | 216.5 | 45.0 | 5.97 | 27.2 |

This **explains the paper's design choice**. Writing the field's affinity as
`A_ij = k_ij / sqrt(R_i C_j)`, the row factor `1/sqrt(R_i)` is common to the
positive and negative blocks and cancels, leaving

```
V_i = m_i (ȳ_i − x̄_i),    ȳ_i ∝ Σ_j k_ij C_j^{−1/2} y_j
```

so the fixed point is `ȳ_i = x̄_i`: the **attention-reweighted** kernel mean
of the data must match that of the cloud. `C_j^{−1/2}` is a *half-strength*
density correction. A converged balancing applies the full one — to the
negative block as well, which uniformizes the repulsion and makes it run
away. The square root is not an approximation to be improved; it is what
keeps attraction and repulsion in balance.

*Caveat:* depth ≥ 1 also changes how row mass splits between the positive
and negative blocks, so this is a comparison of two rules, not a clean
one-parameter family. It refutes "more balancing helps"; it does not
isolate why.

**A bug I found and a bug I didn't.** My first `drift_balanced` shifted the
column softmax by a row-wise maximum — correct only when every row's maximum
is equal, which holds unmasked (the self-distance is 0 in every row) and
fails once self-masking is on, giving a 0.7 relative error. Caught by
checking against the audited `lowdim_drift.drift_paper`, now a permanent
test (#30) that exercises the masked case specifically. I then suspected the
rectangular Sinkhorn of a marginal misspecification and checked before
"fixing" it — the two conventions give **exactly proportional** plans
(ratio 80.000000 everywhere), so that suspicion was wrong and M4's result
stands as measured.

---

## 4. Cloud size is a second axis — and the generator sits on the wrong end

In every CIFAR arm, 64 particles are worse than 512 in both columns:

| arm | 2nd moment (512 → 64) | ED² (512 → 64) |
|---|---|---|
| τ = 1.00 | 0.856 → **0.477** | 0.0708 → **0.7616** |
| ess = 0.90 | 0.754 → 0.453 | 0.1274 → 0.7896 |
| ess = 0.50 | 0.600 → 0.371 | 0.3491 → 1.0557 |

Shrinking the cloud roughly **halves** the second moment and costs an order
of magnitude on ED². The low-dimensional probe agrees that the *particle*
count is what matters and the *positive* count is nearly irrelevant
(512×64 → 1.27 and 512×512 → 1.26, against 64×64 → 2.59 and 64×512 → 2.55).

This matters because **the generator trains with batch 64**, computing its
field from 64 samples used as their own negatives. The generator has been
operating in the regime that costs a factor of ten, at a bandwidth that
costs another five.

---

## 5. What this does to R11

Phase 6A asked whether R11 was compensating for a mis-set **optimizer** and
answered no, cleanly: 0 of 12 cells, across a 60× learning-rate range and
three optimizers, reached the band.

This pass raises the same question about the **kernel**, with much better
support than the optimizer hypothesis ever had:

- the deficit R11 corrects is largely a bandwidth artifact;
- at a better bandwidth the uncorrected *particle* system already reaches
  0.856, inside the band;
- and it reaches ED² 0.071, **2.3× better than the R11-corrected generator's
  0.160**.

R11 has never been tested against a properly set kernel. That is now the
sharpest open question about the program's only positive result, and it is
the direct analogue of the test R11 already survived once.

Note this does not retract anything in Phase 6. Every Phase-6 arm ran at
ESS 0.5, so 6A's negative result stands exactly as scoped — the optimizer is
not the culprit. What changes is that a better-supported candidate now
exists.

---

## 6. Phase 7 — proposed design

### 7A — sweep the kernel for the generator, crossed with R11

The direct analogue of 6A, on the axis this pass implicates.

| axis | values |
|---|---|
| bandwidth | ess 0.5 (incumbent), ess 0.9, τ = 1.0, τ = 2.0 |
| field cloud | batch 64 (incumbent), 256, 512 |
| R11 | off, on |

R15 admissibility is evaluated **first**; inadmissible cells are reported
with their health numbers and not scored. Optimizer held at Adam/2e-3 — 6A
showed it does not matter, and holding it fixed keeps this a one-axis test.

**Gate, declared in advance and able to retire R11:** if any admissible
(bandwidth, cloud) cell *without* R11 reaches a second-moment ratio inside
`[0.7, 1.3]` and an ED² within 25% of the best R11 cell, then **R11 is
superseded by setting the kernel correctly**, and Phases 3, 5 and 6 are
re-scoped as having measured a bandwidth artifact.

Given that free particles already clear both conditions at τ = 1.0, this
gate has a real chance of firing — which is exactly why it should be frozen
before the run.

### 7B — the particle/generator gap, at the good operating point

At ESS 0.5 the corrected generator (0.160) beat free particles (0.349). At
τ = 1.0 free particles (0.071) beat the corrected generator. **The ordering
inverts**, which means the generator — not the field — becomes the
bottleneck once the kernel is set properly. 7B measures the gap across the
bandwidth axis and asks whether the generator can be made to track the
particle result it is supposed to amortize.

### 7C — is there a bandwidth *rule*, or only a lucky value?

τ = 1.0 is the largest value tested and quality was still improving; the
sweep has not found the optimum, only established a direction. 7C extends
past it (τ = 2, 4, 8) to locate the turn, and asks whether the good setting
is predictable from a target-only statistic — the ESS-targeting rule was
supposed to be exactly that and chose 0.5, which this pass shows is poor.
**A calibration rule that picks the right bandwidth from target data alone
is worth more than the best hand-chosen τ**, because it is what transfers.

### Priority

**7A first.** It is the one that can overturn a standing claim, and no
further mechanism work is worth doing until it is settled.

---

## 7. What this pass does not establish

- The bandwidth sweep is at CIFAR-16, raw pixel geometry, free particles,
  3 seeds. **The generator was not run at the better bandwidths at all** —
  that is Phase 7A, and every statement about R11 above is a hypothesis
  until it runs.
- τ = 1.0 is the sweep's edge, not a located optimum.
- The low-dimensional and CIFAR bandwidth axes have opposite signs in the
  regimes tested; only "bandwidth controls the scale" transfers, not the
  direction.
- The refutation of balancing depth compares two rules, not a clean family.
- The mechanism setting the fixed point remains unknown. Six hypotheses were
  already refuted; this pass adds two more (blur factor, balancing depth) and
  proposes no ninth. What it does instead is show the fixed point is a
  *controllable* quantity, which is more useful than another mechanism guess.
- Nothing here concerns ImageNet, FID, or the paper's trained model.

## 8. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy `
  python -m numerics.encoder_independent_drifting.diagnose_phase7 `
  --stage lowdim --seeds 3 --steps 600 --long-steps 3000 --dims 8,32

uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.diagnose_phase7 `
  --stage cifar --seeds 3 --steps 2400 --resolution 16 `
  --out numerics/encoder_independent_drifting/phase7_cifar.json
```
