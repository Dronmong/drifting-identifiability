# Encoder-Independent Kernel Drifting — Phase-8 follow-up

## Four more negatives, one partial mechanism, and a measurement that invalidates an old refutation

*Long methodical pass on Phase 8's results. Code: `diagnose_phase9.py`.
Artifacts: `phase9_probe.json`, `phase9_subspace.json`, `phase9_genavg.json`
(+ `.sha256`). Development seeds (`MASTER_SEED + 15000..`), 3 seeds, 600
steps, CIFAR-16, target ESS 0.9. Nothing feeds a gate.*

---

## 0. Where Phase 8 left it

The generator's second-moment deficit (~0.47 converged) is invariant to the
optimizer, learning rate, η, kernel bandwidth, field cloud size, latent
dimension, model capacity (36×), teacher-fitting quality (2.3×) and training
budget (10×). It is an equilibrium. Free particles under the identical field
at the identical bandwidth reach 0.998.

**First, a statistical check on Phase 8's own claim.** The width medians were
0.354 / 0.401 / 0.349 / 0.331, which invited reading a slight downward trend:

| quantity | value |
|---|---:|
| between-width sd of medians | 0.0256 |
| within-width seed sd | **0.0855** |
| Spearman(width, second moment), 12 runs | **+0.098** |

Seed noise is **3.3× larger** than anything the width axis contributes. There
is no trend, downward or otherwise — capacity does exactly nothing, and
Phase 8's "if anything it falls" should be read as "flat".

---

## 1. H11 — fluctuation does not hold the cloud open (refuted)

In the mean-field limit the generator and the particles implement the *same*
flow, and a partial step only makes the generator slower. So the difference
must be something the mean-field limit discards. The candidate: particles
keep their realized random displacements **in their state**, while the
generator regresses onto a target and averages them away. This is the same
phenomenon the MMD-gradient-flow literature treats with noise injection.

The test reverses the arrow — instead of adding noise to the generator, take
it away from the particles by averaging the field over K independent batches
before moving them:

| K | 1 | 2 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|---:|
| **P1** positive-batch averaging | 1.012 | 1.001 | 1.004 | 1.007 | 1.006 |
| **P2** negative-side averaging | 1.022 | — | 1.019 | 1.023 | — |

A **64× variance reduction moves the particle equilibrium by 0.006.**
Refuted, from both sides of the field.

In hindsight the hypothesis was weak: the particle system is nearly
deterministic given its cloud, and the field is already an average over 64
positives, so there was little stochasticity to lose.

---

## 2. P3 — the noise budget inverts, and it invalidates an old refutation

| step | signal | noise | **noise / signal** |
|---|---:|---:|---:|
| 0 | 254.69 | 1.72 | **0.007** |
| 50 | 119.02 | 144.55 | 1.254 |
| 150 | 105.41 | 159.31 | 1.429 |
| 300 | 57.54 | 211.13 | 3.669 |
| 599 | 44.34 | 197.72 | **4.459** |

Along the generator's trajectory the signal decays 5.7× while the noise grows
115×, inverting the ratio by **three orders of magnitude**.

Phase 3 refuted "minibatch-noise regression attenuation" by measuring a noise
fraction of ≈0.001 and calling it far too small. **That figure corresponds to
initialization** — the one point on the trajectory where it is true. By the
time the generator is anywhere near its equilibrium the field is
noise-*dominated*. The refutation rested on a number measured in the wrong
place.

**This does not resurrect the hypothesis** — §3 tests it directly and it
fails — but the original refutation should not be cited as evidence, and the
plan is corrected accordingly.

---

## 3. Target noise is not it either (refuted, and an apparent positive explained)

The differential test P3 implies: apply P1's exact knob to the **generator**
instead of the particles. P1 was flat for particles, so if the regression is
where noise does its damage, the two systems must respond differently.

| K | per-seed | median 2nd | median ED² |
|---|---|---:|---:|
| 1 | 0.372, 0.409, 0.338 | **0.372** | 1.1201 |
| 4 | 0.466, 0.436, 0.427 | 0.436 | 0.9381 |
| 16 | 0.474, 0.412, 0.484 | **0.474** | 0.9099 |

At face value this is a positive: 0.372 → 0.474, about three seed standard
deviations, monotone, with ED² improving 1.120 → 0.910. The automated flag
duly reports `rises_with_averaging: True`.

**It is not an equilibrium effect.** Compare against the long-run measurement
from Phase 8:

| | second moment |
|---|---:|
| K = 1 at 600 steps | 0.372 |
| **K = 16 at 600 steps** | **0.474** |
| **K = 1 at 6000 steps** (Phase 8, N6) | **0.472** |

Sixteen-fold field averaging reaches, at 600 steps, precisely the value plain
training reaches at 6000. **Averaging accelerates convergence; it does not
move the fixed point.** Both roads end at ~0.47.

That is worth having as a practical result — a 16× cleaner field buys a 10×
shorter run — but as a mechanism for the deficit it is negative. Recorded
this way rather than as the positive the flag reports.

---

## 4. P4 — spectral confinement is real, partial, and saturates

The hypothesis N4 never actually tested. N4 swept the latent dimension and
found the deficit flat — but latent 512 still produced only 0.0058 of its
variance beyond the top 32 directions, against real data's 0.1375. "More
latent room" was refuted; "a cloud missing its tail balances at a smaller
radius" was not.

Free particles confined to the data's top-k principal directions, scored
against the data's variance **inside the same subspace**:

| k | in-subspace 2nd | raw 2nd | data energy in top-k | ED² |
|---|---:|---:|---:|---:|
| unconfined *(P1 control)* | **1.012** | 1.012 | 1.000 | 0.083 |
| 511 *(the data's own span)* | 0.917 | 0.917 | 1.000 | 0.073 |
| 128 | 0.700 | 0.682 | 0.975 | 0.213 |
| 32 | 0.664 | 0.573 | 0.864 | 0.433 |
| 8 | 0.744 | 0.497 | 0.669 | 0.726 |

**This is the first intervention that has moved a particle equilibrium in the
generator's direction** — 1.012 down to ~0.66–0.70. But it **saturates**:
confining 16× harder (128 → 8) does not push past 0.7, and it never
approaches the generator's 0.47. Spectral concentration is a genuine
contributor to roughly the first third of the gap and is not the mechanism.

**A bug in my own probe, found and corrected.** The in-subspace
normalization used the *requested* k rather than the basis's actual rank, so
the k = 768 arm — where the rank is capped at 511 by the 512-sample eval pool
— was misnormalized by 512/768 and printed 0.611 instead of 0.917. The table
above is recomputed; the raw column was always correct. Fifth instrument
defect recorded in this program.

---

## 5. The state of the mechanism question

Eleven hypotheses have now been proposed and refuted. The generator's deficit
is invariant to every one of:

| axis | span | source |
|---|---|---|
| optimizer, learning rate | 3 × 60× | 6A |
| η | 100× (inert under Adam) | R24 |
| kernel bandwidth | realized ESS 0.82–0.99 | 7A |
| field cloud size | 64–512 | 7A |
| latent dimension | 8–512 | N4 |
| model capacity | 36× parameters | 8A |
| teacher-fitting quality | 2.3× realized | 8B |
| training budget | 10× | N6 |
| **field noise** | **16× variance** | **P5** |

Only two things have ever moved it: **changing what the teacher asks for**
(R11), and **confining the cloud spectrally** (P4, partially, in the particle
system).

---

## 6. Where to go from here

### The recommendation: stop testing interventions, solve a tractable case

Nine phases of intervention-testing have produced eleven refutations and one
unexplained working reform. The marginal value of a twelfth intervention is
low. What has actually worked, twice, is **shrinking the problem until it is
solvable** — R23 found the second moment by dropping to 8-dimensional
Gaussian mixtures, and P4 above only became askable once the question was
posed as a subspace constraint.

**Proposed Phase 9: the linear-generator case.** Take `f(z) = Az + b` with
Gaussian latent and Gaussian data. Then:

- the generator's pushforward is exactly Gaussian, parameterized by `AAᵀ`;
- the kernel mean-shift field between two Gaussians is available in closed
  form;
- the stop-gradient regression's fixed point is
  `E_z[(∂f/∂θ)ᵀ V] = 0`, which for a linear map is a **matrix equation in
  `AAᵀ`** — solvable, or at worst numerically solvable to high precision.

Three outcomes, all informative:

1. **The linear model shows the same deficit** → the mechanism is in the
   regression structure itself and is now analytically accessible. This is
   the outcome that would finally produce a derivation rather than a
   refutation.
2. **The linear model has no deficit** → the mechanism requires the
   nonlinearity or the architecture, which narrows the search enormously and
   contradicts every "it's the objective" reading.
3. **The linear model has a *different* deficit** → the size of the effect
   becomes a function of something computable.

It is cheap, it is the only route left that can produce an explanation rather
than another elimination, and it is directly in the tradition of the two
passes in this program that actually worked.

### In parallel: consolidate what is already established

Independently of the mechanism, the program has results worth packaging that
have never been run together:

- the bandwidth optimum (target ESS ≈ 0.9, Phase 7C) — worth **4.9×** on
  free-particle ED² and never used by any arm before Phase 8;
- R11, replicated across five independent settings;
- the anchor (~3.5% on raw geometry, replicated 3/3), **untouched since
  Phase 2** and never combined with either of the above;
- field averaging (§3) as a convergence accelerant: 16× cleaner field for a
  10× shorter run.

No single configuration has ever combined the good bandwidth, R11 and the
anchor. That is a concrete deliverable, it needs no new mechanism, and it is
the natural thing to report if the linear case does not resolve the question.

### What I would not do

Another sweep of a training hyperparameter. The table in §5 spans nine axes
and none of them matters; a tenth is unlikely to differ, and the cost is
about two hours per phase.

---

## 7. Scope and caveats

- Every measurement is CIFAR-16, raw pixel geometry, paper Algorithm-2 field,
  3 seeds, 600 steps except where stated.
- P4's confinement uses the *eval* pool's principal directions, so k = 511 is
  the maximum available rank; a true unconfined control comes from P1, run
  under matched settings.
- P5's acceleration result is at one cloud size and one bandwidth.
- P3's noise/signal figures are measured along the generator's trajectory
  only; a matched measurement along the *particle* trajectory was not made,
  so "noise-dominated near equilibrium" may simply be what any converged
  system looks like and should not on its own be read as diagnostic.
- Nothing here concerns ImageNet, FID, or the paper's trained model.

## 8. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.diagnose_phase9 `
  --stage all --seeds 3 --steps 600
```
