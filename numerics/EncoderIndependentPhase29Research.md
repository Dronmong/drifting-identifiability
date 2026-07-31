# Encoder-Independent Kernel Drifting — research pass on the assignment proposal

## The hard-assignment test was already run in Phase 13. It collapses with fresh latents, and that unifies with Phase 28 into one mechanism.

*Research pass. No new training. Evidence: `phase13_probe.json` /
`.stdout.txt`, `phase13.json`, `run_phase12.py`, `phase28_probe.json`,
`phase28_warm512.json`.*

---

## 1. Correcting my own recommendation

Phase 28 §4 recommended "an assignment-based teacher (each cloud point matched
to *one* real sample, Hungarian or Sinkhorn-hard)" and noted the program "has
Sinkhorn machinery already". That undersold it badly: **the program has already
run this experiment.** `diagnose_phase13.py`'s stated purpose includes

> *"I4 the proposed fix, prototyped: an exact balanced hard assignment
> (Hungarian on the batch), which is both non-colliding and non-averaging — the
> two properties the failed schemes each lacked."*

That is the same reasoning I re-derived, arrived at independently 15 phases
earlier. I should have searched before recommending. Recorded because the
program's value depends on not silently re-running its own experiments.

---

## 2. What the prior art actually found

Phase 13, 3 seeds, targets are an externally-evolved particle cloud:

| arm | latents | tail | 2nd moment | ED² fresh | distinct targets |
|---|---|---:|---:|---:|---:|
| B0_self_fresh | fresh | 0.003–0.005 | 0.24–0.44 | 0.93–1.77 | — |
| B2_index_FIXED | fixed | 0.197–0.229 | 0.82–0.87 | **0.164–0.177** | 256 |
| B3_index_fresh | fresh | — | — | **6.68** | 256 |
| **B4_hungarian_fresh** | **fresh** | **0.0006–0.0010** | 0.55–0.63 | 0.78–0.87 | **256** |
| **B5_hungarian_FIXED** | **fixed** | **0.29–0.35** | 0.79–0.95 | **0.226–0.301** | **256** |
| B6_nearest_fresh | fresh | 0.46–0.48 | **0.000** | **15.9–16.2** | **1** |
| B7_sinkhorn_fresh | fresh | 0.0007 | 0.29–0.43 | 1.10–1.65 | — |

*(particle-cloud ceiling: tail 0.40, ED² 0.075)*

Three things are settled by this table.

**Many-to-one nearest neighbour collapses totally.** `B6` has **distinct = 1** —
all 256 cloud points assigned the same target — with second moment exactly
0.000 and ED² 16. This confirms the structural argument: nearest-neighbour is
many-to-one and cannot preserve coverage. It also explains Phase 22's cliff,
since sharpening the kernel bandwidth pushes the weights *toward* many-to-one.

**Bijection does prevent collision.** Both Hungarian arms hold **distinct = 256**.
The structural property works as intended.

**But bijection is not sufficient.** `B4_hungarian_fresh` has **the worst tail of
any arm measured (0.0006)** despite perfect non-collision. The bijection buys
nothing when the latents are drawn fresh each step.

**The discriminating variable is not the assignment. It is whether the
latent→target correspondence is stable.** Compare within each target scheme:

| scheme | fresh latents | fixed latents |
|---|---:|---:|
| index | ED² 6.68 | ED² **0.164** |
| Hungarian | tail 0.0009 | tail **0.29–0.35** |

---

## 3. This unifies with Phase 28 into a single mechanism

Phase 28 concluded that a parametric generator collapses because it must fit a
family of kernel-averaged teachers with shared parameters. Phase 13 shows the
same collapse with targets that are **not averaged at all** — a bijection onto
distinct real particles — whenever the latents are fresh.

So averaging was never the operative property. The operative property is:

> **With fresh latents, the target is a random variable conditioned on nothing
> stable. Squared-error regression against it converges to its conditional
> mean, and the conditional mean of a target that is re-matched every step is
> the population mean. Coverage collapses regardless of whether any individual
> target was sharp.**

This is a stronger and simpler statement than Phase 28's, it covers both
results, and it explains why free particles behave differently (Phase 26: real
data stationary) — free particles have no shared parameters and no latent
indexing, so there is no conditional mean to collapse onto.

It also predicts, correctly, every "fixed latents" result in the program:
Phase 13's B2/B5 and Phase 28's memorization (KID 0.061, recall 0.224,
recognizable objects) are the same regime — a stable correspondence.

**Consequence for the drifting question.** The paper's objective uses fresh
latents. Under this mechanism, *no* choice of kernel, bandwidth, geometry,
teacher correction or encoder can prevent the collapse, because none of them
makes the correspondence stable. That is the cleanest statement of why 28
phases moved KID between 0.06 and 0.26 and never produced coverage.

---

## 4. What has *not* been tested, and is the one thing worth testing

Every scheme in Phase 13 assigns **cloud → target**: for each generated point,
find its target. All of them either collide (nearest), average (Sinkhorn
barycentric), or depend on a frozen latent set (index, Hungarian-fixed).

**The opposite direction has never been tried here.** For each *real* sample,
draw several candidate generations and pull the nearest one toward it. That is
**IMLE** (Implicit Maximum Likelihood Estimation, Li & Malik 2018), and its
design goal is exactly the quantity this program has measured at 0.000:

- cloud→real assignment optimizes **precision** and permits mode collapse — many
  generated points may claim one real sample (`B6`, distinct = 1);
- real→cloud assignment optimizes **recall** and cannot drop a mode — every real
  sample claims a generated sample by construction.

Precision has never been this program's problem. Phase 28's drifting output has
precision 0.523–0.648 with **recall 0.000**; the moment-matched Gaussian scores
precision 0.842 with recall 0.000. **Recall is the entire deficit, and the
real→cloud direction is the only scheme that targets it.**

IMLE also partially addresses §3's stability problem: drawing *k* candidates per
real sample and taking the nearest produces a correspondence that changes slowly,
because the nearest generation to a fixed real point is stable under small
parameter updates. It is not a frozen pairing, so it is not memorization.

**This is a different algorithm, not drifting.** It must be reported as a test of
the §3 mechanism and as a measurement of what this harness can do at all — never
as a drifting result.

---

## 5. Concrete plan

**Step 1 — re-measure the prior art under the validated metrics (cheap).**
Phase 13's numbers are all ED² and tail, and the Phase 15 audit showed ED² is
saturated by a structureless Gaussian. Re-score `B5_hungarian_FIXED`,
`B4_hungarian_fresh` and `B6_nearest_fresh` on **KID, precision and recall**,
against two anchors now available: the memorization ceiling (KID 0.061, recall
0.224) and the drifting baseline (KID 0.15, recall 0.000). Without this, "ED²
0.30" is uninterpretable.

**Step 2 — implement IMLE and measure it on the same axes.** For each real sample
in a batch, draw *k* = 8 generations, take the nearest in pixel space, regress it
toward that real sample. Fresh latents throughout, so it is a genuine generative
objective and not memorization. Report KID, precision, **recall**, alpha, and the
distinct-target count that exposed `B6`.

**Declared before running:**

- **IMLE reaches recall > 0.1 with fresh latents** → §3's mechanism is right and
  escapable; the deficit was the assignment *direction*, and this harness can
  do generative modelling. Drifting's failure becomes a specific, explained
  negative rather than a property of the setup.
- **IMLE also collapses to recall ≈ 0** → the obstruction is the generator or
  the harness, not the objective family, and *every* result in 28 phases is
  bounded by something outside the drifting question. That would be the
  strongest closure available and would make the memorization ceiling the only
  meaningful number the harness has produced.

**Cost.** IMLE at k = 8 is 8× the generator forward passes per step but needs no
kernel, no Gram matrix and no bandwidth, so it is *cheaper* per step than
drifting. 6 000 steps is minutes.

---

## 6. What this pass rules out

- **Sinkhorn barycentric projection** — averages by construction (`B7`, tail
  0.0007). §3 explains why it cannot help.
- **Nearest-neighbour cloud→real** — collapses to distinct = 1 (`B6`).
- **Hungarian with fresh latents** — bijective and still the worst tail measured
  (`B4`).
- **Hungarian or index with frozen latents** — works, but it is memorization,
  which Phase 28 already quantified more directly (KID 0.061, recall 0.224).
- **Any further kernel, bandwidth, geometry or encoder variation** — §3 shows
  none of them touches the operative variable.

---

## 7. Scope

- Phase 13's arms targeted an externally-evolved **particle cloud**, not real
  data, so its absolute numbers are not directly comparable to Phase 28's.
  The *contrasts within* that table (fresh vs fixed, bijection vs nearest) are
  what §2 relies on, and those are internally controlled at 3 seeds.
- §3 is a mechanism claim assembled from two independent probes, not a
  measurement in its own right. Step 2's declared branches are what test it.
- Still not the paper's method, and Step 2 would be further from it again.
