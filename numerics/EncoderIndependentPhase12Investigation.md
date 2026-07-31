# Encoder-Independent Kernel Drifting — investigating Phase 12

## The assignment was never the problem. The *correspondence* is.

*Thorough investigation of Phase 12. Code: `diagnose_phase13.py`. Artifact:
`phase13_probe.json` (+ `.sha256`). 3 seeds, 600 steps, CIFAR-16, target ESS
0.9, 512 particles. Nothing feeds a gate.*

---

## 0. Summary

| arm | latents | tail | 2nd moment | ED² | distinct particles | target tail | target 2nd |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 self-referential | fresh | 0.0030 | 0.436 | 0.9908 | — | — | — |
| **B1 self-referential** | **fixed** | 0.0061 | 0.347 | **1.0079** | — | — | — |
| **B2 external, index** | **fixed** | **0.2210** | **0.869** | **0.1640** | 256 | 0.3687 | 1.004 |
| B3 external, index | fresh | 0.1573 | 0.002 | 6.6794 | 256 | 0.3687 | 1.004 |
| **B4 external, Hungarian** | **fresh** | **0.0009** | 0.550 | **0.8238** | 256 | **0.3267** | **1.105** |
| B5 external, Hungarian | fixed | 0.3430 | 0.827 | 0.3014 | 256 | 0.3669 | 0.976 |
| B6 external, nearest | fresh | 0.4663 | **0.000** | 16.155 | **1.0** | **0.0000** | **0.000** |
| B7 external, Sinkhorn | fresh | 0.0007 | 0.315 | 1.4966 | — | **0.0037** | 0.396 |

*(the particle cloud itself: tail 0.4059, ED² 0.0752)*

Three findings, in order of importance.

---

## 1. The confound is resolved — my attribution was right

Phase 12's A2 differed from the baseline in **two** ways: the target was
external *and* the latents were fixed. I attributed the whole effect to the
first without running the control. The control:

| | tail | 2nd moment | ED² |
|---|---:|---:|---:|
| B0 self-referential, **fresh** latents | 0.0030 | 0.436 | 0.9908 |
| B1 self-referential, **fixed** latents | 0.0061 | 0.347 | **1.0079** |

**Fixed latents alone do nothing** — 1.0079 against 0.9908, indistinguishable
on every readout. The external target is load-bearing, as claimed.

---

## 2. Both failures diagnosed directly, not by inference

Phase 12 explained the two failing assignments by argument. Measuring the
**target itself**, before the generator ever sees it:

**Greedy nearest-neighbour degenerates completely.** All 256 generated
samples claim **1.0 distinct particles** — a single point. The target's tail
is 0.0000 and its second moment 0.000. The collapse is not a training
pathology; the target is a single point from the first step.

**The barycentric projection destroys the target's shape.** Target tail
**0.0037** against the particle cloud's 0.4059 — a factor of 110 — and second
moment 0.396. The contraction happens in the assignment, before training.

**And the Hungarian assignment fixes exactly that.** Target tail 0.327–0.367
against the cloud's 0.406, second moment 0.976–1.105. A balanced hard
assignment preserves the target essentially perfectly, as designed.

---

## 3. The finding: a perfect target is not enough

This is the result that redirects the line.

| | target tail | → | generator tail | ED² |
|---|---:|---|---:|---:|
| B5 Hungarian, **fixed** latents | 0.3669 | → | **0.3430** | 0.3014 |
| B4 Hungarian, **fresh** latents | 0.3267 | → | **0.0009** | 0.8238 |

**The same assignment, producing a target of the same quality, transfers its
tail when the latents are fixed and destroys it when they are fresh** —
0.3430 against 0.0009, a factor of 380.

So the problem was never assignment quality. Phase 12 concluded the open
problem was "an assignment that is balanced and hard"; the Hungarian
assignment is exactly that, it preserves the target, and it does not fix the
generator on fresh latents.

**What fixed latents provide is a *stable correspondence*.** With them, latent
`z_i` maps to particle `π(i)` on every step, and the generator learns one
consistent function. With fresh latents the assignment is recomputed each
step against a different sample, so a given region of latent space is matched
to different particles over time. The generator averages that inconsistency —
and averaging is what destroys the tail, exactly as the tail-destruction pass
found for the self-referential teacher.

> **The requirement is correspondence *stability*, not assignment quality.
> A per-step optimal assignment is still an unstable correspondence.**

---

## 4. The concrete direction

B2 and B5 are, structurally, a generator fitted to a converged particle cloud
under a fixed pairing — the same object as Phase 4's `fit_to_free` and the
contraction pass's N3, now with the good bandwidth and measured on fresh
latents. B2 reaches **ED² 0.1640 against the particle cloud's 0.0752**: the
amortizer is 2.2× worse than the particles it fits, from only **256 pairs**.

That makes the next question precise and cheap:

**Does the amortizer close the gap to the particle cloud as the number of
(latent, particle) pairs grows?**

Sweep the pair bank at 256 → 1024 → 4096 → 16384, with the correspondence
fixed once and reused (assign at the start, then keep it), and measure ED²
against the particle cloud's own 0.0752. This is the amortization question
properly posed, and it is the natural experiment because:

- it tests the one ingredient shown to matter (a stable correspondence) at
  the one scale that has never been varied;
- it needs no new mechanism, only more pairs;
- it has a clear success criterion — approaching 0.0752 — and a clear
  failure: if ED² plateaus well above the particle cloud regardless of pair
  count, the map cannot represent the particle law and this line ends.

**Secondary arm worth carrying:** a *persistent* assignment with fresh latents
— assign each particle a permanent latent at the start, then sample latents
from that bank with noise. That interpolates between B4 and B5 and would show
whether stability alone, without literal memorization, is sufficient.

### What I would not do

Continue iterating on assignment algorithms. §3 shows an exactly optimal
assignment does not help on fresh latents; a better one cannot.

---

## 5. Caveats

- **The `gap` column in the artifact is not interpretable.** `ed2_fresh` uses
  512 samples and `ed2_on_training_latents` uses 256, and ED² estimates
  depend on sample size, so the ratio confounds generalization with
  estimator bias. It is recorded but should not be read as an amortization
  gap; §4's comparison against the particle cloud's ED² at matched size is
  the sound one.
- No R11 arm was run in this probe, so the numbers here are not directly
  comparable with Phase 12's R11 (0.1112) — different seeds and setup.
- B3 (index pairing with fresh latents) is a random correspondence by
  construction and fails as expected; it is reported as the control it is,
  not as evidence about assignments.
- One dataset, one geometry, one bandwidth, 3 seeds, 600 steps, 512
  particles, 256 pairs.
- Nothing here concerns ImageNet, FID, or the paper's trained model.

## 6. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.diagnose_phase13 `
  --seeds 3 --steps 600
```
