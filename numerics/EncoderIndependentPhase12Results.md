# Encoder-Independent Kernel Drifting — Phase 12 results

## The external target works. The assignment does not.

*Protocol: `EncoderIndependentPhase12Protocol.md`, frozen before the run
(with one amendment recorded there, made after a smoke test and before any
full run). Code: `run_phase12.py`. Artifact: `phase12.json` (+ `.sha256`),
stdout in `phase12.stdout.txt`. 28.5 minutes, 3 fresh seeds
(`MASTER_SEED + 23000..`), CIFAR-16, target ESS 0.9, 512 particles, 600 steps.*

---

## 0. Summary

| arm | latents | tail | 2nd moment | ED² | score | cost |
|---|---|---:|---:|---:|---:|---:|
| A0 self-reference | fresh | 0.0041 | 0.331 | 1.2996 | 5.393 | 1.0× |
| **A1 self + R11** | fresh | 0.0523 | 0.873 | **0.1112** | **1.337** | 1.0× |
| **A2 particles, index pairing** | **fixed** | **0.3243** | **0.922** | **0.2091** | **1.873** | 3.6× |
| A3 particles, nearest | fresh | 0.3616 | **0.000** | 19.51 | 25.48 | 3.6× |
| A4 particles, Sinkhorn | fresh | **0.0007** | 0.343 | 1.3710 | 5.480 | 3.6× |
| A4R Sinkhorn + R11 | fresh | 0.0003 | 0.895 | 0.5356 | 3.003 | 3.6× |
| A3P pre-converged, nearest | fresh | 0.4706 | **0.000** | 15.70 | 18.84 | 3.6× |
| A4P pre-converged, Sinkhorn | fresh | 0.0008 | 0.384 | 1.1967 | 4.855 | 3.6× |

*(real data tail = 0.1320; gate ceiling = 0.139)*

**The gate does not fire** — no arm reaches ED² within 25% of R11's 0.1112.
R11 survives a sixth supersession test.

**But the hypothesis the phase was built on is confirmed.** A2 — an external
target, no self-reference — reaches:

- **tail 0.3243 against A0's 0.0041, a factor of 79**, and 2.5× real data's
  own 0.132;
- **second moment 0.922, inside the band**, against A0's 0.331;
- **ED² 0.2091 against A0's 1.2996 — 6.2× better**, and score 1.873 against
  5.393;
- per-seed and stable: second moment 0.946 / 0.910 / 0.922.

Removing the self-referential anchor does exactly what the audit predicted.
It misses the gate by 1.5× on ED², at 3.6× the training cost.

---

## 1. Why the amortizing arms failed

A2 uses **fixed latents**, so it learns one 256-point map. The arms designed
to generalize that — fresh latents plus an assignment — both fail, and for
two different and identifiable reasons.

### Nearest-neighbour collapses (A3, A3P)

Second moment **0.000** on every seed, ED² 13.9–23.7. A greedy match lets
many generated samples claim the same particle, so the regression target
degenerates onto a handful of points and the cloud implodes. Its *tail* is
high (0.36–0.47) precisely because the target is a few widely-separated
particles — the shape statistic looks healthy while the generator is
destroyed, which is a good reminder that the tail is a means and ED² is the
outcome.

### Sinkhorn's barycentric projection contracts (A4, A4P)

Tail **0.0007** — a fifth of the self-referential baseline's, and 190× below
A2's. The barycentric projection is a conditional expectation, so by the law
of total variance it has strictly lower variance and lower effective rank
than the cloud it projects onto. **It reintroduces exactly the contraction
this program has spent nine phases characterizing**, and it does so more
severely than the self-referential teacher it was meant to replace.

That is a design error on my part rather than a fact about drifting: a
contracting assignment cannot deliver an uncontracted target. It should have
been foreseen from the program's own findings.

---

## 2. What this establishes

**Established:**

- **an external, non-self-referential target restores both shape and scale** —
  tail ×79, second moment 0.331 → 0.922, ED² 6.2× better — confirming the
  audit's reading;
- the effect **generalizes to fresh latents**: A2's numbers are all measured
  on a fresh probe, so the learned *map* carries the tail, not just the 256
  fitted points;
- **the open problem is the assignment, not the principle.** Both schemes
  tried are pathological: greedy matching collides, barycentric projection
  contracts;
- the cost of an external target is **3.6× the training kernel work**;
  inference is unchanged at NFE = 1.

**Not established:** that this can be amortized. A2 pairs a fixed latent set
with a fixed particle index, which does not scale to a general sampler. Until
an assignment works, this is a demonstration that the diagnosis is right, not
a method.

---

## 3. The next experiment

The requirement is now precise: **an assignment that is balanced (no
collisions) and hard (no averaging).** Neither scheme tried satisfies both —
nearest is hard but unbalanced, Sinkhorn barycentric is balanced but soft.

Three candidates, in increasing cost:

1. **Sinkhorn followed by rounding** — compute the entropic plan, then take a
   hard assignment from it (row-wise argmax with capacity, or greedy
   rounding). Balanced *and* hard, and reuses the machinery already written.
2. **Hungarian / auction assignment** on the batch — exactly balanced and
   exactly hard, `O(n³)` but n = 256 is tractable.
3. **Sharper entropic regularization** — ε → 0 recovers a hard plan in the
   limit; sweeping ε would show whether contraction falls away smoothly and
   would connect the two regimes.

**Declared prediction:** a balanced hard assignment reproduces A2's tail and
second moment *with fresh latents*, and closes some of the remaining 1.5× ED²
gap to R11. **Refuted if** it collapses like A3 (the balance was not the
issue) or contracts like A4 (the hardness was not the issue) — in which case
the fixed pairing in A2 is doing something neither captures, and the line
should be abandoned rather than iterated on further.

The same supersession gate applies, and the 3.6× cost ledger must be carried
into any claim.

---

## 4. Scope and caveats

- The protocol was amended once, after a smoke test and before any full run,
  to add the two pre-converged arms; the reason and the fact that no
  threshold changed are recorded in the protocol itself. Both `*P` arms
  behaved like their concurrent counterparts, so the amendment changed
  nothing about the conclusion.
- A2's latent set is fixed, which is the reason it cannot yet be called a
  method (§2).
- Sinkhorn uses one ε and one iteration count, declared and not swept —
  candidate 3 above is exactly the sweep that was not done.
- One dataset, one geometry, one bandwidth, 3 seeds, 600 steps.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  anchor stays disabled; the geometry thread stays closed.

## 5. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase12 `
  --seeds 3 --steps 600
```
