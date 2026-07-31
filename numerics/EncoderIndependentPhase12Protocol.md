# Encoder-Independent Kernel Drifting — Phase 12 protocol

## Replace the self-referential teacher with an external one

*Frozen pre-outcome design. Source: `EncoderIndependentProgramAudit.md` §5.
Results go to `EncoderIndependentPhase12Results.md`.*

---

## 1. Why this phase exists

The audit established that the deficit is caused by **self-reference**, not by
the size or coherence of the teacher's demand:

- rollout K=16 — a large *and* committed displacement — leaves the tail at
  0.0035; 16× batch averaging leaves it at 0.0035 and makes quality worse;
- every failing arm has the form `T = f + Δ`, anchored to the generator's own
  output, and rollout does not change that;
- the two arms whose target never references `f` reach tail **0.0948–0.1590**
  against the moving teacher's 0.0035–0.0049, and better ED² (0.360/0.466
  against 0.815) — measured on fresh latents, so the tail is in the learned
  map rather than in memorized points.

The particle system already produces what the generator cannot: tail
0.26–0.41, second moment ~1.0, ED² 0.07 at the good bandwidth. A **fixed**
particle cloud already beat the moving teacher with an *arbitrary* pairing.
The missing piece is the assignment.

---

## 2. The arms

The particle cloud **evolves under the field alongside training** — it is not
pre-converged, because a converged cloud is not free and the cost must be
paid honestly. All arms in a seed share one particle trajectory, advanced
once per step, so they differ only in what the generator is asked to match.

| arm | target | latents |
|---|---|---|
| A0 | `T = f + ηV` *(incumbent recipe)* | fresh |
| A1 | A0 + R11 *(incumbent reform)* | fresh |
| A2 | evolving particles, **index pairing** | **fixed** |
| A3 | evolving particles, **nearest-neighbour** assignment | fresh |
| A4 | evolving particles, **Sinkhorn** barycentric projection | fresh |
| A4R | A4 + R11 | fresh |
| **A3P** | **pre-converged** particles, nearest-neighbour | fresh |
| **A4P** | **pre-converged** particles, Sinkhorn | fresh |

### Amendment, made after the smoke test and before any full run

The two `*P` arms were **added after a 60-step smoke test**, and the reason is
recorded here rather than folded in silently. With the cloud evolving
*concurrently*, the generator's samples sit far from the particles early in
training, and both assignments degenerate there: nearest-neighbour collapses
many samples onto a few particles, and Sinkhorn's barycentric projection
averages so heavily that it contracts the target (smoke: second moment 0.005).
The assignment is starved exactly when the generator is furthest away.

The audit's winning arms used a **pre-converged** cloud, so the `*P` arms
restore that condition: the particles are advanced for `steps` field steps
first, then held fixed while the generator trains. **The total particle cost
is identical** — the same number of field evaluations, differently ordered —
so the cost ledger is unaffected.

No threshold, gate or metric was changed. The originally declared arms still
run and are still reported; a design flaw found in smoke is fixed before the
run, not tuned to a result.

**A2 uses fixed latents by necessity** — index pairing only defines a
consistent map if the latents are fixed — and it reproduces the audit's
`fixed_particles` arm with an evolving cloud. A3 and A4 use fresh latents and
are the actual proposal: a genuine amortizer must generalize to new latents,
not memorize P points. The latent-sampling difference is a confound between
A2 and A3/A4 and is reported as such, not glossed.

CIFAR-16, target ESS 0.9, 512 particles, generator batch 256, 64 positives,
Adam/2e-3, 3 fresh seeds (`MASTER_SEED + 23000..`), 600 steps.

---

## 3. What is measured

**Primary readouts: the tail trace and ED².** The mechanism claim is about
the tail; the score is the consequence.

**A cost ledger is mandatory.** The particle cloud is extra work: it adds
`512 × (64 + 512)` kernel pairs per step on top of the generator's own field.
The paper's standing claim is one-forward-pass inference (NFE = 1), which
amortization preserves at *inference* but not at *training*. Every arm
reports field evaluations, kernel pairs and wall time, and any claim must
state the training cost it bought.

### 12 gate

> **If an amortization arm without R11 reaches a second-moment ratio inside
> `[0.7, 1.3]` and an ED² within 25% of the best R11 cell, the external
> target supersedes R11.**

The same gate R11 has survived five times (6A, 7A, 8A, 10B, 11).

---

## 4. Declared failure branches

- **A3/A4 clear the gate** → the self-reference account is confirmed and R11
  is superseded by a structural change rather than a correction.
- **A3/A4 behave like A0** → the fixed-target result depends on the target
  being *static*, not *external*. That refutes the audit's reading and sends
  the question back to the dynamics. This is the informative failure.
- **A2 works but A3/A4 do not** → the effect is memorization of a fixed
  latent-to-particle map and does not amortize. Report it that way; it would
  make the whole line a dead end rather than a result.
- **The gate fires but the cost ledger is unfavourable** → report as a
  quality result with an explicit cost, not as an improvement to drifting.
- **No tuning.** Arms and hyperparameters are declared above.

## 5. What Phase 12 cannot conclude

- Nothing about ImageNet, FID, or the paper's trained model.
- The Sinkhorn assignment uses one regularization and one iteration count,
  declared in the code and not swept.
- A positive result is scoped to CIFAR-16, raw pixel geometry, one generator
  family, target ESS 0.9, 3 fresh seeds, 600 steps.
- The anchor stays disabled; the geometry thread stays closed.
