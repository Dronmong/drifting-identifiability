# Encoder-Independent Kernel Drifting — Phase 11 results

## The metric is not the problem. The self-referential teacher is.

*Protocol: `EncoderIndependentPhase11Protocol.md`, frozen before the run.
Code: `run_phase11.py`. Artifact: `phase11.json` (+ `.sha256`), stdout in
`phase11.stdout.txt`. 3 fresh seeds (`MASTER_SEED + 20000..`), CIFAR-16,
target ESS 0.9, field cloud 256, 600 steps.*

---

## 0. Summary

The derived intervention **failed its primary prediction**, and the failure
localizes the cause precisely.

| arm | tail @init | tail @100 | tail end | 2nd moment | ED² |
|---|---:|---:|---:|---:|---:|
| W0 γ=0 *(baseline)* | 0.2035 | 0.0095 | 0.0050 | 0.418 | 0.913 |
| W1 γ=0.5 | 0.2035 | 0.0105 | 0.0034 | 0.352 | 1.202 |
| W2 γ=0.9 | 0.2035 | 0.0090 | 0.0056 | **0.233** | 1.785 |
| W3 γ=0.99 | 0.2035 | 0.0109 | 0.0097 | **0.231** | 1.739 |
| W2R γ=0.9 + R11 | 0.2035 | 0.0106 | 0.0308 | 0.966 | 0.138 |
| **E1 R11** | 0.2035 | **0.0447** | **0.0665** | **1.009** | **0.131** |

*(real data tail = 0.1416)*

**Prediction 1 fails.** The tail still collapses ~20× in the first 100 steps
in every whitened arm — 0.0090 to 0.0109 against the baseline's 0.0095, a
ratio of 0.94–1.15. Whitening the regression metric does nothing to tail
destruction.

**And it is not for lack of strength.** The realized metric conditioning:

| γ | 0.5 | 0.9 | 0.99 |
|---|---:|---:|---:|
| scale ratio (max/min direction weight) | 13.7× | 39.9× | **137.3×** |

A 137-fold reweighting in favour of the trailing directions changed the tail
by 15%. The declared "tail does not rise" branch required reporting this
before concluding, and it rules out the obvious excuse.

**Whitening also hurts.** At γ ≥ 0.9 the second moment falls to 0.233/0.231
against the baseline's 0.418, and ED² roughly doubles.

### A correction to my own reading

The 120-step smoke showed γ=0.99 *improving* the second moment (0.362 →
0.526) and ED² (0.999 → 0.629), and I reported that as directionally
encouraging. **At 3 seeds it reverses** — 0.418 → 0.231, ED² 0.913 → 1.739.
The smoke was a one-seed artifact and the intervention is harmful, not
helpful.

---

## 1. What the failure localizes

R11 is the only arm that preserves tail — **0.0665 against the baseline's
0.0050, a factor of 13** — which is consistent with everything measured
before but does not say *why* the baseline loses it.

A discriminating measurement, same architecture, same optimizer, same
ordinary `‖·‖²` loss, differing only in **what the target is**:

| target | tail @0 | @150 | @300 | @450 | @600 |
|---|---:|---:|---:|---:|---:|
| **drifting** (`T = f + ηV`, recomputed each step) | 0.216 | 0.0069 | 0.0049 | 0.0048 | **0.0037** |
| a **fixed** particle cloud (tail 0.275) | 0.217 | 0.0203 | 0.0406 | 0.0691 | **0.0960** |
| a **fixed** real-data cloud | 0.217 | 0.0442 | 0.0633 | 0.0806 | **0.0897** |

Fitting a *fixed* high-tail target, the tail dips and then **grows back to
0.096 and is still rising at step 600**. Under the drifting teacher it
collapses to 0.0037 and stays flat — a **26× difference** with everything
else held identical.

> **The generator is perfectly capable of building and holding a spectral
> tail. It does not do so under the drifting recipe because of what the
> teacher asks for, not because of the architecture, the optimizer, or the
> metric.**

`T = f + ηV` anchors the target to the generator's own current cloud. The
field's contribution is a small per-step nudge — `‖ηV‖ ≈ 0.5` against an
output norm of order 9, about 6% — whose direction is recomputed from a fresh
batch every step. It never accumulates into a persistent demand for a
different *shape*, only a persistent demand for a small displacement. A fixed
target demands the shape every step, and the generator builds it.

---

## 2. What this does to the mechanism chain

The tail-destruction pass proposed:

1. ~~least-squares regression discards tail *because its metric under-weights
   trailing directions*~~ → **refuted here** (137× reweighting, no effect);
2. the field cannot compensate — **stands**;
3. equilibrium tail ≈ 0.004 — **stands**;
4. the Phase-10 law maps tail to scale — **stands**, with the limits already
   recorded;
5. the generator sits at 0.32–0.42 — **stands**.

So link 1 is replaced: the tail is destroyed by the **self-referential,
per-step-recomputed teacher**, not by the loss's direction weighting. Links
2–5 are untouched, and the Phase-10 law is not in question — this phase never
reached the stage that would have tested it (the declared "tail rises but the
second moment does not follow" branch never triggered, because the tail never
rose).

**The gate does not fire.** R11 survives a fifth supersession test.

---

## 3. The next experiment

The measurement says the target's *shape demand must persist*. Two ways to
make it, both cheap and both already partly present in this repository:

**Rollout teachers.** Apply the field `K` times before regressing, so the
target is `f` advanced `K` steps rather than one. The displacement grows with
`K` and commits to a direction rather than being re-randomized every step.
Phase 2's diagnosis already carries a rollout arm and
`AdaptiveRolloutConfirmationProtocol.md` exists in this tree, so the
machinery and the prior results are available for comparison — but neither
was ever evaluated against the *tail*, which is now the readout that matters.

**An EMA / persistent target.** Maintain a slowly-updated target cloud that
the generator chases, so the shape demand is coherent across steps rather
than resampled.

**Primary readout is again the tail trace**, with the second moment as the
consequence. The declared prediction is that tail should rise with `K` (or
with EMA persistence), and — by the Phase-10 law, which this phase leaves
intact — the second moment should follow without any scale intervention.

This is the same gate R11 has now survived five times.

---

## 4. Scope and caveats

- **The discriminating measurement in §1 is one seed.** The effect is large
  (26×) and monotone in the trace, but it must be replicated with the
  declared 3 seeds in the next phase before it carries a conclusion's weight.
  It is reported here as the reason for the next design, not as a result.
- "Spectral bias of least-squares" was my reading and it is now refuted as
  the operative cause; what remains established is only that the tail is
  destroyed and that the target's structure is what governs it.
- Whitening is tested at one ridge floor (1e-3) and one covariance estimator;
  a different regularization might behave differently, though the 137×
  conditioning makes that unlikely to matter.
- One dataset, one geometry, one bandwidth, 3 seeds, 600 steps.
- Nothing here concerns ImageNet, FID, or the paper's trained model. The
  anchor stays disabled; the geometry thread stays closed.

## 5. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase11 `
  --seeds 3 --steps 600
```
