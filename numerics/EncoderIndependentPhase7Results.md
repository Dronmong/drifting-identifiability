# Encoder-Independent Kernel Drifting — Phase 7 results

## Two defects, not one: the field's is a bandwidth artifact, the generator's is not

*Protocol: `EncoderIndependentPhase7Protocol.md`, frozen before the run.
Code: `run_phase7.py`, `train.py`, `config.py`. Artifact: `phase7.json`
(+ `.sha256`), stdout in `phase7.stdout.txt`. 119 unit tests passing.
145.9 minutes, 3 fresh seeds (`MASTER_SEED + 12000..`), CIFAR-10 at 16×16,
600 steps, Adam/2e-3 throughout. Every cell R15-admissible — no exclusions.*

---

## 0. Summary

| stage | question | verdict |
|---|---|---|
| **7A** | is R11 a workaround for a mis-set kernel? | **gate not passed — R11 survives again** |
| **7B** | does the particle/generator ordering invert? | **yes, and the direction depends on bandwidth** |
| **7C** | is there a target-only bandwidth rule? | **an interior optimum exists; no monotone rule** |

The headline is a **decomposition**. The Phase-6 follow-up showed the
free-particle deficit is bandwidth-controlled and hoped the generator's was
the same defect. It is not:

> **The field's deficit and the generator's are two different things.**
> Bandwidth moves the particle fixed point from 0.23 to 0.99 of the data's
> variance. It moves the generator's essentially not at all — 0.34 to 0.38
> across the same axis, never within a factor of two of the band.

---

## 1. 7A — bandwidth does not touch the generator's deficit

Four bandwidths × three field-cloud sizes × R11 × 3 seeds = 72 runs, all
admissible.

| bandwidth | cloud | uncorrected ED² / 2nd | + R11 ED² / 2nd |
|---|---:|---|---|
| ess = 0.50 | 64 | 1.6737 / 0.271 | 0.2441 / 0.995 |
| ess = 0.50 | 512 | 1.0185 / 0.375 | 0.1615 / 1.062 |
| ess = 0.90 | 64 | 1.8985 / 0.235 | **0.1477** / 1.047 |
| ess = 0.90 | 512 | 1.0293 / 0.377 | 0.1675 / 1.057 |
| τ = 1.00 | 64 | 2.2427 / 0.185 | 0.1748 / 1.045 |
| τ = 1.00 | 512 | 1.0750 / 0.347 | 0.1755 / 1.051 |
| τ = 2.00 | 512 | 1.2313 / 0.336 | 0.1826 / 1.038 |

Across **all 36 uncorrected runs** the second moment spans **0.101 – 0.443**;
across all 36 R11 runs, **0.874 – 1.159**. Zero of twelve uncorrected cells
reached the band. The best uncorrected cell (1.0185) misses the supersession
ceiling of 0.1847 by **5.5×**.

**The gate does not fire. R11 survives a second sharp test**, now against the
best-supported alternative explanation the program had.

This is a genuine negative for the hypothesis I raised in the follow-up. I
wrote there that "the deficit R11 corrects is largely a bandwidth artifact"
and that R11 "has never been tested against a properly set kernel". It has
now been, at four bandwidths spanning realized ESS 0.82 → 0.99, and the
answer is that **the generator's deficit is bandwidth-independent.** The
follow-up's inference was reasonable from particle data and wrong about the
generator.

**Cloud size (R27) does matter, but does not close it.** Going 64 → 512 at
fixed bandwidth improves the uncorrected generator consistently (1.67 → 1.02
at ess 0.5; 2.24 → 1.08 at τ = 1) and lifts the second moment 0.27 → 0.38.
Real, worth having, and an order of magnitude short of what R11 does.

---

## 2. 7B — the ordering inverts, and which way depends on the bandwidth

Matched bandwidth, matched cloud (512), same field, same budget:

| bandwidth | particles | generator + R11 | generator ÷ particles |
|---|---:|---:|---:|
| ess = 0.50 | 0.2500 / 0.638 | 0.1615 / 1.062 | **0.65×** *(generator wins)* |
| ess = 0.90 | **0.0749** / 0.998 | 0.1675 / 1.057 | **2.24×** *(particles win)* |
| τ = 1.00 | 0.1214 / 1.121 | 0.1755 / 1.051 | 1.45× |
| τ = 2.00 | 0.2138 / 1.259 | 0.1826 / 1.038 | 0.85× |

The corrected generator is remarkably **flat** — 0.16 to 0.18 across the
whole axis — while the particles swing by 3.3× and pass right through it.
So the inversion the follow-up predicted is real, but it is not a property of
the method: it is the crossing point of a flat curve and a U-shaped one.

Read together with 7A this gives the decomposition cleanly:

- **the field has a bandwidth-controlled fixed point.** Set it well and free
  particles reach 0.0749 with a second moment of 0.998 — essentially correct
  on both axes, and better than anything the generator achieves;
- **the generator adds a second, bandwidth-independent contraction** that
  costs it a factor of ~2 at the good bandwidth and which only R11 repairs.

At the incumbent ESS-0.5 setting these two happened to partly cancel, which
is why the generator looked *better* than particles for six phases (0.65×).
That was an artifact of a badly set kernel, not amortization succeeding.

---

## 3. 7C — the optimum is interior, and it is not where the program sits

Free particles, 3 seeds, all admissible:

| arm | target-only ESS | ED² | 2nd moment |
|---|---:|---:|---:|
| ess = 0.5 *(incumbent)* | 0.8124 | 0.2317 | 0.641 |
| ess = 0.7 | 0.8545 | 0.1118 | 0.772 |
| **ess = 0.9** | **0.9299** | **0.0729** | **0.989** |
| ess = 0.95 | 0.9608 | 0.1222 | 1.131 |
| ess = 0.99 | 0.9915 | 0.2326 | 1.267 |
| τ = 1 | 0.9611 | 0.1269 | 1.123 |
| τ = 2 | 0.9890 | 0.2193 | 1.258 |
| τ = 4 | 0.9973 | 0.2448 | 1.147 |
| τ = 8 | 0.9992 | 0.2846 | 1.072 |

**The turn is located.** Quality is **U-shaped** with a clear interior
optimum at ess = 0.9, and degrades in both directions — 3.2× worse at the
incumbent ess = 0.5 and 3.9× worse at τ = 8. The follow-up could not see this
because τ = 1.0 was the edge of its sweep; extending past it settles the
question.

Notably, **the second moment crosses 1.0 almost exactly at the quality
optimum** (0.989 at ess = 0.9). The second-moment ratio is not merely
correlated with quality here — on this axis it is an accurate pointer to the
best setting.

### The rule check reports "no rule", and that is an artifact of my instrument

`_rule_7c` tests whether target-only ESS **monotonically** orders quality
(Spearman ≤ −0.9). It returns +0.733 and "does not order quality on its own".
That is correct as a test of monotonicity and **the wrong test for a
unimodal relationship** — no rank correlation can detect a U.

What the data actually show is stronger than the instrument reports: there is
a **well-localized interior optimum at target-only ESS ≈ 0.93**, bracketed on
both sides by arms that are 1.5–3× worse. That is a usable calibration
target, computed from target data alone. I am recording the instrument's
mis-design rather than quietly re-reading its output, and 7C was declared
"reported, not gated", so nothing rests on the flag.

The honest status: **a candidate rule exists** — calibrate the bandwidth so
the target-only ESS lands near 0.93 — **and it has not been validated**, since
it was read off the same sweep that produced it. Confirming it needs a fresh
target and fresh seeds.

---

## 4. Reform landed

| reform | what | status |
|---|---|---|
| **R27** | `TrainConfig.field_cloud` — the field's cloud size, previously tied silently to the target batch | **implemented**, 1 test; measurable effect, insufficient alone |

A bug the test caught: `config.field_cloud or config.batch` treats an
explicit `0` as "unset" and silently falls back instead of raising. Fixed to
an `is None` check.

---

## 5. What Phase 7 establishes

**Established:**
- the generator's second-moment deficit is **bandwidth-independent** (0.101–
  0.443 across 36 runs spanning realized ESS 0.82–0.99) and is a *different
  defect* from the field's;
- **R11 survives a second sharp test** and remains the program's only
  confirmed positive, now in a fifth independent setting;
- free particles at ess = 0.9 reach ED² 0.0749 with second moment 0.998 —
  **better than any generator configuration measured**, corrected or not;
- bandwidth quality is **U-shaped with an interior optimum**, and the
  program's ESS-0.5 setting is 3.2× off it;
- field-cloud size is a real but insufficient axis (R27).

**Overturned:** the follow-up's hypothesis that R11 compensates for a mis-set
bandwidth. True of the particle deficit, false of the generator's.

**Still unknown:** why the *generator* contracts. That is now a sharply
posed question — it is not the optimizer (6A), not the kernel bandwidth
(7A), not the cloud size (7A), not the field's own fixed point (7B/7C), and
five direct hypotheses about the stop-gradient regression were refuted in
Phases 3–5. `EncoderIndependentMechanismSynthesis.md` proposes a ninth
account (H9) grounded in mean-shift and MMD-flow theory; note that H9
addresses the *flow's* deficit, which 7A has now shown is the one that is
**not** the generator's problem, so H9 needs restating before it is tested.

## 6. Scope

CIFAR-10 at 16×16, raw pixel geometry, paper Algorithm-2 field, one
generator family, 3 fresh seeds, 600 steps, Adam/2e-3. The anchor was not
enabled in any arm and the geometry thread stays closed. The 7C candidate
rule is unvalidated. Nothing here concerns ImageNet, FID, or the paper's
trained model.

## 7. Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase7 `
  --stage all --seeds 3 --steps 600 --resolution 16

uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.tests.run_all
```
