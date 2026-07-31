# Encoder-Independent Kernel Drifting — formal program audit

## What is established, what is refuted, and the one structural change the evidence points to

*Formal pass over eleven phases and their diagnostic passes, plus three new
measurements. Code: `diagnose_phase12.py`. Artifact: `phase12_probe.json`
(+ `.sha256`). 3 seeds, 600 steps, CIFAR-16, target ESS 0.9. Gate outcomes
below were re-read from the sealed artifacts, not from memory.*

---

## 1. Established

| claim | evidence | status |
|---|---|---|
| **R11** (scalar second-moment teacher match) improves quality | Phase 3 (18/18 paired), Phase 5 (9/9), and **five supersession gates survived** — 6A optimizer, 7A kernel, 8A capacity, 10B shape, 11 metric | strongest result in the program |
| the bandwidth optimum is target-ESS ≈ 0.9 | Phase 7C, U-shaped, worth **4.9×** on free-particle ED²; the program used ESS 0.5 for six phases | established, unvalidated as a *rule* |
| the generator's deficit is invariant to nine axes | optimizer, lr, η, bandwidth, cloud size, latent dim, capacity (36×), teacher-fitting (2.3×), budget (10×) | established |
| free particles reach ~1.0, the generator ~0.42 | 6C, 7B, and every phase since | established |
| the **spectral tail** sets the field's equilibrium radius | Phase 10: 9× over packing; predicts 6/7 arms to <0.11; free particles out-of-sample to **0.015** | established, with a known limit (§4) |
| the generator's tail is **destroyed**, not absent | starts 0.221 (data: 0.164), collapses ~29× in 100 steps | established |
| **only a non-self-referential target restores it** | this pass, §3 | new |

**Verified gate ledger** (re-read from artifacts, all `passed=false`):
`phase6` ×2, `phase7`, `phase8`, `phase10`, `phase11`. R11 has never been
superseded by anything tried.

---

## 2. Refuted

Thirteen mechanism hypotheses, each with the measurement that killed it:

| # | hypothesis | killed by |
|---|---|---|
| 1 | minibatch-noise regression attenuation | noise fraction ≈0.001 — *but see §4, that measurement was taken at the wrong point* |
| 2 | anisotropic teacher contraction | isotropic to 4 s.f. |
| 3 | per-application trajectory contraction | 0.997 on the real trajectory |
| 4 | anisotropic tangent-space realization | no consistent direction |
| 5 | under-fitting the teacher | fixing it overshoots; 8B: 2.3× better fit, identical cloud |
| 6 | RMS-normalization limit cycle | P1/P2 both flat |
| 7 | mis-set optimizer | 6A: 0/12 cells, 60× lr range |
| 8 | mis-set bandwidth (for the generator) | 7A: 0/12 cells, ESS 0.82–0.99 |
| 9 | least-squares shrinkage / capacity | 8A: 36× params, second moment flat (between-width sd 0.026 vs within-seed 0.086) |
| 10 | fluctuation maintains spread | 64× variance reduction moves particles 1.012→1.006 |
| 11 | target noise | 16× averaging = 10× longer run, same fixed point |
| 12 | field bias / the self-term | 15% inward, stable across 32× batch; and a CIFAR control showed self-term dominance was an artifact of *my isotropic reduction* |
| 13 | **spectral bias of the loss metric** | Phase 11: **137× metric conditioning, no effect on the tail** |
| 14 | **the tail demand is too small / incoherent** | **this pass** — rollout K=16 and 16× averaging both fail (§3) |

---

## 3. This pass — three measurements, and my last conclusion corrected

### S1 — the coherence claim is true descriptively

At the generator's cloud, the field's demand resolved into the data's leading
and trailing directions, across two disjoint positive batches:

| cloud | bulk coherence | tail coherence | ratio |
|---|---:|---:|---:|
| **generator** | 0.3750 | 0.0658 | **5.7×** |
| particles | 0.0724 | 0.0678 | 1.1× |
| real data | 0.2553 | 0.2288 | 1.1× |

The generator's bulk demand is reproducible across batches; its tail demand
is much less so. (Particles and real data are converged, so their field is
small and both parts are noise — the generator's is the informative cell.)

### S3 — but coherence is not the *cause*, and neither is smallness

If the tail demand fails because `ηV` is too small or too incoherent, then
enlarging it (rollout: K committed field steps) or averaging it (M batches at
fixed displacement) should help. **Neither does:**

| arm | tail end | 2nd moment | ED² | score |
|---|---:|---:|---:|---:|
| moving K=1 | 0.0048 | 0.547 | 0.8145 | 3.389 |
| moving K=4 | 0.0036 | 0.455 | 0.8793 | 3.374 |
| moving K=16 | 0.0035 | 0.489 | 0.7738 | 3.036 |
| moving avg=4 | 0.0049 | 0.502 | 0.9034 | 3.451 |
| moving avg=16 | 0.0035 | 0.404 | 1.0487 | 3.692 |
| **fixed particle cloud** | **0.1590** | 0.572 | **0.4658** | **2.230** |
| **fixed data cloud** | **0.0948** | **1.165** | **0.3595** | **1.602** |

Sixteen committed rollout steps — large *and* coherent — leave the tail at
0.0035. Sixteen-batch averaging leaves it at 0.0035 and makes everything
worse. **Hypothesis 14 is refuted, and it was mine from last pass.**

### The distinguishing feature is self-reference

Every failing arm has the form `T = f + Δ`: the target is *anchored to the
generator's own current output*. Rollout does not change that — `T` is still
"where you are, after K steps". The two arms that work are the two whose
target does not reference `f` at all.

> **The self-referential anchor is what preserves the cloud's shape. It is
> not the size of the displacement, nor its coherence — it is that the target
> is always the generator's own cloud plus a perturbation, so whatever shape
> the generator has is what it keeps.**

### A confound in my own Phase-11 measurement — checked, and it survived

Phase 11's discriminating probe measured the tail of the fixed-target arms on
their **training latents** (the same 256 used for fitting) while the drifting
arm used fresh latents. That compares memorized points against a learned map
and would have invalidated the conclusion.

Re-measured here on a **fresh probe for every arm**: fixed targets still
reach tail 0.0948–0.1590 against the moving teacher's 0.0035–0.0049, a
**20–45× gap**. The tail is in the learned map, not in memorized points. The
conclusion survives; the original measurement should not have been made that
way.

---

## 4. Methodological audit

**Five instrument defects**, all found by inspecting a flag that disagreed
with its own numbers:

| # | defect | consequence |
|---|---|---|
| 1 | P2 flag fired on floating-point noise below the 4th decimal | reported as spurious, not as support |
| 2 | 7C tested *monotonicity* on a unimodal relationship | reported "no rule" when the data showed a clear interior optimum |
| 3 | 8A's two-point trend test measured a ratio without checking which side moved | reported "substitutes" when R11 had merely got worse |
| 4 | N6's verdict treated *negative* growth as "still moving" | read as "not converged" when it meant "wobbling" |
| 5 | P4's normalization used the requested `k` rather than the basis rank | misnormalized one arm by 512/768 |

**Eight corrections to my own claims**, each made when a measurement
overturned it: "byte-identical" → identical to reported digits; Phase 4's
"regression costs 2×" → underparameterization artifact; "the deficit belongs
to drifting's dynamics" → bandwidth artifact; H9 → scoped to the particle
flow; "the self-term is the mechanism" → reduction artifact; "the field is
tail-blind" → refuted in the opposite direction; "spectral bias of the
metric" → refuted at 137× conditioning; "the demand is too small and
incoherent" → refuted here.

**One claim I made last turn that was simply wrong**: that Phase 2 carried
prior rollout *results*. The rollout probe exists in `diagnose_phase2.py` but
**was never executed** — the artifact contains only `F1_paper_field` and no
stdout records it. The machinery existed; the evidence did not. Rollout was
run for the first time in this pass, and it fails.

**One refutation that should no longer be cited**: hypothesis 1's dismissal
rested on a noise fraction of ≈0.001 measured at *initialization*, where the
signal is large. Along the trajectory the ratio inverts to 4.459 by step 599.
The hypothesis still fails on direct test (P5), but not for the reason
originally given.

---

## 5. The proposal

The evidence now points at one structural change, and it is not another
hyperparameter.

**Replace the self-referential teacher with an external one: have the
generator chase an evolving particle cloud.**

The particle system already produces what the generator cannot — tail
0.26–0.41 and second moment ~1.0 at the good bandwidth, with ED² 0.07 against
the generator's 0.42. Regressing onto a *fixed* particle cloud already beats
the moving teacher on every metric here (ED² 0.466 vs 0.814, score 2.230 vs
3.389) **with an arbitrary pairing between latents and particles** — no
assignment at all.

The missing piece is the assignment. A proper scheme maintains particles
under the field and matches generator samples to them by transport
(nearest-neighbour, or Sinkhorn on the batch), so the target is external,
persistent, and correctly paired.

**This is transport-based amortization, and this repository already has a
track on it** — `numerics/coherent_transport/`,
`AnchoredCoherentTransportResearchPlan.md`, and the
`ConditionedTransportAmortization*` documents — developed independently of
this program. The two lines meet here for the first time, on evidence rather
than on analogy.

**Proposed Phase 12:**

| arm | target |
|---|---|
| A0 | `T = f + ηV` *(the incumbent recipe)* |
| A1 | A0 + R11 *(the incumbent reform)* |
| A2 | evolving particle cloud, **arbitrary** pairing |
| A3 | evolving particle cloud, **nearest-neighbour** assignment |
| A4 | evolving particle cloud, **Sinkhorn** assignment |

Primary readouts: the tail trace and ED². The same supersession gate R11 has
survived five times, plus a cost ledger — the particle cloud is extra work
and must be paid for honestly against the paper's NFE-1 claim.

**Declared prediction:** A3/A4 reach tail ≳ 0.1 and second moment in band
*without* R11, because they remove the self-referential anchor rather than
overriding its consequences. **Refuted if** the assignment arms behave like
A0 — which would mean the fixed-target result depends on the target being
static, not on it being external, and would send the question back to the
dynamics.

---

## 6. Scope

Every measurement in this program is CIFAR-10 at 16×16, raw pixel geometry,
the paper's Algorithm-2 field, one generator family, 3 seeds, 600 steps
unless stated. Nothing concerns ImageNet, FID, or the paper's trained model.
The geometry thread has been closed since Phase 2 and the anchor has been
disabled since then — it remains the program's only untested source-
correctness mechanism, and folding it into a final configuration alongside
the ESS-0.9 bandwidth and R11 is a deliverable that needs no new mechanism.
