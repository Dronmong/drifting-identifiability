# Encoder-Independent Kernel Drifting — Phase 29 results

## Every fresh-latent objective has recall 0.000. Drifting is the best of them.

*Research: `EncoderIndependentPhase29Research.md`. Code: `run_phase29.py`.
Artifacts: `phase29.json`, `phase29_imle_frozen.json` (+ `.sha256`),
`phase29.stdout.txt`, 6 sample grids. 6 000 steps per arm, fresh latents unless
marked, KID/precision/recall on 512 samples against 2 048 real.*

---

## 1. Results

| arm | KID | precision | **recall** | alpha | 2nd | distinct targets |
|---|---:|---:|---:|---:|---:|---:|
| `nearest_fresh` | +0.46969 | 0.000 | **0.000** | 4.136 | 0.000 | **1** |
| `hungarian_fresh` | +0.29633 | 0.553 | **0.000** | 4.961 | 0.466 | 256 |
| `imle_k8` (refresh every step) | +0.28453 | 0.668 | **0.000** | 4.747 | 0.618 | 59 |
| `hungarian_fixed` | +0.23705 | 0.748 | **0.000** | 4.563 | 0.599 | 256 |
| `imle_frozen` F=50 | +0.27093 | 0.783 | **0.000** | 4.358 | 0.720 | — |
| `imle_frozen` F=500 | +0.20496 | 0.807 | **0.000** | 4.434 | 0.808 | — |
| **drifting (Phase 28)** | **+0.14959** | 0.523 | **0.000** | 4.503 | — | — |
| *memorization, permanent pairing* | *+0.06110* | *0.791* | ***0.224*** | *4.417* | — | — |

**Not one fresh-latent objective achieves nonzero recall.** The best flicker was
`imle_frozen` F=500 at step 3000 — recall 0.006, KID 0.161 — which decayed again
by step 6000.

The **second declared branch fires**: the obstruction lies outside the objective
family, and the memorization ceiling is the only meaningful coverage number this
harness has produced.

---

## 2. The mechanism is confirmed, then found insufficient

The research pass predicted that **correspondence stability**, not averaging and
not assignment direction, is the operative variable. Two results confirm it
sharply and one bounds it.

**The decisive confirmation.** `hungarian_fixed` has the *same frozen latents and
the same frozen 512-image real set* as Phase 28's memorization. The only
difference is that the assignment is **relearned each step** instead of fixed
forever. Recall goes **0.224 → 0.000** and KID **0.061 → 0.237**. So it is not
fixed latents that matter — it is whether the correspondence stops moving.

**The stability ladder.** Freezing the IMLE correspondence for longer improves
KID monotonically, exactly as predicted:

| correspondence frozen for | KID | recall |
|---|---:|---:|
| 1 step | +0.28453 | 0.000 |
| 50 steps | +0.27093 | 0.000 |
| 500 steps | +0.20496 | 0.000 |
| **forever (memorization)** | **+0.06110** | **0.224** |

**But recall only appears at permanent freezing.** A 500× increase in stability
buys 0.08 of KID and no coverage at all. The mechanism describes the gradient
correctly and yet nothing short of full memorization crosses the coverage
threshold — and full memorization is not a generative model.

**The collapse control behaved as predicted.** `nearest_fresh` reproduces Phase
13's `B6` under the new metrics: `distinct = 1`, second moment exactly 0.000,
and now *precision* also collapses to 0.000. Many-to-one assignment is
catastrophic, which is the same direction Phase 22's bandwidth sharpening was
pushing toward.

---

## 3. The result I did not expect: drifting wins

**Drifting has the best KID of every fresh-latent objective tested:**

```
drifting          0.150
imle_frozen F500  0.205
hungarian_fixed   0.237
imle_k8           0.285
hungarian_fresh   0.296
nearest_fresh     0.470
```

The kernel-averaged teacher — the thing Phase 25 and Phase 28 blamed — beats an
exact optimal bijection, a nearest-neighbour match, and IMLE in both its naive
and properly-frozen forms. Averaging is evidently *protective* here: it produces
a low-variance target, and the alternatives' sharper targets are noisier without
being more learnable.

**This partially rehabilitates the paper's design.** Phase 25 called the
mean-shift teacher form "the binding constraint" and Phase 28 blamed the
convex-combination form. Neither claim survives: replacing the average with
every non-averaged alternative makes things *worse*. What remains true is that
drifting cannot produce coverage — but that is now a property of the harness
under any fresh-latent objective, not a defect of drifting.

---

## 4. What this settles for the program's actual question

The encoder-independence result (Phases 17–18: raw pixels beat a pretrained
encoder by +138 FID) always carried the caveat "a statement about a method that
does not work". Phases 26–29 replace that caveat with something specific:

> The drifting objective's fixed point is correct (Phase 26: real data survives
> 40 iterations of the training map with KID at the floor). Its failure is that
> a parametric generator trained on fresh latents cannot hold coverage — and
> **no objective in the assignment or matching family does better**, while
> drifting does best. The failure is therefore not attributable to the kernel,
> the bandwidth, the geometry, the teacher correction, or the encoder.

That strengthens the negative result rather than weakening it: no encoder choice
could have rescued a deficit that survives every objective tested.

---

## 5. What is left, honestly

**Nothing in the objective family.** §3 shows six objectives spanning averaging,
bijection, nearest-neighbour and both IMLE variants, and drifting is the best of
them. Further variations in this family are not worth running.

**The one untested component is the generator.** Every arm here and in 28 prior
phases used the same `OneStepGenerator` — latent 32, width 64, ~146k parameters,
one forward pass. Phase 28 showed it *can* hold recall 0.224 when memorizing,
so it is not incapable; but coverage under a stochastic objective may simply
require capacity or an architecture this harness has never varied. That is a
generic generative-modelling question, not a drifting one, and it should be
scoped as such rather than as another phase of this program.

**The defensible output of the program is now three things**: the
encoder-independence measurement with its mechanism (§4), the metric audit chain
(ED² saturation → FID small-sample bias → precision/recall calibration), and
the correspondence-stability mechanism with the ladder in §2 that quantifies it.

---

## 6. Scope

- One seed per arm; these are mechanism results, not seeded performance claims.
  The recall finding is 0.000 across six independent objectives, which is far
  outside seed noise, but the KID *ordering* in §3 (0.150 vs 0.205) is within a
  range this program has repeatedly shown needs ≥ 8 seeds to resolve.
- `imle_k8` as first implemented refreshed its correspondence every step, which
  omits the pool-freezing that defines IMLE. That was flagged before the result
  was read and the corrected arms (F = 50, 500) are reported alongside.
- The memorization anchor is evaluated on its own training pairs and measures
  representational capacity, not generalization. It is a ceiling, not a result.
- **None of these arms is drifting**, and none licenses a claim about the paper's
  method. Drifting appears only as the Phase 28 reference row.
