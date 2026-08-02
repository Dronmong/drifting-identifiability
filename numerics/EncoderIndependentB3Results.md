# Stage B3 results — the capacity confound is closed

## At matched parameters, matched instrument and matched estimator, the drifting proxy reaches recall 0.0000 while the bridge reaches 0.17–0.24.

*Protocol: `EncoderIndependentB3Protocol.md`. Artifacts:
`stage_b3/b3_matched_reference.json` (sha256 `820a43deab579a4b…`), unit
artifacts `b3_unit_600/601/602.json`, preflight `b3_preflight.json` (GO).
2 arms × 3 units × 30 000 steps, ~7.09 h per unit, ~21 h total.*

**This is a reference measurement. There is no PASS category.**

---

## 1. The matched table

In-domain instrument (CIFAR-10 test, `in_domain_development_reused`), medians
over three units:

| model | parameters | recall | precision | KID | FID | eff. rank | drift energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B3-native** | 146 691 | **0.0000** | 0.494 | 0.1409 | 193.7 | 4.21 | 17.03 |
| **B3-capacity** | **3 864 003** | **0.0000** | 0.576 | 0.1286 | 190.7 | 4.42 | 17.07 |
| B0 (bridge) | 3 893 443 | 0.2114 | 0.771 | 0.1056 | 150.7 | 13.41 | 20.50 |
| B1 (bridge+anchor) | — | **0.2397** | 0.756 | **0.0831** | **133.2** | 12.15 | 16.13 |
| B2 (bridge+drift) | — | 0.1719 | 0.740 | 0.0982 | 145.4 | 8.21 | **15.30** |

The shifted instrument (CINIC-10 ImageNet-only, hash-bound disjoint) gives the
same ordering: B3-capacity recall 0.0000, KID 0.1376, FID 204.2. **The two
sources are reported separately and were never pooled.**

Per-unit B3-capacity recall: 0.00049, 0.0, 0.0 (in-domain); 0.0, 0.00049, 0.0
(shifted).

---

## 2. What this closes

**B3-capacity has 3 864 003 parameters against the bridge's 3 893 443 — within
1%.** Same reference images, sample counts, estimator, `k`, control groups and
drift-audit roles.

Since B0 I have repeatedly flagged that "the objective was the obstruction"
was **unestablished**, because drifting used `OneStepGenerator` at 147 k–1.1 M
while the bridge used a 3.89 M UNet — so objective, architecture and capacity
all changed together. That is now resolved:

> **A 26× parameter increase changes drifting's recall by nothing (0.0000 →
> 0.0000), while the bridge at the same scale reaches 0.17–0.24. Capacity was
> never the limitation.**

`B3-capacity` ≈ `B3-native` on every axis, exactly as Phase 30 predicted from
its ladder to 1.1 M. The extra 3.7 M parameters bought ~0.08 precision and
~0.012 KID, and no coverage.

---

## 3. Two secondary findings

**Effective rank sits between the two regimes.** Drifting arms measure 4.21–4.42
against the bridge's 8.21–13.41 and F1's free-particle attractor of ≈1.7. A
*trained generator* under the drifting objective compresses geometry
substantially without fully collapsing to the free-particle attractor — the two
are related but not identical, as the protocol anticipated.

**Drifting's own mechanism is better served by the bridge.** On B2's
normalized-Laplace axis at the frozen τ = 7.085388360479058:

| model | drift energy |
|---|---:|
| B0 | 20.50 |
| B3 (both arms) | ~17.05 |
| B1 | 16.13 |
| **B2** | **15.30** |

Drifting does reduce drift energy relative to the plain bridge — but **B1 and
B2 reduce it further**. This is not circular and not paradoxical: B3 trains on
Algorithm 2's bi-softmax field, which is a *different object* from the
normalized Laplace field being measured. Still, the arm trained on a drifting
objective is not the arm that best minimizes the theory-aligned drift energy.

---

## 4. Claims that can now be restated on matched axes

Several comparisons in this repository were made across differing references
and hedged accordingly. They can now be stated properly:

- **"B0's KID beats every drifting arm."** I asserted this (0.096 vs 0.131),
  then correctly withdrew it as indicative because F3B allocated its own
  reference. **Matched: 0.1056 vs 0.1286. The claim holds.**
- **"Drifting reaches recall 0.000 while the bridge reaches 0.15–0.20."**
  Previously cross-instrument. **Matched: 0.0000 vs 0.1719–0.2397. Categorical,
  not indicative.**
- **"Capacity is not the obstruction."** Previously unestablished. **Now
  established within this proxy** (§2).

---

## 5. Limits, recorded from the artifact

1. Three units give **coarse consistency, not high-powered inference**. Small
   differences carry no significance claim.
2. No source was pooled; no intermediate checkpoint selected the result. Only
   step 30 000 is primary — 10 k and 20 k are diagnostic.
3. Cross-architecture intervals resample generated sets **independently**;
   generated-index pairing is invalid between different latent spaces and
   samplers. Only the two B3 arms share evaluation latents and support paired
   intervals.
4. **This is not an evaluation of the complete published paper model.** B3 is
   this repository's R11-corrected, raw-pixel, `smooth_laplace`, one-step
   drifting *proxy*. B4 remains the arm that would test the paper's method.
5. Both instruments are development-scope. The in-domain source was adaptively
   consumed by B1; the shifted source becomes development-scope once B3 uses it.
6. Phase 30 found **very low, not literally always-zero** drifting recall
   (median 0.004; one w128/p256 seed at 0.044). B3's 0.0000 medians do not
   retroactively rewrite those values.

---

## 6. Where this leaves the program

The encoder-independence result of Phases 17–18 — a pretrained encoder is
actively harmful as a drifting kernel, and pretraining rather than architecture
is the cause — carried the standing caveat that it described a method that did
not work. B0/B1/B2 discharged half of that by producing an encoder-free
generator that does work. **B3 discharges the other half**: the failure is
attributable to the drifting objective itself and not to the small generator it
was always run on.

What remains is **B4** — an actual implementation of the paper's method rather
than a proxy — and the deferred **B2.5** factorial, which asks whether B1's
geometry retention and B2's drift reduction are complementary.
