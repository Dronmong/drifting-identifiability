# Stages B0, B1, B2 — a working encoder-free generator, and the first positive drifting result

*Artifacts: `f3b_confirmatory.json` (sha256 `cd60e98a773fd4c5…`),
`b1_confirmatory.json` (`26770733c98ec346…`),
`stage_b2/b2_confirmatory.json` (freeze `4345e6e22dd95021…`).
All three stages paired at the model initialization and flow-batch level.*

---

## 1. Summary

| stage | question | verdict |
|---|---|---|
| **B0** | can a prescribed bridge reach detected coverage without an encoder? | **PASS 3/3** |
| **B1** | can an encoder-free spectral anchor reduce held-out discrepancy at no cost to coverage? | **PASS 3/3** |
| **B2** | is the drifting field's *zero set* useful as a gradient signal? | **PASS 2/3** |

These are the first positive results in this program. Phases 17–30 and F1
established a chain of negatives; B0 supplies the positive control those
negatives had always lacked.

---

## 2. B0 — detected coverage without an encoder

Frozen `(compact, 30 000 steps, NFE 32)`, selected on a development ladder with
a disjoint development reference, then confirmed on three fresh units.

| unit | recall | precision | KID | FID | rank |
|---|---:|---:|---:|---:|---:|
| 300 | 0.2031 | 0.820 | 0.10188 | 148.47 | 13.49 |
| 301 | 0.1558 | 0.838 | 0.10528 | 151.96 | 12.30 |
| 302 | 0.1538 | 0.803 | 0.09340 | 143.10 | 14.44 |
| *matched real-vs-real* | *0.755* | *0.67–0.71* | *~1e-4* | — | *~8.9* |

All three clear the 0.05 gate by 3–4× against a null exceedance bounded at
0.0149, with every calibrated veto passing.

**This is detectable coverage, not good generation.** Recall 0.15–0.20 is
20–27% of the matched-real ceiling, FID is 143–152, and rank sits above real
data's ~8.9. The artifact's own `claim_scope` records the limit: *"detectable
fresh-sample reachability for this frozen bridge; not general flow-matching or
harness validity."*

---

## 3. B1 — an encoder-free anchor, at no cost to coverage

λ calibrated to **0.931013** so that
`λ·‖∇L_anchor‖/‖∇L_flow‖ = 0.25` on three outcome-blind units, giving an
average-event ratio of 0.0250 at the 1-in-10 cadence.

| unit | B0 recall | B1 recall | floor | anchor reduction | veto | control |
|---|---:|---:|---:|---|---|---:|
| 300 | 0.1475 | **0.1689** | 0.0975 | ✓ | ✓ | 0.7275 |
| 301 | 0.2061 | 0.1768 | 0.1561 | ✓ | ✓ | 0.7090 |
| 302 | 0.1528 | **0.1548** | 0.1028 | ✓ | ✓ | 0.7173 |

All three units satisfied every condition simultaneously: recall
non-inferiority, ≤0.5× median anchor excess above the matched-real floor,
5-of-6 paired audit wins, collapse and augmentation-aware memorization vetoes,
and a valid metric control.

This is the §19.1 configuration the research memo wanted: a fixed source-space
criterion running **beside** the transport loss as a separately nonnegative
term, demonstrably active on held-out directions, with no pretrained encoder
anywhere in the objective.

**Caveat.** Unit 301 dropped 0.0293, 59% of the declared 0.05 margin. It passes,
but a degradation of that size is not reliably resolvable at three units. The
honest reading is "no degradation detected at this replicate count".

---

## 4. B2 — the drifting field's zero set is useful, and my prediction was wrong

### 4.1 The distinction the stage was built on

F1 established that the drift **map** carries every tested start — including
real data — into a single rank-≈1.7 attractor. B2 separates two objects that
had been conflated throughout the program:

| object | F1 | B2 |
|---|---|---|
| the **flow** `x ← x + ηV̂(x)` | collapses from every start | — |
| the **zero set** `{p : V(p) ≡ 0}` | never tested | tested here |

Penalizing `E‖V‖²` descends toward the zero set through *model parameters*. It
never iterates the map, so F1's collapse dynamics do not apply by construction.

### 4.2 Result

| unit | B0 recall | B2 recall | Δ | non-inferior | drift reduction | veto | passes |
|---|---:|---:|---:|---|---|---|---|
| 300 | 0.1157 | **0.1206** | +0.0049 | ✓ | ✓ | ✓ | ✓ |
| 301 | 0.1851 | 0.1812 | −0.0039 | ✓ | ✓ | ✓ | ✓ |
| 302 | 0.1138 | 0.1133 | −0.0005 | ✓ | **✗** | ✓ | ✗ |

**PASS, 2 of 3.**

### 4.3 The prediction I recorded, and got wrong

Before the run I wrote: *"given F1, recall degradation is the more likely
outcome. Recording that prediction now so a FAIL is not retrofitted as expected
and a PASS is not overstated."*

**Recall did not degrade in any unit.** Deltas of +0.0049, −0.0039, −0.0005 are
an order of magnitude inside the 0.025 margin. Unit 302 failed the
*drift-reduction* gate, not the coverage gate. I expected the drift term to drag
the model toward F1's rank-1.7 region; it did not, in any unit.

So the flow/zero-set distinction is not merely a formal escape — it is
empirically load-bearing. The field whose iteration destroys every distribution
is, as a gradient signal, harmless to coverage and in 2 of 3 units reduces its
own held-out discrepancy by ≥50%.

**This is the first positive result for the drifting mechanism in this program,
and it required a working generator to become visible at all.**

---

## 5. A bandwidth consistency check

B2's corrected calibration (self-excluded ESS, target 0.60) returns
**τ = 7.085**. Phase 25's legacy calibration (nominally 0.05, diagonal-included)
returned **τ = 7.714**. Measured on CIFAR-32 raw pixels, median pairwise
distance 37.49:

| calibration | τ | τ/median | ESS diag-included | ESS self-excluded |
|---|---:|---:|---:|---:|
| B2 corrected | 7.085 | 0.189 | 0.0458 | **0.5593** |
| Phase-25 legacy | 7.714 | 0.206 | 0.0803 | **0.6025** |

The two land within 9% on τ and at self-excluded ESS ≈ 0.56–0.60.

This **rehabilitates Phases 7–30 rather than impugning them**. The Phase-25
mislabelling changed the *description*, not the operating regime — and Phase 22
independently measured the empirical optimum at realized ESS 0.52–0.71. Three
routes agree. τ/median ≈ 0.19–0.21 is the dataset-level anchor any future
geometry must reproduce to sit in the same regime.

*(Corroboration, not exact re-derivation: B2 uses exact Laplace over raw pixels,
Phases 7–30 used `smooth_laplace` through the block-kernel machinery.)*

---

## 6. Limits that attach to all three stages

**The capacity confound is open.** B0/B1/B2 use a 3.89M-parameter
`TimeConditionedUNet`; drifting used `OneStepGenerator` at 147k–1.1M. So
"drifting fails, the bridge works" changes objective, architecture *and*
capacity together. Phase 30 tested capacity to 1.1M under drifting and got ~0,
but nothing has tested the bridge at drifting's scale or drifting at the
bridge's. **"The objective was the obstruction" remains unestablished.**

**B2's instrument is not CIFAR.** B1 consumed 9 728 of 10 000 CIFAR test images,
so B2's confirmation uses CINIC-10 ImageNet-only (6 000 images, CC BY 4.0,
archive hash recorded). Distribution shift is real — B0 reads recall 0.114–0.185
there against 0.148–0.206 on CIFAR test — and B2's numbers are not directly
comparable to B0/B1's. This was judged preferable to adaptive reuse of images
B1 had already consumed.

**B2 may not claim to beat B1.** They optimize different discrepancies; B1 is a
report-only incumbent and B2's causal comparison is against B0 only.

**B2's weight is a light touch.** λ = 0.000193, four orders below B1's 0.931.
"Stronger at a larger weight" is untested, and unit 302's failure to reduce
excess is consistent with that.

**None of this is identifiability.** B1's finite random-feature V-statistic and
B2's finite minibatch energy are stochastic surrogates; zero finite loss does
not identify a law. The population statements are the authority, and they are
not what was measured.

---

## 7. Where this leaves the program

The encoder-independence result of Phases 17–18 — a pretrained encoder is
actively harmful as a drifting kernel, and pretraining rather than architecture
is the cause — always carried the caveat that it described a method that did not
work. **That caveat is now discharged in one direction:** there is an
encoder-free generator in this harness that produces detected coverage, and two
encoder-free correctness terms that operate inside it without costing coverage.

What remains untested is B3 (drifting under matched reporting) and B4 (an actual
implementation of the paper's method), which are what would let the encoder
ladder finally be run inside a system that works.
