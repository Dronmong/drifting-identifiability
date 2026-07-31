# Encoder-Independent Kernel Drifting — Phase 2 results (CIFAR-10)

*Executes `EncoderIndependentPhase2Protocol.md`, frozen before the run.
Sealed artifacts: `phase2_entry.json` (entry gate) and `phase2_screen.json`
(screen, 9.8 min, 3 seeds × 5 arms), each with a `.sha256` sidecar.*

---

> **Partly superseded by `EncoderIndependentPhase2Diagnosis.md`.** The
> geometry verdict below stands and has since been reconfirmed twice
> independently. The *skyline gap* claim does not: it was a variance
> contraction in the stop-gradient regression, not a property of drifting.
> With that corrected (reform R11) encoder-free raw-pixel drifting reaches
> 1.92 against the skyline's 1.84 — parity. Every statement below of the form
> "drifting itself is the weak link" is withdrawn; see the diagnosis §6 for
> the itemized corrections.

## Verdict

**Phase 2A entry gate: PASS (all four conditions).**
**Phase 2B exit gate: FAIL (three of six conditions).**

Fixed compositional geometry is **37% worse** than raw pixel drifting on real
images, losing all three paired seeds — in the one regime this program spent
two phases searching for, where its neighbour ranking is measurably 57%
better than pixels.

This is the **third consecutive negative** for fixed compositional geometry.
Per the protocol's declared failure branch (§7), it is reported as such and
**not retuned**.

---

## Phase 2A — the entry gate passed

For the first time, the testbed satisfied every precondition before an arm
was run.

| ID | Condition | Result | Verdict |
|---|---|---|---|
| **E1** | testbed solvable | skyline precision **1.000** at 300 steps (bar: 0.473) | **PASS** |
| **E2** | pixel geometry is the bottleneck | pixel k-NN .245, scattering **.383** — ratio **1.567** (bar: 1.25) | **PASS** |
| **E3** | fields can reach the target | residual/floor 0.42–0.49, all ≤ 2.0 | **PASS** |
| **E4** | no kernel collapsed | collapsed-row fraction 0 everywhere | **PASS** |

Zero-set reachability on CIFAR (reform R5), median over 3 seeds:

| configuration | residual / floor | descended | verdict |
|---|---:|---:|---|
| `raw::standard` | 0.49 | 63.0% | reaches |
| `wavelet::standard` | 0.43 | 74.1% | reaches |
| `scattering::standard` | 0.42 | 71.1% | reaches |

Every admitted arm's field genuinely reaches the `q = p` floor — so the
Phase-1 failure mode (a field whose zero-set does not contain the target) is
**excluded by measurement** here. The budget was derived from the skyline:
300 steps clears, so 600 was used for headroom.

---

## Phase 2B — the screen

Median over 3 seeds. Score is `normalized_geometry_score_v2`; lower is better;
the achievable floor is ≈1.2–1.7 (reform R8).

| arm | geometry | score v2 | ED² | precision | coverage | wall s | kernel pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| **B0** | raw pixel | **7.491** | 1.877 | 1.000 | 0.965 | 18.0 | 5.0e6 |
| B1 | wavelet | 10.115 | 3.321 | 1.000 | 0.926 | 43.9 | 2.5e8 |
| B2 | scattering | 10.775 | 3.762 | 1.000 | 0.910 | 68.6 | 4.1e8 |
| B3 | wavelet + anchor | 9.296 | 2.880 | 1.000 | 0.936 | 45.8 | 2.5e8 |
| *SKY* | *sliced Wasserstein* | *1.837* | *0.182* | *1.000* | *0.969* | — | *0* |

### Gate

| ID | Condition | Result | Verdict |
|---|---|---|---|
| **P2.1** | a fixed-geometry arm beats raw pixels | B1/B0 = **1.366** [1.203, 1.570], **0/3** wins | **FAIL** |
| **P2.2** | majority of components | wins 1 of 4 (`nearest_real` only) | **FAIL** |
| **P2.3** | every seed | 1.570 / 1.203 / 1.350 — all > 1 | **FAIL** |
| **P2.4** | the anchor is not destructive | B3/B1 = **0.906** [0.873, 0.930], **3/3** wins | **PASS** |
| **P2.5** | the objective is not vacuous | loss descends 97.4–98.5% in every arm | **PASS** |
| **P2.6** | within reach of the skyline | winner is **5.5×** the skyline | **FAIL** |

---

## What the instrumentation shows

The reforms earn their keep here: this is the first negative in the program
that comes with a mechanism rather than a shrug.

### 1. Fixed geometry trades distribution match for per-sample realism

Per-component normalized ratios (median; lower better):

| arm | ed2 | sw1 | patch_ed2 | **nearest_real** |
|---|---:|---:|---:|---:|
| B0 raw | **29.65** | **5.49** | **36.61** | 0.528 |
| B1 wavelet | 53.59 | 7.44 | 57.83 | **0.500** |
| B2 scattering | 60.69 | 7.87 | 62.87 | **0.490** |
| B3 wavelet+anchor | 45.83 | 6.97 | 47.65 | 0.526 |
| *SKY* | *2.87* | *1.65* | *2.70* | *0.998* |

The structured arms win **`nearest_real`** — their samples sit measurably
*closer to real images* (0.490–0.500 against raw's 0.528) — and lose every
distributional component by a wide margin, with coverage falling in step
(0.965 → 0.926 → 0.910).

So the fixed kernel does exactly what its better neighbour ranking predicts:
it pulls samples toward high-density regions of the data. That produces more
individually-plausible images and a **worse model of the distribution**. A
46–57% better k-NN ranking converts into density-seeking, not into
distribution matching.

This is only visible because reform R4 replaced the saturating `off_support`
with the graded `nearest_real`; under the Phase-1 metric all four arms would
have read as uniformly bad.

### 2. This is not an optimization failure

Every arm drives its unnormalized geometry loss down **97.4–98.5%** (P2.5).
Under Phase-1 instrumentation that loss was pinned at η² and could not have
been checked at all. The objectives are being solved well; solving them is
what produces the worse model.

Combined with E3 — every field provably reaches the `q = p` floor — the two
explanations that survived Phase 1 are both **excluded by measurement** here.
The remaining explanation is that the fixed-geometry drifting objective is
optimized correctly and is the wrong objective.

### 3. Drifting itself is the weak link, not the geometry — **WITHDRAWN**

*The section below is retained as the audit trail of what was concluded at
the time. `EncoderIndependentPhase2Diagnosis.md` shows the gap was a
generator-side variance collapse (effective dimension 2.34 against the data's
8.32); the same field on free particles scores 1.89 against the skyline's
1.84, and a one-scalar teacher correction brings the generator to 1.92. The
gap is an implementation defect, not a property of drifting.*

The skyline scores **1.837** against the best arm's 7.491 — a 4× gap, and on
ED² alone 2.87 against 29.65, an order of magnitude. A plain
sliced-Wasserstein objective on the *same generator, same budget, same data*
is dramatically better than drifting with any kernel tested.

That reframes the whole program. The question "which kernel geometry should
drifting use?" is being asked inside a regime where the drifting objective
costs an order of magnitude in distribution match against a simpler
alternative. **No choice of geometry recovers that.**

### 4. The anchor result replicates

| phase | comparison | ratio | wins |
|---|---|---:|---:|
| Phase 1 (synthetic) | A5/A4 | 0.701 | 24/27 |
| Phase 1 (synthetic) | A6/A4 | 0.658 | 26/27 |
| **Phase 2 (CIFAR)** | **B3/B1** | **0.906** | **3/3** |

The spectral anchor improves a fixed-geometry arm on real images, with a
bootstrap interval excluding 1 and a clean 3/3 sweep. It has now helped in
every configuration it has ever been placed in, across synthetic and real
data, two arm sets, and two direction rules — while costing no measurable
wall-clock. It also nudges `nearest_real` back toward raw's level
(0.500 → 0.526), consistent with the Phase-1 diagnosis finding that it pulls
the generator back toward the data manifold.

### 5. Cost

B1 buys **49× the kernel pairs** and 2.4× the wall clock of raw pixel
drifting for a 37% worse result. B2 buys 81× for 44% worse. The plan's stop
condition on cost without benefit (§10.4) is met, again.

---

## What this establishes

**Survives:**
- The Phase-2A entry gate as a working instrument: it admitted a testbed that
  was solvable, discriminating, and free of the Phase-1 failure mode, and all
  three admissions held up.
- The negative: fixed compositional geometry does not improve drifting on
  CIFAR-16, at 37–44% worse, on every seed, on three of four components.
- The mechanism: better neighbour ranking → density-seeking → worse
  distribution match, with coverage as the visible cost.
- The anchor's positive result, now replicated on real data.
- The skyline gap as the dominant fact about drifting at this scale.

**Not established:**
- Nothing about ImageNet, FID, 32×32 or above, or the paper's trained model —
  which uses a pretrained encoder and a protocol this does not match.
- Nothing about pretrained encoders: none was loaded. The audit's supervised
  control was a small CNN on 2048 images.
- The skyline comparison is a *distribution-matching* comparison at one
  budget on one generator; it is not a claim that sliced Wasserstein is a
  better generative method in general.
- Three seeds, one resolution, one generator, one budget.

---

## Declared consequence

The protocol's failure branch (§7) applies verbatim: this is the third
consecutive negative for fixed compositional geometry and it is **not
retuned**. The geometry thread of this program is closed.

The two threads the second-pass audit identified as alternatives are now the
recommendation, and both are strengthened by this result:

**(a) The anchor.** It is the only component that has passed every condition
ever asked of it, across four phases, and it costs essentially nothing. The
question — *can a spectral anchor make an arbitrary drifting method verifiably
source-correct at negligible cost?* — is a correctness contribution
independent of geometry, and connects directly to
`DriftingIdentifiability/FeatureSpaceIdentifiability.lean`.

**(b) Zero-set reachability.** G0.5 predicted the Phase-1 ranking before any
generator was trained, and in Phase 2 it correctly certified that the fields
were sound — leaving the objective as the culprit, which is what the results
show. It is a cheap, general, predictive diagnostic for kernel-based
generative objectives.

A third possibility is now on the table that was not before: the skyline gap
suggests the interesting question may not be *which kernel* drifting should
use, but *whether the drifting objective is competitive at all* against
simpler distribution-matching losses at small scale. That is a question about
the paper's method rather than about encoder independence, and it would need
its own protocol.

## Reproduce

```powershell
uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase2_entry

uv run --python 3.12 --with torch==2.7.1 --with torchvision --with numpy `
  --with scipy python -m numerics.encoder_independent_drifting.run_phase2_screen
```

CIFAR-10 must be present under `~/.cache/cifar`; both runners refuse to guess
if it is not. 98 unit tests cover the package, including the disjointness of
the CIFAR train/eval splits.
