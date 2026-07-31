# Encoder-Independent Kernel Drifting — Phase 27 results

> **§2's alpha framing is CORRECTED by `EncoderIndependentPhase28Results.md`.**
> The measurements below stand. But I read the unrecoverable direction as "loss
> of high-frequency content, measured by alpha", and **alpha does not measure
> image quality**: fixed-pairing regression produces recognizable CIFAR objects
> at alpha **4.417**, and autoencoder reconstructions are recognizable at
> **4.858** — both essentially the drifting output's 4.43–4.49. What the dynamics
> destroy is **coverage**, not sharpness. Phase 28 drove a generator from recall
> 0.230 to 0.000 in 1500 drifting steps with alpha holding at 4.39–4.45
> throughout. §3 below already flagged that this document's auto-verdict wrongly
> called the basin "wide"; Phase 28 confirms the warm start it recommended does
> not survive.

## The basin is wide in every direction except the one the generator actually fails in

*Code: `diagnose_phase27.py`. Artifact: `phase27_probe.json` (+ `.sha256`),
`phase27_probe.stdout.txt`, 18 start grids. 5 000-step generated cloud, 512
particles, 3 interpolation paths × 6 λ × 40 iterations of the training map.*

---

## 1. Results

| path | λ | start KID | start recall | final KID | final recall | final alpha | attractor |
|---|---:|---:|---:|---:|---:|---:|---|
| mixture | 0.0 | +0.00039 | 0.767 | +0.00063 | 0.760 | 3.608 | data |
| mixture | 0.4 | +0.02360 | 0.739 | +0.02352 | 0.736 | 3.702 | data |
| mixture | **0.8** | +0.09443 | 0.777 | +0.09148 | 0.762 | 3.932 | **data** |
| mixture | 1.0 | +0.14695 | 0.000 | +0.14060 | 0.000 | 4.485 | collapsed |
| blend | 0.2 | +0.00260 | 0.714 | +0.00287 | 0.715 | 3.621 | data |
| blend | 0.4 | +0.02235 | 0.576 | **+0.01909** | **0.608** | 3.685 | data |
| blend | **0.6** | +0.08075 | 0.283 | **+0.06596** | **0.335** | 3.841 | **data** |
| blend | 0.8 | +0.15556 | 0.005 | +0.14841 | 0.004 | 4.166 | collapsed |
| **blur** | **0.2** | +0.05483 | 0.288 | +0.05674 | **0.271** | 4.758 | **partial** |
| blur | 0.4 | +0.12084 | 0.004 | +0.11034 | 0.008 | 4.542 | collapsed |
| blur | 1.0 | +0.30701 | 0.000 | +0.27819 | 0.000 | 4.161 | collapsed |

**Basin edge — largest λ still returning to the data attractor:**

| path | edge |
|---|---|
| mixture | **0.8** |
| blend | **0.6** |
| **blur** | **0.0** |

---

## 2. The finding: the recoverable directions are not the failing direction

**Off-manifold perturbation is recoverable.** At blend λ = 0.4 and 0.6 the map
*actively improves* the state — KID falls and **recall rises** (0.576 → 0.608,
0.283 → 0.335). It is pulling a substantially degraded cloud back toward the
data attractor. At mixture λ = 0.8, four fifths of the cloud is generated
garbage and the real fifth is not dragged down (recall 0.777 → 0.762).

**Loss of high-frequency content is not.** A single 3×3 box blur pushes real data
out of the basin, and the map makes it *worse* (recall 0.288 → 0.271).

The matched comparison is what makes this airtight:

| start | KID | recall | after 40 steps |
|---|---:|---:|---|
| **blur λ=0.2** | 0.0548 | 0.288 | KID **worse** (0.0567), recall **falls** to 0.271 |
| **blend λ=0.6** | 0.0808 | 0.283 | KID **better** (0.0660), recall **rises** to 0.335 |

The blended start is **further** from the data in KID and has the *same* recall,
yet the map repairs it while it degrades the blurred one. So this is not about
how far the cloud is — it is about **which direction** it is displaced in.

**And the generator's failure is exactly the unrecoverable direction.** Its
signature is spectrum alpha **4.485** against real data's 3.608, with recall
0.000 — it is blurred, not merely displaced.

Interesting detail that does *not* rescue it: along the blur path the map does
push alpha back down (4.758 → 4.542 → 4.364 → 4.244 → 4.161 as λ rises), so it
is adding high-frequency energy. Recall stays at 0.000 throughout. It restores
high-frequency *content* but not the *right* content — the same
precision-without-coverage failure seen everywhere in this program.

---

## 3. The auto-verdict is too optimistic and should not be quoted

The runner printed *"the good basin is WIDE — a warm start suffices; pretrain the
generator by regression onto real samples, then hand over to drifting"*, because
my declared `basin_is_wide` test only consulted the blend and mixture edges.

That conclusion does not survive the blur row. **The basin is wide along
directions that do not matter and has width ~0 along the one that does.** Worse,
the specific intervention it names is likely to fail for the reason just
measured: regression onto real samples under a squared loss with limited
capacity produces a *blurry* generator, which is precisely the unrecoverable
starting condition.

I should have made the blur path a veto in the verdict logic rather than one
vote among three. The measurements are right; the summary rule was wrong.

---

## 4. Where this leaves the program, and the one test that decides it

Phase 26 established the data distribution is a stable attractor. Phase 27
establishes the map can reach it from off-manifold displacement but not from
blur. That isolates a single question, and it is not about the field at all:

> **Where does the generator's blur come from?** If a `OneStepGenerator` trained
> by direct regression onto real images is itself blurry, the blur is a property
> of the *generator* — capacity, architecture, or the squared loss — and no
> change to the kernel, bandwidth, teacher or geometry can fix it, because the
> field was never the cause. If direct regression produces a sharp generator,
> then the blur is produced by the drifting dynamics and a sharp warm start is
> worth building.

**Next probe (cheap, no long run).** Train the identical generator by MSE
regression onto real CIFAR images — a sharp, in-distribution target, no kernel
involved — and measure alpha, recall, precision and KID. Declared in advance:

- **regression generator is sharp** (alpha ≈ 3.6, recall > 0.3) → the generator
  can represent the data attractor; the blur is dynamical; build the warm start
  and hand over to drifting. This is the first route with a measured basis.
- **regression generator is blurry** (alpha > 4.2, recall ≈ 0) → the blur is the
  generator's own, the field is exonerated, and every phase of this program has
  been tuning the wrong component. The fix would be architectural (a sharper
  decoder, an adversarial or perceptual term) and lies outside the drifting
  question entirely.

Either answer is worth more than another sweep, and both close a question that
has been open since Phase 1.

---

## 5. Scope

- One 5 000-step cloud, one seed, 512 particles, 40 iterations per cell. The
  blur-versus-blend contrast is large and matched (§2), but this is a
  single-configuration probe, not a seeded performance claim.
- Free particles throughout: no generator, no optimizer, no Jacobian. A trained
  generator sees the map through its parameterization and may behave
  differently — §4's probe is the first step toward checking that.
- `mixture` λ=0.8 keeps recall high partly *because* a fifth of the cloud is
  literally real data; the informative part of that row is that the map does not
  destroy those samples, not the absolute recall value.
- Attractor thresholds (recall ≥ 0.30 data, ≤ 0.05 collapsed) were declared
  before running and sit far from both observed values (0.72–0.78 and exactly
  0.000).
- Still not the paper's method: pixel-space drift with feature-space kernel
  weights.
