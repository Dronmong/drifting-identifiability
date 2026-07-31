# From Identifiability to Adaptive Drifting

Short presentation draft for an audience with introductory machine-learning
background.

---

## Slide 1 — The project in one sentence

**Question:** if a drifting model says there is no remaining motion, has it
actually learned the target distribution?

We began by proving when the answer is **yes**, then used the proof's lessons
to design better distribution-matching procedures.

Our current proof-of-concept result:

> On a fresh synthetic neural benchmark from 2D to 16D, our confirmed adaptive
> rollout beat both our previous best method and the matched implementation of
> the paper's drifting rule.

---

## Slide 2 — What is a drifting model?

Let:

- \(p\) be the target or data distribution;
- \(q\) be the model's current distribution;
- \(V_{p,q}(x)\) be a **drift field**: an arrow telling a generated point
  \(x\) how to move.

The paper builds these arrows from attraction to target samples and repulsion
from generated samples. Nearby points matter more through the Laplace kernel

\[
k_\tau(x,y)=\exp\!\left(-\frac{\lVert x-y\rVert_2}{\tau}\right).
\]

Here, a **kernel** is simply a similarity rule, and \(\tau\) controls how local
the comparison is.

---

## Slide 3 — Identifiability: does “no drift” mean “correct”?

The easy direction is:

\[
p=q \quad\Longrightarrow\quad V_{p,q}=0.
\]

The important reverse question is:

\[
V_{p,q}=0 \quad\Longrightarrow\quad p=q\;?
\]

This is **identifiability**: the training signal should have only one perfect
solution—the correct distribution.

We machine-checked the full converse for the Laplace kernel on
finite-dimensional Euclidean space:

> For arbitrary probability distributions \(p\) and \(q\), if the ideal
> population drift is zero everywhere, then \(p=q\).

“Population” means the ideal infinite-data field, before minibatch noise,
finite training, or neural-network approximation enters.

---

## Slide 4 — What formalization taught us

The proof guarantees the correct destination, but not that ordinary training
will reach it quickly. It highlighted four practical lessons:

1. **Coverage before polishing.** A local field can move a collapsed cloud as
   one group while failing to send different samples toward different modes.
2. **Mass matters.** A good update must preserve how much probability belongs
   in each region, including small or rare modes.
3. **Persistent information helps.** A single minibatch is a noisy picture of
   a distribution; pooled target statistics give a more stable global guide.
4. **Global and local corrections are complementary.** Global transport finds
   the right regions; the Laplace field can refine local shape.

These are design principles inspired by the theorem, not a claim that the
theorem alone proves neural-training performance.

---

## Slide 5 — First attempt: the 70/30 Quantile-to-Laplace split

Our first algorithm, **QLD**, used two phases:

```text
first 70%: rank-match generated and target samples
final 30%: use the paper's Laplace drift for local refinement
```

A **quantile** is a position in sorted data—for example, the median is the
50th percentile. Rank matching gives different generated samples different
destinations, breaking collapse.

On the sealed 1D test:

- squared energy-distance ratio: **0.9105**;
- sliced-Wasserstein ratio: **0.8949**;
- wins: **23 of 32** target/initialization cases.

A ratio below \(1\) is better. This was a real but modest improvement—roughly
9–11%—and it missed our deliberately strict 20% improvement target.

---

## Slide 6 — The larger jump: KLL–PSQT

We next moved from one sorted line to many 1D views of 2D data.

- **Projection:** view a point cloud from one direction.
- **Sliced transport:** repeat a simple 1D comparison across many directions.
- **PSQT:** Persistent Sliced Quantile Transport; it remembers pooled target
  ranks instead of averaging noisy minibatch quantiles.
- **KLL sketch:** a compact streaming summary that approximately preserves
  quantiles without storing the full data stream.

On 64 fresh 2D targets, KLL–PSQT achieved:

| Comparison | Squared energy distance | Held-out sliced Wasserstein |
|---|---:|---:|
| versus historical PSQT | **0.337×** | **0.519×** |
| versus registered paper arm | **0.076×** | **0.289×** |

This was our first large, broad improvement. However, it directly optimized a
small particle cloud; it was not yet a conventional neural generator.

Suggested visual: the perturbed-ring comparison in
`numerics/psqt_accumulator_confirmatory_runs/20260722-005506-confirmatory/visual_CF-F5-PERTURBED-RING-00.png`.

---

## Slide 7 — Why add a neural network?

A particle method can carefully reposition one small cloud, but it cannot
immediately generate new samples. A practical model needs a function

\[
G_\theta(z)=\text{generated sample},
\]

where \(z\) is fresh random noise.

We therefore use **teacher–student amortization**:

```text
TARGET DATA → reusable target atlas ─────────────────────┐
                                                        ↓
NOISE → current generator → 512 temporary particles → transport rollout
                                                        ↓
                                             local Laplace correction
                                                        ↓
                                               frozen teacher particles
                                                        ↓
                                  neural generator learns the destinations
```

The transported particles are the **teacher**; the neural generator is the
**student**. “Amortization” means doing expensive transport during training so
that later generation requires only the student network.

---

## Slide 8 — Step 1: turn target data into global directions

The flagship first builds a **target atlas** once:

1. choose many viewing directions \(u_1,\ldots,u_L\);
2. project target samples onto each line using
   \(\langle y,u_\ell\rangle\);
3. sort the projected values and store their quantiles.

Intuitively, we photograph the target cloud from many angles and retain where
the 1st percentile, median, 99th percentile, and other ranks should lie.
The confirmed flagship uses an exact stored atlas; KLL is our compressed
alternative.

For a generated particle \(x_j\), we find its rank in every view and look up
the target value \(t_{j\ell}\) at the same rank. Its global correction is

\[
c_j=\frac dL\sum_{\ell=1}^{L}
\left(t_{j\ell}-\langle x_j,u_\ell\rangle\right)u_\ell .
\]

Each term says: “along this viewing direction, move from your current
position to the target position for your rank.” Combining many views gives a
multidimensional movement.

This is **mass-aware**: different generated ranks receive different target
ranks, rather than every particle being pulled toward the same nearby average.

---

## Slide 9 — Step 2: adaptive same-particle rollout

One global move was enough in low dimensions but too short in higher
dimensions. We therefore move and rerank the **same 512 particles**:

\[
x_j^{(r+1)}=x_j^{(r)}+0.5\,c_j^{(r)}.
\]

After each substep, \(c_j^{(r)}\) is recomputed because the particles' ranks
may have changed.

| Dimension | Move–rerank substeps |
|---:|---:|
| 2D | 1 |
| 4D | 1 |
| 8D | 2 |
| 16D | 4 |

This frozen rule depends only on dimension. It does not inspect target labels
or evaluation scores. No additional generator evaluation occurs inside these
substeps: the temporary particles are simply moved in data/feature space.

---

## Slide 10 — Step 3: local refinement and neural learning

Global quantile transport handles coverage and mass allocation. We then add a
smaller version of the paper's Laplace drift to improve local shape:

\[
v_{\text{local}}(x)
\approx
\text{nearby target average}
-
\text{nearby generated average},
\]

where “nearby” is weighted by
\(k_\tau(x,y)=e^{-\lVert x-y\rVert/\tau}\).
Its scale is normalized relative to the global correction.

The final teacher is

\[
y_j=\operatorname{stopgrad}
\left(x_j^{(R)}+0.25\,\widetilde v_{\text{local},j}\right).
\]

**Stop-gradient** means the teacher is treated as a fixed destination. We do
not backpropagate through sorting or transport.

The network learns these destinations with ordinary regression:

\[
\mathcal L_{\text{student}}(\theta)
=\frac1{512}\sum_{j=1}^{512}
\left\|G_\theta(z_j)-y_j\right\|_2^2 .
\]

One training cycle is:

```text
512 noise inputs → build one teacher cloud
                 → shuffle into eight groups of 64
                 → eight Adam updates
                 → refresh the generator particles and teacher
```

The experiment repeats this for 20 cycles. During inference the entire
teacher pipeline disappears:

```text
fresh noise z → trained Gθ → generated sample
```

There is no atlas lookup, sorting, kernel calculation, or rollout at
inference—only one neural-network forward pass.

---

## Slide 11 — Flagship results

Fresh benchmark:

- 32 unseen targets;
- dimensions 2, 4, 8, and 16;
- mixtures, rare modes, heavy tails, and nonlinear shapes;
- both concentrated and broad initializations;
- matched neural-generator evaluation budget.

| Adaptive rollout versus | Squared energy-distance ratio | Held-out sliced-Wasserstein ratio |
|---|---:|---:|
| previous strongest fixed control | **0.857** | **0.924** |
| matched paper implementation | **0.269** | **0.497** |

The model beat the matched paper implementation on **all 32 targets on both
metrics**. It improved every tested target family and preserved the previous
2D/4D outputs exactly while concentrating its extra work in 8D/16D.

Metric intuition:

- **Energy distance** checks overall distribution mismatch.
- **Held-out sliced Wasserstein** compares sorted samples from unseen viewing
  directions, reducing the chance that we merely memorized the training
  projections.

Suggested visual:
`numerics/adaptive_rollout_confirmation_figures/quality_and_cost.png`.

---

## Slide 12 — What the result does and does not say

What we have demonstrated:

- a theorem-backed design story;
- strong, preregistered synthetic results;
- successful neural amortization;
- improvement from 2D through 16D;
- partial improvement in rare-mode placement.

What we have **not** demonstrated:

- superiority to the paper's published ImageNet or FID results;
- image-scale training;
- lower total computation.

The adaptive rollout uses the same number of generator forward calls as the
previous control, but about **1.39×** its measured training time and more
projection/kernel work. It is presently a **quality win**, not a universal
speed win.

---

## Slide 13 — What comes next?

1. Test frozen real encoder features before attempting full images.
2. Improve retention of very rare teacher particles in the neural generator.
3. Replace the exact target atlas with KLL or another compressed atlas where
   memory matters.
4. Learn when to stop rolling out instead of using dimension alone.
5. Finally, test image datasets with the paper's metrics, including FID,
   throughput, and memory.

---

## Closing message

> Formalization showed that zero Laplace drift has the correct unique target.
> Experiments then showed that reaching that target requires global,
> mass-aware assignments, persistent target information, and enough transport
> depth. The confirmed adaptive rollout is our strongest proof of concept so
> far—but the next test is whether that advantage survives real features and
> images.

---

## Repository sources for the presenter

- Formal converse: `DriftingIdentifiability/LaplaceRnRoadmap.md`
- Original 70/30 result: `numerics/QuantileFissionDriftingPlan.md`
- KLL–PSQT result: `numerics/PSQTAccumulatorConfirmatoryResults.md`
- Neural bridge: `numerics/NeuralConditionedTransportConfirmatoryResults.md`
- Flagship confirmation: `numerics/AdaptiveRolloutConfirmationResults.md`
