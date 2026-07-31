# Encoder-Independent Kernel Drifting — the mechanism, ninth attempt

## Drifting is an MMD-flow-type particle system, and it inherits a known pathology

*Research note written while Phase 7 runs. It uses only already-recorded
measurements (Phases 1–6, the Phase-6 follow-up) plus published theory —
**no Phase-7 result informs it**, so the frozen protocol is untouched.
Nothing here is measured; it is a hypothesis with a test design.*

Eight mechanism hypotheses have been proposed and refuted. This is the
ninth, and it is the first that is (a) grounded in published theory about
this exact algorithm class and (b) consistent with **every** measurement the
program has recorded.

> **Post-Phase-7 amendment (added after this note was written).** Phase 7
> established that there are **two** deficits, not one: the field's, which is
> bandwidth-controlled, and the generator's, which is **bandwidth-independent
> and unaffected by everything tried**. H9 as written explains the *field's*
> deficit — the interacting-particle flow — which 7A showed is precisely the
> one that is *not* the generator's problem. So §2's table is still valid for
> the particle results, but the row "R11 works" now belongs to the other
> defect and must be struck from H9's evidence. **H9 is now a hypothesis
> about the particle flow only, and its explanatory value for R11 is
> withdrawn.** P1–P3 in §3 remain worth running as tests of the flow; they
> are no longer candidate explanations of R11.

---

## 1. What Algorithm 2 actually is

The Phase-6 follow-up reduced the field to

```
V_i = m_i (ȳ_i − x̄_i),
   ȳ_i ∝ Σ_j k_ij C_j^{−1/2} y_j     (attention-reweighted mean of the DATA)
   x̄_i ∝ Σ_k k_ik C_k^{−1/2} x_k     (attention-reweighted mean of the CLOUD)
```

That is **attraction toward the data minus repulsion from the cloud itself**
— structurally the witness-function gradient of an MMD, i.e. an interacting
particle system of exactly the type studied as *MMD gradient flow*. `p = q`
is a fixed point only because the two terms cancel.

Two published results bear on this directly.

**(a) Blurring mean-shift provably collapses.** In Gaussian Blurring
Mean-Shift — where the cloud is updated by its own mean-shift, which is what
the negative term is computed from — the dataset's diameter decreases *at
least geometrically* and the algorithm converges to a single all-points-
coincident cluster. It needs a stopping criterion precisely because it does
not have a non-degenerate fixed point.

**(b) MMD gradient flows collapse around modes and get stuck**, and the
standard remedy in that literature is **noise injection** into the gradient,
tuned over iterations — a heuristic introduced because the unmodified flow
demonstrably fails on simple cases.

So each half of Algorithm 2, taken alone, is a known-contracting process.
The algorithm's correctness rests entirely on the two halves cancelling.

---

## 2. The hypothesis

> **H9 — the cancellation-residual account.** The second-moment deficit is
> the residual of an *imperfect cancellation* between two mode-seeking
> drifts. Whenever the cloud-side term is estimated less well than the
> data-side term, the difference leaves a net mode-seeking drift and the
> cloud contracts; in regimes where the cloud-side term is over-estimated
> instead, it leaves a net repulsion and the cloud expands. The sign is
> regime-dependent; the mechanism is the same.

### It is consistent with everything already measured

| observation | phase | H9 explains it |
|---|---|---|
| the deficit exists in **free particles**, no generator | 6C | it is a property of the flow, not the port |
| **more particles → less deficit** (CIFAR .371→.600), and the sign flips in low dimension (64 → 2.59 vs 512 → 1.27) | follow-up M3 | the *cloud-side estimate* is the error source; its bias can go either way |
| **positives barely matter** (512×64 ≈ 512×512) | follow-up M3 | the data side is already well estimated — it is not the bottleneck |
| **bandwidth controls it monotonically** | follow-up M6 | bandwidth sets the relative smoothing of the two KDEs, hence the cancellation error |
| **the self-mask is protective** (.521 vs .303) | 4 | it deletes the single most biased term of the self-estimate |
| **full Sinkhorn balancing detonates the cloud** (1.27 → 248) | follow-up M4 | it applies a full density correction to the repulsion, destroying the balance |
| **R11 works, and is isotropic** | 3, 5, 6 | mode-collapse is an isotropic loss of spread; re-inflation is the right shape of fix |
| **richer direction changes do not beat one scalar** | 6B | the residual is not anisotropic, so resolving directions buys nothing |
| **over-correcting (gain 1.2) is decisively worse** | 6B | there is a correct amount of re-inflation, not "more is better" |

That last block matters. 6B's negative result — that per-coordinate and
eigendirection matching cannot beat a single scalar — was unexplained when
recorded. H9 explains it: the defect is a scalar-shaped loss of spread, so a
scalar is the right-shaped correction and the richer ones only add variance.

### What R11 becomes under H9

**R11 is an ad-hoc, deterministic substitute for noise injection.** The MMD
literature re-inflates a collapsing flow by adding entropy; R11 re-inflates
it by rescaling to a matched second moment. Same target, different mechanism
— which would explain why R11 works while having no derivation, why it is
insensitive to being made richer, and why its "undershoot" is not a deficit
to close.

---

## 3. Phase 8 — the test design

Three predictions, each cheap and each able to fail.

### P1 — noise injection should do R11's job, from theory rather than fiat

Add annealed noise to the particle/teacher update, the literature's own
remedy, with **no** second-moment match. If H9 is right this should move the
second moment toward the band and improve quality on its own.

- arms: noise scale σ ∈ {0, 0.01, 0.03, 0.1} × schedule {constant, annealed}
- crossed with R11 on/off, to see whether they are redundant or additive
- **the informative outcome is redundancy**: if noise injection alone matches
  R11, R11 is explained; if they add, H9 is incomplete.

### P2 — decouple the negative cloud from the probe cloud

If the cloud-side estimate is the error source, then estimating it better
should shrink the residual **without touching the probe batch**.
`kernel_gradient.field` already takes `negative` separately from `generated`;
training currently passes the same tensor for both, which is a choice nobody
has examined.

- probe cloud fixed at 64 (the generator's real batch)
- negative cloud ∈ {64, 256, 1024}, drawn from the *same* generator but as an
  independent sample
- prediction: the deficit shrinks monotonically in the negative cloud size,
  at a cost that scales only with the field, not with the gradient batch

This is the sharpest of the three: it isolates the hypothesised error source
and changes nothing else. It is also the most practically valuable, since a
bigger negative cloud is cheap while a bigger training batch is not.

**It is a deviation from the paper, and must be labelled as one.** Algorithm
2 deliberately reuses the particle cloud as its own negatives —
`lowdim_drift.drift_paper` calls this "the paper reuse pattern" — and
`train._branch_drifts` implements it by passing the same tensor for probes
and negatives (`KG.field(probes, positives, probes, ...)`). `field` already
takes the two separately, so the change is small, but a positive P2 result
would be evidence *against the paper's design choice*, not a bug fix. That is
a stronger claim and needs the stronger framing: the paper's reuse is what
makes the self-term's estimation error scale with the training batch, and
whether that is a real cost is exactly what P2 measures.

### P3 — the deficit should track multimodality

Mode-collapse should be worse when there are more modes to collapse onto.
In the low-dimensional harness, sweep the mixture component count
{1, 2, 4, 16} at fixed dimension and bandwidth.

- prediction: a unimodal Gaussian shows the smallest deficit; it grows with
  component count
- a flat result across component count would be strong evidence **against**
  H9, which is why it is worth running

### What would refute H9 outright

- P2 flat (a 16× better self-estimate changes nothing) — the cloud-side
  estimate is then not the error source;
- P1 and R11 strictly additive with no overlap — then R11 is doing something
  noise injection is not;
- P3 flat across modality — then "mode-collapse" is the wrong word for it.

---

## 4. Standing caveats

- **H9 is unmeasured.** It is a hypothesis fitted to nine existing
  observations, which is exactly the situation in which the previous eight
  looked good too. The table in §2 is retrodiction, not evidence.
- The MMD-flow correspondence is structural, not exact: Algorithm 2's
  `sqrt(row ⊙ col)` reweighting has no counterpart in the standard MMD flow,
  and the follow-up showed that reweighting is load-bearing.
- Phase 7 may change the framing before any of this runs — if 7A retires
  R11, then §2's "R11 is a substitute for noise injection" needs restating,
  though H9 itself would be untouched since it is about the flow, not the
  correction.
- Priority is below Phase 7 and below whatever Phase 7's gate decides.

## 5. Sources

- Carreira-Perpiñán, *Fast Nonparametric Clustering with Gaussian Blurring
  Mean-Shift* (ICML 2006) and *A review of mean-shift algorithms for
  clustering* — GBMS collapses to a single point, diameter decreasing at
  least geometrically.
- Arbel, Korba, Salim, Gretton, *Maximum Mean Discrepancy Gradient Flow*
  (NeurIPS 2019) — particles collapse around the mode or stick at local
  minima; noise injection proposed as the remedy.
- Chen et al., *(De)-regularized Maximum Mean Discrepancy Gradient Flow*
  (2024) and *Interaction-Force Transport Gradient Flows* (2024) — later
  attempts to avoid particle collapse without heuristic noise injection.
