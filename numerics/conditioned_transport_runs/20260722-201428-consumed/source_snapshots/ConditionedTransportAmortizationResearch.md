# Conditioned transport-then-amortize: diagnosis and implementation plan

**Status:** mathematical identities rechecked; implementation and consumed-
registry reproduction complete; fresh confirmation not yet run  
**Date:** 2026-07-22  
**Predecessors:** `KLLPSQTNeuralAmortizationResearch.md`,
`NeuralPooledRankPhase1Results.md`

## 1. Scope and evidence boundary

Phase 1 did not establish general neural KLL-PSQT superiority. Small KLL-RSR
had the best aggregate held-out SW1 and strong rare-mode behavior, but paper
retained the best ED2 and the target-level paired uncertainty included zero.
The large-RSR and hybrid arms failed the registered go/no-go gate.

This document records a post-hoc diagnosis on the consumed Phase-1 registry.
The resulting numbers are mechanism evidence and hyperparameter development,
not confirmation. Any performance claim requires a fresh registry generated
after the revised algorithm and all thresholds are frozen.

## 2. Failure A: the audited direction frame was the wrong frame

The existing `frame_diagnostics` checks the first-order operator

\[
  \frac dL\sum_{\ell=1}^L u_\ell u_\ell^\top.
\]

This certifies that feature-space correction vectors can be reconstructed from
their projections. It does **not** certify that projected distributions, or
even projected variances, determine the underlying law.

For a symmetric covariance matrix `Sigma`, projected variance is the linear
measurement

\[
  q_\ell=u_\ell^\top\Sigma u_\ell
       =\langle u_\ell u_\ell^\top,\Sigma\rangle_F.
\]

Thus covariance sensing is controlled by the span and conditioning of the
outer products `u_l u_l^T` in `Sym_d`, whose dimension is

\[
  s=d(d+1)/2.
\]

### 2.1 Orthogonal-block dimension count

Suppose the directions are `k` complete orthonormal bases. Within every block,

\[
  \sum_{i=1}^d u_i u_i^\top=I.
\]

Consequently all `k` block sums give the same trace measurement, producing at
least `k-1` linear dependencies. The quadratic sensing rank is therefore at
most

\[
  kd-(k-1)=k(d-1)+1.
\]

A necessary condition for covariance identification is

\[
  k(d-1)+1\ge d(d+1)/2,
\]

equivalently

\[
  k\ge \lceil(d+2)/2\rceil.
\]

For `d=16`, the registered four blocks have maximum rank `61`, while
`dim Sym_16 = 136`. Direct computation found rank exactly `61` for every 16D
target. In dimensions `2,4,8`, the registered bank had full quadratic rank.
This exactly matches the empirical dimension at which held-out performance
reversed.

### 2.2 Full rank is not sufficient for stability

Nine 16D blocks (`L=144`) reach full rank, but on the consumed registry their
quadratic sensing condition numbers ranged from roughly `371` to `2505`.
Twelve blocks (`L=192`) reduced that range to approximately `19--22`.

The implementation must therefore require both:

1. full quadratic rank; and
2. an explicit positive lower singular value / bounded condition number.

This is the numerical analogue of the project's formal
`InteractionFrameBound`: unnamed injectivity is not enough. The proposed
default grows orthogonal blocks until the condition number is at most `25`,
subject to an explicit maximum-block failure rather than silently accepting a
bad bank.

## 3. Failure B: matched generator examples destroyed optimizer time

For the Phase-1 standard budget of 20,480 generator-example evaluations:

```text
ordinary paper / minibatch SW: 320 Adam updates
small RSR (B=64):              160 Adam updates
large RSR (B=512):              20 Adam updates
```

The rank-loss normalization makes its neural gradient scale approximately
independent of population size. Adam's parameter trajectory is not independent
of the number of calls to `optimizer.step`, however. A larger learning rate
removed the gross initial failure but could not turn 20 nonlinear parameter
updates into 160.

A controlled probe held `B=512` fixed and changed only optimizer time. On the
16D nonlinear target with a full-rank bank:

```text
B=512,  20 updates, matched compute: ED2 0.0762, SW1 0.1458
B=512, 160 updates, 8x compute:       ED2 0.0315, SW1 0.0921
```

Large ranks are useful. The registered accounting policy simply made their
standard exact-RSR use too infrequent. This is consistent with
Run-Sort-ReRun's stated tradeoff: it removes the activation-memory barrier but
still pays additional training computation for a large effective population.

## 4. Verified gradient identity

For fixed directions and a rank assignment away from ties, the implemented
loss is

\[
  J_B(x)=\frac d{2LB}\sum_{j,\ell}
  (\langle x_j,u_\ell\rangle-t_{j\ell})^2.
\]

The implemented PSQT feature correction is

\[
  c_j=\frac dL\sum_\ell
  (t_{j\ell}-\langle x_j,u_\ell\rangle)u_\ell
     =-B\nabla_{x_j}J_B.
\]

Define coherent frozen teacher particles

\[
  y_j=\operatorname{stopgrad}(x_j+\eta c_j)
\]

and the student regression loss

\[
  S_B(\theta)=\frac1{2B}\sum_j
  \|G_\theta(z_j)-y_j\|^2.
\]

At the parameter value used to construct the teacher,

\[
  \nabla_\theta S_B=\eta\nabla_\theta J_B.
\]

Proof: `x_j-y_j=-eta*c_j` and the chain rule gives

\[
  \nabla_\theta S_B
  =-\frac\eta B\sum_j(D_\theta G(z_j))^\top c_j
  =\eta\sum_j(D_\theta G(z_j))^\top\nabla_{x_j}J_B.
\]

This identity holds for the first student gradient at a refresh. After an
optimizer update, the fixed teacher is intentionally a distillation target;
subsequent micro-updates are not claimed to equal newly reranked RSR gradients.

## 5. Conditioned transport-then-amortize algorithm

For a global population `B=512` and student microbatch `b=64`:

1. run all 512 latent samples without autograd;
2. globally rank their projections and compute the coherent PSQT correction
   `c` from a persistent exact or KLL target atlas;
3. compute the paper local field `v` on the same generated population;
4. RMS-normalize `v` against `c`, with an epsilon and an explicit scale cap;
5. freeze

   \[
   y=x+\eta\left(c+\lambda
     \min\left\{\frac{\operatorname{RMS}(c)}
                       {\operatorname{RMS}(v)+\epsilon},s_{\max}\right\}v
   \right);
   \]

6. permute the 512 latent/teacher pairs deterministically;
7. take eight ordinary Adam updates, one per 64-sample pair block; and
8. refresh ranks and teacher particles.

With 20 macro-steps this uses exactly:

```text
teacher run:       20 * 512 = 10,240 generator examples
student updates:   20 * 512 = 10,240 generator examples
total:                        20,480 generator examples
Adam updates:      20 * 8   = 160
```

It therefore combines 512-sample ranks with the optimizer frequency of small
RSR under the original generator-example budget. It is best described as
**conditioned transport-then-amortize**, not exact RSR.

The local term is complementary. The random-projection correction protects
global ranks and occupancy; the paper field supplies local geometry in
directions that a finite atlas can miss. Local-field RMS normalization is
necessary because its raw RMS was 50--230 times smaller than the PSQT
correction in the initial 16D populations.

## 6. Post-hoc mechanism results

The following settings were explored on consumed targets:

```text
particle step eta       0.5
local normalized weight 0.25
global population       512
student microbatch      64
Adam learning rate      0.016
direction rule          keep registered 64 unless quadratic conditioning
                        requires more; 16D used 192
```

On one concentrated-initialization replication across all 16 targets, the
exact-atlas version beat the tuned paper arm on both ED2 and held-out SW1 for
14/16 targets. Its medians were `0.01645 / 0.08695`, versus paper's
`0.04079 / 0.14435`. These are post-hoc numbers and must not be promoted.

The 16D KLL check closely matched exact:

| Family | Exact ED2 / SW1 | KLL ED2 / SW1 |
|---|---:|---:|
| balanced GMM | 0.05724 / 0.11672 | 0.05669 / 0.11645 |
| rare GMM | 0.02872 / 0.09046 | 0.02832 / 0.09021 |
| correlated-t | 0.02623 / 0.08306 | 0.02536 / 0.08070 |
| nonlinear | 0.02199 / 0.08109 | 0.02178 / 0.08003 |

The rare component retained full coverage. This strengthens the earlier
conclusion that sketch error is not the active bottleneck.

## 7. Protected-tail safety layer

A local field can improve global geometry while reducing a small component's
mass. The implementation should expose, but not overstate, a conservative
particle-space guard:

- compute the fixed-assignment rank loss before the particle step;
- use the global-only particle candidate as the safety baseline;
- compute it after each candidate local weight;
- separately compute loss on the outer protected rank fraction;
- accept only weights that do not increase either quantity relative to the
  global-only candidate, up to an explicit tolerance; and
- choose the largest safe registered candidate.

This is a one-step empirical safeguard, not a theorem of mode preservation.
It must be logged, ablated, and tested on atoms, gaps, and rare mixtures.

## 8. Implementation order

1. Add `QuadraticFrameDiagnostics` and a condition-certified orthogonal-bank
   extension routine.
2. Add the transport-then-amortize primitive and exact work ledger.
3. Prove by autograd regression that the first teacher gradient equals
   `eta * grad(rank_loss)`.
4. Add deterministic, zero-step, zero-local-field, KLL, budget, rank-deficient,
   full-rank, and conditioning tests.
5. Add the optional protected-tail local-weight selector.
6. Build a separate consumed-registry development runner; do not rewrite the
   Phase-1 artifacts.
7. Reproduce the post-hoc diagnostic with exact and KLL atlases and save all
   states and outputs.
8. Only after implementation settings are frozen, generate a fresh registry
   for target-level paired confirmation.

## 9. Literature boundary

- Run-Sort-ReRun establishes large-population rank computation with microbatch
  replay and explicitly treats it as a memory/compute tradeoff:
  <https://proceedings.mlr.press/v139/lezama21a.html>.
- Minibatch Wasserstein optimizes an implicitly regularized objective rather
  than a literal unbiased population distance:
  <https://proceedings.mlr.press/v108/fatras20a.html>.
- Informative/adaptive slicing is a known response to weak random directions:
  <https://proceedings.mlr.press/v235/nguyen24l.html> and
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html>.
- The recent W-Flow preprint independently supports the high-level pattern of
  constructing a particle Wasserstein flow and then compressing it into a
  static generator, but it uses a different Sinkhorn energy and does not prove
  this algorithm:
  <https://arxiv.org/abs/2605.11755>.
- Streaming sliced OT supports target-side quantile approximation, not the
  neural amortization step:
  <https://arxiv.org/abs/2505.06835>.

The proposed contribution is the particular combination of an audited
quadratic sensing certificate, persistent KLL target atlas, coherent PSQT
particle target, normalized local Drifting field, and compute-matched
multi-update amortization. No novelty claim should be made before a dedicated
paper search and fresh confirmation.

## 10. Implementation checkpoint

Steps 1--7 above are complete. The implementation is in
`neural_pooled_rank.py`, its regression coverage is in
`neural_pooled_rank_tests.py`, and the separate auditable runner is
`run_conditioned_transport_development.py`. The authoritative outputs and
their interpretation are recorded in
`ConditionedTransportAmortizationResults.md`.

The sole remaining experimental step is item 8: freeze the selected candidate
and its gates, generate a fresh registry, and run target-level paired
confirmation without further tuning on those outcomes.
