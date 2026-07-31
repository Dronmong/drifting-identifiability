# Projection and kernel cost optimization plan

## Purpose and scope

The conditioned transport-then-amortize result established a quality gain at a
matched generator-example budget, but it did **not** establish a compute or
wall-clock gain.  This document audits the two additional costs that matter
before attempting image-feature experiments:

1. repeated projection, sorting, and backprojection through the persistent
   quantile atlas; and
2. the dense local Algorithm-2 affinity between the generated population and
   the positive/negative supports.

The first implementation stage below is exact: it must produce the same PSQT
correction and the same local field up to ordinary floating-point roundoff.
Direction subsampling, local-field scheduling, coresets, and kernel feature
maps are later experimental stages and must not be described as exact.

## Audited current cost

The confirmed candidate uses a particle population `B = 512`, 20 transport
macro-steps, and between 64 and 192 registered directions.  Its recorded work
per run is:

| Quantity | Candidate | Paper port |
|---|---:|---:|
| generator-example evaluations | 20,480 | 20,480 |
| generator forward calls | 320 | 320 |
| local kernel pairs | 10,485,760 | 2,621,440 |
| direction/sample dot products, 2D--8D | 2,621,440 | 0 |
| direction/sample dot products, mean 16D | 7,782,400 | 0 |

The ledger name `projection_scalar_products` counts complete direction/sample
dot products, not their scalar coordinate operations.  A dense `d`-dimensional
dot product still costs approximately `d` multiply-adds.

The local pair formula is

```text
T * B * (B_positive + B_negative) = 20 * 512 * 1024.
```

The 16D projection count is larger because the registered 64-direction bank is
extended to 176 or 192 directions until the explicit quadratic sensing matrix
has full rank and condition number at most 25.

## Mathematical re-check

### Exact normalized-kernel identity

Let `K_ij = exp(-||x_i-y_j|| / tau)`, with the masked entries changed in the
same way as the implementation.  Define

```text
r_i = sum_j K_ij,
c_j = sum_i K_ij.
```

The implementation forms a row softmax and a column softmax and takes their
geometric mean.  Entrywise,

```text
sqrt((K_ij / r_i) * (K_ij / c_j)) = K_ij / sqrt(r_i * c_j),
```

because `K_ij >= 0`.  This is an exact algebraic identity, not a kernel
approximation.

Partition the affinity `A` into positive and negative columns and write

```text
alpha_i = sum_(j in positive) A_ij,
beta_i  = sum_(j in negative) A_ij,
U_i     = sum_(j in positive) A_ij y_j,
V_i     = sum_(j in negative) A_ij y_j.
```

Then the implemented field is exactly

```text
field_i = beta_i * U_i - alpha_i * V_i.
```

A numerical audit over float64 and float32 random inputs, both mask settings,
and an extreme underflow case found maximum discrepancies of approximately
`6.7e-16` and `3.6e-7`, respectively, between the original softmax expression
and the direct mass expression.  An underflow-safe log-sum-exp fallback is
needed when a complete raw-kernel row or column rounds to zero.

### Exact projection reuse

For fixed directions `u_l`, assigned target quantiles `t_jl`, and generated
features `h_j`, the free-particle correction is

```text
g_j = (d / L) * sum_l (t_jl - <h_j,u_l>) u_l.
```

The current `psqt_feature_correction` first computes all `<h_j,u_l>` to sort
and assign ranks, then recomputes the identical matrix before backprojection.
The first matrix may be reused.  Rank assignment uses its detached value while
the correction may retain the original autograd graph, so this change does not
alter rank semantics or differentiation.

### Unbiased active-direction estimator

Let `g_l` denote the contribution from direction `l`.  For a uniformly chosen
subset `S` of size `a`, sampled without replacement,

```text
g_hat = (d / a) * sum_(l in S) g_l.
```

Since each direction is included with probability `a/L`, linearity of
expectation gives

```text
E[g_hat] = (d / L) * sum_(l=1)^L g_l.
```

Thus direction sharding is unbiased for the fixed finite-atlas correction.
It is nevertheless a stochastic algorithmic change: its variance can alter
the transport trajectory, and an active subset need not carry the full frame
certificate at a single step.

This statement applies to a fresh conditionally uniform subset.  The first P1
implementation below instead prioritizes balanced exposure: it always chooses
whole blocks having the least prior exposure and uses randomness only for
ties.  That schedule is not claimed to be a conditionally unbiased gradient
estimator after conditioning on its history.  Its justification is complete,
near-uniform registered-bank coverage with lower schedule variance.  A fresh
uniform-block mode remains a separate P1 comparator.

### Why the ambient quadratic certificate cannot scale

The space of symmetric `d x d` covariance matrices has dimension

```text
s = d(d+1)/2.
```

Full covariance sensing therefore requires at least `s` scalar projection
measurements.  At `d = 512`, `s = 131,328`.  If the current explicit sensing
matrix were approximately square, it would have about 17.25 billion float64
entries, or roughly 138 GB, before running an SVD.  Applying that full bank to
512 particles for 20 macro-steps would require about 1.38 trillion dense
projection multiply-adds.

This certificate is useful in low dimensions, but it must not be promoted to
an ambient image-feature requirement.  The scalable replacement should be:

- an exact first-order tight-frame certificate;
- held-out empirical distribution and covariance diagnostics;
- optional full quadratic sensing only in a frozen target-only intrinsic
  subspace; and
- explicit residual probes outside that subspace.

This weaker certificate must be described honestly: it does not prove that a
finite high-dimensional direction bank identifies every covariance matrix or
every probability law.

## Optimization layers

### E0: exact engineering changes

1. Reuse the generated projection matrix in `psqt_feature_correction`.
2. Update the conditioned-transport cost ledger from two generated projection
   passes per macro-step to one.
3. Replace the two explicit softmax matrices by the normalized-kernel identity.
4. Use an underflow-safe fallback and preserve the literal self-mask.
5. Add float64/float32, mask/no-mask, matched-batch, and extreme-scale
   regression tests.
6. Profile `torch.cdist` and tiled reductions separately; neither is called a
   speed improvement until measured on the intended device.

These changes preserve the mathematical algorithm.  Projection reuse should
remove almost half the repeated projection arithmetic in the conditioned
teacher.  Kernel fusion reduces intermediate storage and redundant
normalization work but leaves the `O(B^2 d)` pair complexity unchanged.

### P1: registered direction sharding

Keep the complete frozen atlas but activate only 16, 32, or 64 directions per
macro-step.  Compare:

- deterministic round-robin orthogonal blocks;
- uniform blocks sampled without replacement; and
- full-atlas correction.

Every registered direction must receive equal exposure over a complete cycle.
Held-out directions remain evaluation-only.  Record correction variance,
training quantile residual, held-out SW1, ED2, rare-mode coverage, and wall
time.  The first experiment should not introduce adaptive directions.

At 16D, active counts 64, 32, and 16 reduce repeated projection and sorting
work by approximately 3x, 6x, and 12x relative to a 192-direction step.

For genuinely high-dimensional feature spaces, investigate signed-Hadamard or
DCT orthogonal blocks.  A complete block remains a first-order tight frame and
can be projected with a fast transform, although sorting is still required for
each resulting coordinate.

### K1: local-field cadence

The persistent global PSQT correction carries occupancy and distributional
matching.  The local paper field is auxiliary, so test it on 20, 10, and 5 of
the 20 macro-steps.  Do not reuse a stale local field across changed generated
particles; skipped steps simply use zero local contribution.

Five exact local evaluations give 2,621,440 kernel pairs, matching the paper
port's recorded pair count while retaining the 512-particle global correction.

### K2: weighted positive/negative representatives

Replace each 512-point local support by `M` weighted representatives.  With
`M` representatives on both sides, local work becomes `2*T*B*M`:

| `M` | Pairs | Reduction from current |
|---:|---:|---:|
| 128 | 2,621,440 | 4x |
| 64 | 1,310,720 | 8x |
| 32 | 655,360 | 16x |

Target representatives may be persistent.  Generated representatives must be
refreshed.  Multiplicities must enter row/column masses, and rare atlas cells
must receive a minimum representative allocation.

For `k(x,y)=exp(-||x-y||/tau)`,

```text
|k(x,y)-k(x,c)| <= ||y-c||/tau.
```

Cluster radius and a positive lower bound on normalization masses therefore
give an auditable absolute field-error route.  Record row-mass error,
column-mass error, field relative L2 error, field cosine, and rare-mode errors
against the exact local field.

### K3: positive low-rank kernel features

If a nonnegative map `phi : R^d -> R_+^r` satisfies

```text
K(x,y) approximately equals <phi(x),phi(y)>,
```

then all row masses, column masses, and weighted centroid numerators factor
through `r`-dimensional sums.  The normalized field can be evaluated in
roughly `O(B*r*d)` rather than `O(B^2*d)`, with the generated self-mask handled
as a sparse diagonal correction.

Nonnegativity is essential.  Ordinary sine/cosine random Fourier features for
the Euclidean Laplace kernel can produce negative approximate entries and are
therefore unsafe inside square-root mass normalization.  Plain Nystrom also
does not guarantee entrywise nonnegativity.

If an approximation satisfies the entrywise relative bound

```text
(1-epsilon) K_ij <= Khat_ij <= (1+epsilon) K_ij,
```

then its row and column masses satisfy the same bound and

```text
(1-epsilon)/(1+epsilon)
  <= Ahat_ij/A_ij
  <= (1+epsilon)/(1-epsilon).
```

This is the preferred certificate for a later positive-feature experiment.

## Experimental order and gates

Implementation order:

1. E0 exact-equivalence changes and microbenchmarks.
2. P1 direction sharding in isolation.
3. K1 local cadence in isolation.
4. K2 weighted representatives in isolation.
5. Combine the best P1 and K1/K2 settings.
6. Run fresh 32D, 64D, and 128D synthetic stress tests.
7. Attempt K3 only after the simpler cost frontier is understood.

Every approximate stage must report both quality and actual cost:

- ED2 and held-out SW1 against paper and the target-wise baseline envelope;
- rare-mode coverage and rare-mass error;
- target accesses, direction/sample dot products, sorting work, and kernel
  pairs or feature-rank operations;
- CPU/GPU wall time and peak device memory; and
- approximation-specific field and mass errors.

A cost-optimized arm is successful only if a fresh paired confirmation still
places the upper confidence bound for its ED2 and SW1 ratios below one against
the paper port, does not lose rare-mode coverage, and materially reduces
measured total work.  Matching generator-example count alone is insufficient.

## Approaches deliberately deferred

- Lowering the 512-particle population: it directly threatens the rare-mode
  and occupancy mechanism that produced the confirmed gain.
- Approximate sorting: rank errors can jump across atoms or low-density gaps.
- Plain random Fourier features: kernel entries can be negative.
- Plain Nystrom: entrywise nonnegativity and stable normalization are not
  guaranteed.
- Pure nearest-neighbor sparsification: it can remove matrix support and omit
  rare but important interactions.  Sparse/local correction is a later option
  only with omitted-mass and support diagnostics.
- Fully adaptive projection directions: they complicate atlas persistence,
  held-out validity, and frame certification.  A small adaptive supplement may
  be considered after the registered-core experiment.

## Literature anchors

- Charlier et al., *Kernel Operations on the GPU, with Autodiff, without
  Memory Overflows*, JMLR 2021:
  <https://jmlr.csail.mit.edu/papers/v22/20-275.html>
- Scetbon and Cuturi, *Linear Time Sinkhorn Divergences using Positive
  Features*, NeurIPS 2020:
  <https://papers.nips.cc/paper_files/paper/2020/hash/9bde76f262285bb1eaeb7b40c758b53e-Abstract.html>
- Gasteiger, Lienen, and Guennemann, *Scalable Optimal Transport in High
  Dimensions*, 2021: <https://arxiv.org/abs/2107.06876>
- Choromanski et al., *Orthogonal Random Features*, NeurIPS 2017:
  <https://research.google/pubs/orthogonal-random-features/>
- Nadjahi et al., *Fast Approximation of the Sliced-Wasserstein Distance Using
  Concentration of Random Projections*, NeurIPS 2021:
  <https://proceedings.neurips.cc/paper/2021/hash/6786f3c62fbf9021694f6e51cc07fe3c-Abstract.html>
- Ahir and Pandit, *Feature Maps for the Laplacian Kernel and Its
  Generalizations*, 2025: <https://arxiv.org/abs/2502.15575>

## Current checkpoint

The analysis, algebraic re-check, and first E0 implementation are complete:

- `psqt_feature_correction` now reuses its generated projection matrix;
- the conditioned-transport ledger counts one generated projection pass per
  macro-step;
- the local field uses the direct normalized-kernel identity with a
  log-sum-exp underflow fallback; and
- `exact_cost_refactor_test` checks correction values and gradients, both
  floating dtypes, both mask modes, and the underflow case.

The complete `neural_pooled_rank_tests.py` suite passes.  A preliminary
single-thread CPU microbenchmark at `B=512`, `d=16`, and `L=192` measured:

| Operation | Before | E0 | Interpretation |
|---|---:|---:|---|
| one PSQT correction | 7.44 ms | 7.64 ms | no measured speed gain; sorting dominates |
| one local field | 7.94 ms | 1.76 ms | about 4.5x faster in this environment |

The float32 local-field maximum absolute difference in that benchmark was
`2.98e-7`; the independent float64 NumPy-port comparison was `3.33e-16`.
These are implementation-specific CPU timings, not GPU or end-to-end speed
claims.  Projection reuse is currently justified by exact work accounting and
future high-dimensional scaling, not by this small CPU timing.

P1 direction sharding infrastructure is now implemented:

- `QuantileAtlas.select_directions` creates immutable registered subsets and
  carries the corresponding exact/KLL rows and serialized KLL states;
- `balanced_orthogonal_block_schedule` checks every contiguous registered
  `d x d` block for orthogonality, selects only complete blocks, introduces no
  within-step duplicate, and keeps final block exposures within one;
- `train_arm` accepts an active direction count and applies the scheduled
  subset separately at every macro-step;
- the CLI accepts `--active-directions 16|32|64` (or zero for the full bank);
  and
- the ledger now separates one-time atlas projection/sorting from repeated
  training projection/sorting and records active counts and exposure.

An explicit one-macro training comparison between the default full path and
`active_direction_count = L` produced bitwise-identical model parameters and
identical ledgers.  The complete numerical regression suite passes.

The first active-16 engineering smoke also passed artifact audit.  Relative to
the full smoke, the overall exact-hybrid median changed from ED2 `0.06908` /
SW1 `0.17858` to `0.07228` / `0.17955`.  The two-step 16D smoke was worse by
about 7.3% ED2 and 9.5% SW1 because it could visit only two of twelve
orthogonal blocks.  This is not a valid quality verdict for the 20-step
registered experiment, where active-16 visits every block at least once.

For the 16D smoke row, repeated training projection work fell from 196,608 to
16,384 direction/sample dot products (12x), while the separately reported
one-time atlas projection remained 393,216.  End-to-end CPU field/training
time for that row fell from about 35.7 ms to 20.9 ms, but the smoke is too
small for a speed claim.

The 20-step P1 development screen is complete.  Active-32 is the selected
development configuration: paired median exact-hybrid ED2/SW1 ratios against
full were `0.9486 / 0.9553`, repeated projection work halved overall, and
median CPU training time fell about 16%.  Detailed cell and cost results are in
`ProjectionKernelCostOptimizationResults.md`.

K1 local-field scheduling is also implemented and screened.  Midpoint-spaced
10/20 and 5/20 schedules reduced kernel pairs by 2x and 4x, but monotonically
degraded the exact-hybrid endpoint.  The full 20-call field remains the
best-quality selection; 10 calls remains an explicit Pareto option rather than
the promoted default.

K2 weighted-representative approximation is now implemented and screened.
The retained API uses balanced registered-direction projection trees, integer
multiplicities, and per-conceptual-column deleted-self normalization.  The
full `M=B` path agrees with the dense field to float64 roundoff, and optional
dense audits report field cosine/L2 error, row/column-mass error, and cluster
radii.  The tree is vectorized across every cluster at a level.

With active-32 and all 20 local calls fixed, `M=128` reduced local kernel pairs
4x and measured end-to-end CPU time about 8.2%, while paired median ED2/SW1
increased about 3.8%/2.2% against dense active-32.  It matches the repository
paper port's kernel-pair count and retains 16/16 development quality wins on
both primary metrics.  `M=64` is a more aggressive 8x pair reduction with a
larger quality cost.  Exact artifacts, the bandwidth-confound exclusion, and
the full interpretation are in `ProjectionKernelCostOptimizationResults.md`.

The fresh paired confirmation of frozen active-32 plus `M=128` is complete,
with dense active-32, `M=64`, and a matched current paper port as comparators.
It passed the predeclared ED2/SW1 gates; see
`ProjectionKernelOptimizationConfirmationResults.md`. Positive feature
approximation (K3) remains unimplemented and must not be selected on either
the consumed development registry or the completed confirmation registry.
