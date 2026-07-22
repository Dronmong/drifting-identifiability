# PSQT minibatch-quantile accumulator defect

**Status:** causal development audit; not a confirmatory result  
**Affected implementation:** `persistent_sliced_quantile_transport.py`  
**Scope:** the target-statistic layer of scalar PQT and projected PSQT

## Executive conclusion

The diagonal-mixture regression is not primarily a failure of sliced
transport, two-dimensional geometry, the particle count, or reconstruction
depth.  It is caused by the target statistic currently accumulated by PQT:

```text
running mean of empirical minibatch quantile functions
```

For a fixed projection, this converges to the mean empirical quantile
`E[Q_hat_B]`, not to the population quantile `Q_p`.  The former is the
one-dimensional Wasserstein barycenter of random size-`B` empirical measures.
For disconnected or strongly multimodal laws it smooths a true quantile jump
into a ramp.  Particle reconstruction then correctly follows that corrupted
target and places artificial mass in a low-density gap.

This distinction matters: the geometric PSQT mechanism remains promising.
When the same reconstruction is supplied pooled rather than batch-averaged
quantiles, its error drops sharply and the previously failing diagonal family
beats the selected paper estimator.

## 1. Exact mechanism

Fix a unit projection `theta` and probability knot `u`.  The current update
forms a batch empirical quantile `Q_hat_b(u)` and stores

```text
Q_bar_T(u) = (prior + sum_b Q_hat_b(u)) / (prior_mass + T).
```

Ignoring the vanishing initialization prior, the law of large numbers gives

```text
Q_bar_T(u) -> E[Q_hat_B(u)].
```

There is no general identity `E[Q_hat_B(u)] = Q_p(u)`.  Quantiles are nonlinear
statistics, so averaging separately sorted batches does not equal sorting all
observations together.

### Two-point calculation

Let the projected target be the idealized mixture

```text
p = 0.5 delta_{-a} + 0.5 delta_{a}.
```

For a batch of size `B`, let `C ~ Binomial(B, 0.5)` be the number of left-mode
observations.  Up to the selected empirical-quantile convention, the rank-`r`
quantile is

```text
Q_hat_r = -a  if r <= C,
           a  if r > C.
```

Consequently,

```text
E[Q_hat_r] = a * (1 - 2 P(C >= r)).
```

The population quantile jumps immediately from `-a` to `a`.  The expectation
above instead follows the binomial CDF through a transition containing
`O(sqrt(B))` ranks.  On the probability scale this has width `O(B^(-1/2))`.
With `K` stored knots, approximately `K / sqrt(B)` knots lie in the artificial
bridge.  Increasing `K` refines the ramp; it does not reduce its mass.

The argument applies independently to every projection that separates two
modes.  The two-dimensional visualization makes a scalar statistical bias
visible as particles between modes.

## 2. Controlled evidence

The most diagnostic projection is the diagonal of the target with narrow
Gaussian modes near `(-1,-1)` and `(1,1)`.  With 300 batches of 64 targets and
64 quantile knots:

| Quantile table | knots in projected gap | largest central jump |
|---|---:|---:|
| mean of batch quantiles | 7 | about 0.28 |
| all 19,200 observations pooled | 0 | about 2.40 |
| 200,000-observation population approximation | 0 | about 2.40 |

The reconstructed cloud matched the biased table: seven to eight of its 64
particles occupied a region whose target probability was effectively zero.

Holding directions, particles, and initialization fixed and changing only the
table produced:

| Target table | ED2 | gap particles |
|---|---:|---:|
| batch-mean, 30 or more reconstruction sweeps | about 0.0143 | 8 |
| pooled, 100 reconstruction sweeps | about 0.0017 | 2 |
| population approximation, 100 sweeps | about 0.0018 | 2 |

Hundreds of additional reconstruction sweeps did not improve the batch-mean
case.  The optimizer had already converged to the wrong projected statistic.

### Batch-size scaling

At a fixed total of 65,536 target observations, the number of artificial gap
knots fell only gradually:

| Batch size | gap knots (of 64) |
|---:|---:|
| 16 | 14 |
| 32 | 10 |
| 64 | 7 |
| 128 | 5 |
| 256 | 4 |
| 512 | 2 |
| 1,024 | 2 |

At batch size 64, the bridge fraction stayed near 11--13% as the knot count
increased from 16 to 512, matching the two-point calculation.

### Exact source of the randomness

When every size-64 batch was artificially stratified to contain exactly 32
observations from each mode, the average batch table had no gap knots and ED2
fell to about 0.0019.  This diagnostic is not a usable general algorithm, but
it establishes that random batch component counts generate the ramp.

## 3. Primary and secondary defects

The evidence separates two mechanisms.

### Primary: target-statistic bias

The batch-mean table is wrong before reconstruction starts.  This is
responsible for most of the diagonal error and affects scalar PQT as well as
every projected version.

### Secondary: finite projected consistency

Even the pooled table sometimes leaves two bridge particles.  Increasing the
direction count can remove them: in controlled pooled-table runs, 64
particles with 128 directions reached zero bridge particles.  This smaller
residual comes from reconciling finitely many rank assignments and projected
constraints.  It should be addressed only after repairing the target
statistic.

## 4. What the defect is not

- **Not insufficient reconstruction:** the biased-table result converges in
  roughly 30 sweeps and does not improve through 300.
- **Not insufficient knot resolution:** bridge *fraction* remains stable as
  knots increase.
- **Not particle count alone:** adding particles represents the artificial
  ramp more accurately.
- **Not merely stale tables:** refreshing a reservoir-derived table every 16
  batches performed like refreshing every batch.
- **Not an inherently high-dimensional failure:** the biased operation is
  one-dimensional and is applied separately on every slice.
- **Not repaired by Sinkhorn after the fact:** a joint-assignment correction
  cannot recover target information already erased by the accumulator.

## 5. Full-screen implication

Using pooled target quantiles with the same 32 directions and 64 particles was
not a diagonal-only patch.  Across the original nine-family development
screen, it achieved approximately:

```text
pooled-table ED2 / current PSQT ED2 = 0.49
pooled-table ED2 / selected paper ED2 = 0.18
```

Every family improved relative to the current PSQT accumulator.  On the
diagonal dependence family, the pooled version was about `0.11` times the
paper error, reversing the only original family regression.  These numbers
are post-hoc development evidence and an upper bound on what a deployable
streaming approximation might achieve, not a frozen performance claim.

## 6. Candidate repairs already probed

| Accumulator | Benefit | Limitation |
|---|---|---|
| exact pooled observations | strongest reference; preserves jumps | storage grows with the stream |
| median of batch quantiles | removes the equal-mixture bridge | stores/history or online sketches needed; slight skew-family regression |
| raw point reservoir | simple; supports new directions | subsampling variance in rare/component mass |
| per-direction quantile sketch | uses the whole stream with bounded state | fixed directions; approximation must be audited |

A 1,024-point online reservoir was already sufficient to remove the median
diagonal bridge, but its measured mode proportion ranged roughly from 0.467
to 0.527 across seeds.  That variability explains its tail risk.  A streaming
quantile sketch per fixed direction uses every target observation and is the
preferred scalable repair.  A small raw reservoir remains useful for proposing
or initializing new adaptive directions.

## 7. Literature cross-check

This diagnosis agrees with, but is more specific than, known minibatch OT
warnings:

- Fatras et al., *Learning with minibatch Wasserstein: asymptotic and gradient
  properties* (AISTATS 2020), show that expected minibatch OT is a distinct
  objective and may lose distance properties:
  <https://proceedings.mlr.press/v108/fatras20a.html>.
- Fatras et al., *Unbalanced minibatch Optimal Transport* document undesirable
  minibatch smoothing and cross-cluster effects:
  <https://arxiv.org/abs/2103.03606>.
- Jang and Noh, *On the Finite-Sample Bias of Minimizing Expected Wasserstein
  Loss Between Empirical Distributions* (AISTATS 2026), derive finite-sample
  Wasserstein-loss bias even in well-specified one-dimensional settings:
  <https://openreview.net/forum?id=NkKkUL9380>.
- The recent preprint *Estimating the Wasserstein barycenter of
  one-dimensional distributions under sparse sampling* studies severe bias
  in naïve empirical one-dimensional barycenters:
  <https://arxiv.org/abs/2606.10096>.  It is especially close to the observed
  mechanism but should be treated as a recent preprint.
- Karnin, Lang, and Liberty, *Optimal Quantile Approximation in Streams*, give
  the KLL streaming-quantile construction motivating the proposed repair:
  <https://arxiv.org/abs/1603.05346>.

## 8. Decision

Do not add more geometric machinery to the batch-mean PSQT arm.  First replace
the target statistic and demonstrate that a bounded-memory online method
approaches the pooled-table ceiling without family regressions.  Only then
should discrepancy-focused directions, sequential ridge layers, or sparse
joint-assignment refreshes be evaluated against the remaining geometric
residual.

