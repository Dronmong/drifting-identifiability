# Encoder-independent balanced Sinkhorn mechanism protocol

**Status:** S0.1--S0.2 implemented and mechanically verified. No development
or confirmatory outcome has been run, and no research hyperparameter is frozen
by this document yet.

This protocol implements Stage S0--S1 from
`MeanFlowDriftingMechanismAnalysis.md`. It asks one narrow question before the
pixel MeanFlow build begins:

> On the already validated B0 bridge, does a two-sided balanced
> cross-minus-self Sinkhorn correction improve a held-out distributional
> discrepancy while retaining coverage and effective rank?

B0 remains a multi-step flow-matching bridge. A positive result licenses
carrying the correction into a later one-call pMF experiment; it is not itself
a one-step-generation result.

## 1. Empirical functional

At correction event `e`, draw three independent roles:

- primary generated endpoints `x_i = G_theta(z_i)`, with gradients;
- self-support endpoints `x'_k = G_theta(z'_k)`, under `no_grad`;
- augmented real samples `y_j`, detached.

The prior streams for `z` and `z'` are distinct. Both generated batches use
the current model and the same frozen Euler NFE, but they share neither noise
nor storage.

For Stage S1 use raw flattened pixels and the normalized quadratic costs

```text
C_qp[i,j] = ||x_i - y_j||^2 / (2 * cost_scale)
C_qq[i,k] = ||x_i - x'_k||^2 / (2 * cost_scale).
```

`cost_scale > 0` is estimated once from target-only training data and then
frozen. The exact raw identity geometry is used; Haar is deferred to Stage S2.

For uniform row and column masses, compute

```text
pi_qp = argmin_pi <pi,C_qp> + epsilon KL(pi || u_B tensor u_M)
pi_qq = argmin_pi <pi,C_qq> + epsilon KL(pi || u_B tensor u_B')
```

with log-domain Sinkhorn scaling. Both marginals must pass the frozen maximum
relative residual. A row's conditional weights are `pi[i,:] / u_B[i]`, not a
new softmax or a geometric mean of one-sided normalizations.

The transformed endpoint velocity is

```text
T_p(x_i) = sum_j pi_qp[i,j] / u_B[i] * y_j
T_q(x_i) = sum_k pi_qq[i,k] / u_B[i] * x'_k
V_i      = T_p(x_i) - T_q(x_i).
```

The correction is the detached drifted-target regression

```text
L_sinkhorn = mean_i ||x_i - stopgrad(x_i + eta V_i)||^2.
```

Plans, real samples, self samples, velocity, and target are detached. Only the
primary generated endpoint and its Euler trajectory receive gradients. There
is no diagonal mask.

## 2. Why this loss is used

For exact population Sinkhorn velocity `V = -grad delta F / delta q`, detached
target regression has parameter gradient proportional to `grad_theta F`.
For the finite two-batch construction above, the corresponding empirical
cross-minus-self energy is

```text
F_hat = OT_epsilon(q_hat, p_hat) - OT_epsilon(q_hat, q'_hat),
```

up to the target-only constant in the full Sinkhorn divergence. With exact
plans and quadratic cost,

```text
grad_x L_sinkhorn = 2 eta grad_x F_hat.
```

This identity is a finite-particle envelope calculation for the declared
two-batch estimator. Finite sampling and finite solver tolerance remain
approximations to the population flow. Finite `epsilon` defines the chosen
regularized objective; it is not called an estimator error relative to that
objective.

## 3. Chronological implementation gates

### Gate S0.1 -- solver

Require:

- square and rectangular uniform marginals;
- log-domain updates;
- maximum relative row and column residuals below tolerance;
- deterministic iteration count and replay;
- finite plans for large shifted costs;
- conditional row weights summing to one;
- explicit cap-hit reporting.

### Gate S0.2 -- gradient semantics

Require:

- zero velocity when cross and self problems are identical;
- gradients through primary endpoints only;
- no gradients through real or self supports;
- central finite-difference agreement for `F_hat`;
- detached-target gradient equal to `2 * eta * grad F_hat` within solver and
  finite-difference tolerance;
- independently seeded primary and self trajectories.

### Gate S0.3 -- resource and numerical preflight

On the smoke model and then one outcome-blind compact calibration unit, record:

- target-only cost scale;
- epsilon candidate;
- solver tolerance, iterations, and cap-hit rate;
- normalized conditional entropy and maximum conditional weight;
- velocity RMS and update-to-sample RMS ratio;
- event time and peak allocated/reserved memory;
- flow and correction parameter-gradient norms and cosine.

No FID, recall, KID, or sample image may select these constants.

### Gate S1 -- matched mechanism screen

After a GO preflight, freeze and compare:

| arm | objective |
|---|---|
| `B0-control` | B0 flow loss |
| `B0-Laplace` | B0 plus the already frozen B2 event |
| `B0-Sinkhorn-I` | B0 plus the frozen identity Sinkhorn event |

All arms share B0 initialization, flow data, endpoint noise, bridge time,
augmentation, optimizer setup, update count, and evaluation allocations.
Intervention streams are separate and explicitly recorded.

## 4. Preflight-only starting values

These are candidates to test mechanically, not frozen research choices:

| item | initial value |
|---|---:|
| correction cadence | every 10 updates |
| primary / self generated batch | 64 / 64 |
| real support batch | 128 if memory permits, otherwise 64 |
| generated trajectory | 8 Euler steps |
| normalized quadratic-cost median | 1 |
| epsilon candidates | 0.05 and 0.10 |
| maximum relative marginal residual | `1e-3` |
| maximum Sinkhorn iterations | 100 |
| event velocity step `eta` | calibrated from update/sample RMS, starting 0.05 |
| event gradient ratio | 0.25 of the B0 flow gradient |

Published epsilon or step values are not copied across feature scalings. The
preflight may reject both epsilon candidates and return NO-GO.

## 5. Promotion and claim boundary

Sinkhorn is promoted only if the frozen units show:

- valid marginal convergence and two-batch role separation;
- improved held-out Sinkhorn discrepancy relative to paired B0;
- B0 coverage and effective-rank retention;
- no memorization or collapse veto;
- a consistent improvement on at least one preregistered quality/coverage
  axis;
- no regression that makes it clearly inferior to the frozen B2 incumbent.

Lower training loss alone is not a pass. This stage cannot claim one-step
generation, encoder-independence at ImageNet scale, or superiority to the
published paper model. It is the component test required before the pMF build.

## 6. Implementation checkpoint (2026-08-02)

The isolated implementation lives in
`encoder_independent_drifting/stage_sinkhorn/`:

- `core.py` implements normalized quadratic costs, target-only cost scaling,
  rectangular log-domain balancing, explicit relative marginal checks,
  conditional barycenters, the two-batch cross-minus-self velocity, the
  detached target loss, and the matching finite empirical energy;
- `training.py` integrates the correction with B0's differentiable Euler
  endpoint while preserving B0's model initialization and flow streams;
- `tests/test_core.py` and `tests/test_training.py` currently contain eleven
  passing mechanical tests.

Verified properties include:

- square and rectangular marginal convergence;
- stable constant-cost behavior even at a very large common cost;
- zero velocity for identical cross and self transport problems;
- rejection of same-tensor self support;
- explicit cap-hit failure;
- no gradients through real or self supports;
- finite-difference agreement for the two-batch entropic energy;
- exact detached-target gradient agreement
  `grad L = 2 eta grad F_hat` at numerical tolerance;
- deterministic correction replay;
- gradients reaching model parameters through the primary Euler trajectory;
- paired B0/Sinkhorn initialization and first flow loss;
- distinct primary/self seeds and the absence of a diagonal mask.

Both stage-local test modules pass, and Ruff reports no formatting or lint
errors. The next chronological task is S0.3: an outcome-blind cost/epsilon,
gradient, memory, and timing preflight. Passing unit tests do not freeze an
epsilon or authorize the long S1 screen.
