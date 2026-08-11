# Identity-Sinkhorn Stage S0 Results

Date: 2026-08-02

## Scope

This is a mechanical component result, not an image-generation result. Stage
S0 checks whether the proposed encoder-free, identity-pixel,
cross-minus-independent-self Sinkhorn correction is numerically defined,
differentiates through the intended branch, and can run on the available GPU.
No FID, KID, recall, generated image, or candidate checkpoint was inspected.

## What was implemented

- Rectangular balanced transport in the log domain.
- Explicit row/column marginal residuals and iteration-cap failure.
- Target-only normalization of squared pixel distance.
- Independent primary generated, self generated, and real support batches.
- Detached transport targets with gradients only through the primary Euler
  trajectory.
- The exact finite-batch check
  `grad L = 2 eta grad (OT_epsilon(q,p) - OT_epsilon(q,q'))`.
- Reproducible stream separation, solver diagnostics, gradient geometry,
  timing, memory, and hash-checked freeze artifacts.

Fourteen stage-local tests pass: eight transport/gradient tests and six B0
integration/freeze tests. Ruff formatting and lint checks also pass.

## Preflight audit trail

The first completed preflight returned `NO-GO` even though both transport
solvers and the memory gate passed. The cause was an inappropriate aggregation
rule: the median of the three per-unit event weights did not minimize the
worst multiplicative gradient-ratio error. This was a preflight-design error,
not a failed transport mechanism.

The repair selects the Chebyshev center in log weight space,

```text
lambda_common = sqrt(min_i lambda_i * max_i lambda_i),
```

which is the common positive weight minimizing the worst multiplicative
deviation from the requested event-level gradient ratio. A dedicated test
now covers this rule. The preflight was rerun from scratch with source hashes
and the previously missing flow/correction gradient cosine.

## Final S0.3 measurements

Both candidates passed every mechanical gate on B0 units 300--302.

| quantity | epsilon 0.05 | epsilon 0.10 |
|---|---:|---:|
| maximum solver iterations | 17 | 9 |
| maximum relative marginal residual | 9.999e-4 | 7.346e-4 |
| mean normalized cross entropy | 0.721 | 0.873 |
| mean normalized self entropy | 0.867 | 0.955 |
| maximum conditional coupling weight | 0.619 | 0.275 |
| mean update/sample RMS | 3.97% | 3.16% |
| frozen gradient-ratio range | 0.153--0.407 | 0.146--0.428 |
| flow/correction gradient cosine | 0.631--0.745 | 0.648--0.765 |
| mean event time | 1.60 s | 1.42 s |
| peak allocated GPU-memory fraction | 87.5% | 87.5% |

The positive cosines are encouraging: at the measured B0 checkpoints, the
correction is not fighting the flow objective. They are not evidence that the
trained candidate will improve image quality.

## Frozen decision

Stage S0 freezes:

- identity-pixel geometry;
- `epsilon = 0.10`;
- target cost scale `693.6276245117188`;
- event weight `lambda = 0.0006913055808885343`;
- correction every 10 updates;
- primary/self/real batches `64/64/128`;
- eight Euler steps for each generated correction support;
- velocity step `eta = 0.05`;
- relative marginal tolerance `1e-3` and cap 100.

The outcome-blind selection rule prefers, among mechanically valid
candidates, the least concentrated coupling, then fewer iterations, lower
event time, and smaller epsilon. It selects `epsilon = 0.10` because its
couplings are substantially less concentrated and its solver is faster.

Authoritative artifacts:

- `encoder_independent_drifting/stage_sinkhorn/s0_preflight_v3.json`;
- `encoder_independent_drifting/stage_sinkhorn/s0_freeze.json`.

## Next chronological step

Stage S1 is a matched component screen:

1. reuse the immutable historical B0 control and B2 Laplace results;
2. train only the new `B0-Sinkhorn-I` arm from the exact B0 initialization
   with the exact B0 flow streams;
3. evaluate it with paired priors and the same developmental reference source;
4. require lower held-out identity-Sinkhorn drift than B0 while retaining
   coverage, effective rank, and memorization safety;
5. compare against B2 as the incumbent mechanism, without calling this reused
   source a fresh confirmation set.

No pixel MeanFlow or Haar geometry is introduced until this identity mechanism
screen succeeds. A full S1 training run is deliberately not launched by S0;
it is a separate multi-hour compute decision.
