# Identity-Sinkhorn Stage S1 Continuation Protocol

Status: corrected v2 design, written before any valid S1 candidate outcome.
The aborted v1 process trained from random initialization and produced no
checkpoint, metric, or unit shard; it is invalid for this protocol.

## Question

Starting from the already validated B0 generator, does an encoder-free
identity-pixel balanced Sinkhorn correction improve its held-out field and a
preregistered quality/coverage score without causing collapse, and is it
competitive with a matched continuation using the frozen B2 Laplace
correction?

This is a developmental component screen. It is not yet a one-step MeanFlow
result and cannot establish superiority to the paper.

## Matched continuation arms

For each unit, load the same immutable B0 EMA checkpoint into three arms:

| arm | continuation objective |
|---|---|
| `control` | B0 flow loss only |
| `laplace` | B0 flow loss plus the frozen B2 correction |
| `sinkhorn` | B0 flow loss plus the frozen S0 identity-Sinkhorn correction |

The historical B0 checkpoint contains model weights but no optimizer or data
cursor. Therefore this is explicitly a **matched fresh-optimizer
continuation**, not a restoration of the unavailable historical optimizer:

- every arm receives an identical fresh AdamW state;
- every arm uses the same new flow-data, endpoint-noise, bridge-time, and
  augmentation streams;
- every arm performs 5,000 continuation updates;
- correction-specific streams are independent and recorded;
- the only causal difference is the correction objective at its frozen
  cadence.

The starting state dict, optimizer configuration, flow seed manifest, update
count, EMA, and evaluation allocation must match exactly across arms.

## Frozen mechanisms

The S0 freeze supplies the Sinkhorn geometry, cost scale, epsilon, solver,
velocity step, event weight, batches, NFE, and cadence. These values are not
retuned.

The passing B2 freeze supplies the exact Laplace bandwidth, event weight,
probe construction, batches, NFE, and cadence. These values are not retuned.
The preflight records the realized B2 and Sinkhorn gradient ratios at the B0
continuation point; they are diagnostics, not tuning inputs.

## Expedited unit staging

Exactly two initial units are used. Their identities are determined without
looking at B2 or Sinkhorn outcomes:

1. concatenate the finalized protocol SHA-256, S0-freeze SHA-256, the fixed
   label `sinkhorn-s1-continuation-v2`, and each unit ID;
2. hash each string with SHA-256;
3. sort units by the resulting hexadecimal scores;
4. use the first two and reserve the third.

The staging verdict is fixed:

- two unit passes: `PROVISIONAL-GO`;
- zero unit passes: `STOP`;
- exactly one unit pass: `RUN-TIEBREAKER-<reserved-unit>`.

The two-unit result is explicitly provisional. The reserve cannot be run just
because a point estimate is inconvenient.

## Developmental evaluation

The CINIC-10 ImageNet-only pool already consumed by B2 is reused so the new
arms can be interpreted beside B2. It is not a fresh confirmation source.
Each unit uses its historically assigned control group, not its loop position.

Every continuation arm is evaluated with identical priors and allocations:

1. 512 generated images for the existing FID/KID, precision/recall, PR-F1,
   effective-rank, diversity, range, and memorization diagnostics;
2. four disjoint paired identity-Sinkhorn audits, each using an independent
   generated primary batch, independent generated self batch, real support,
   and real-real floor;
3. the frozen field energy

   ```text
   E = mean_i ||T_real(x_i) - T_self(x_i)||_2^2.
   ```

The paired B0-control and Sinkhorn calls use the same primary and self priors.
All solver residuals and cap hits are recorded.

## Unit gate

The Sinkhorn arm passes a unit only if all conditions hold:

- its field energy beats control in at least 3 of 4 paired audits;
- its median field energy is at most 75% of control's median;
- its recall is at least `control recall - 0.025`;
- its effective rank is at least 80% of control's;
- its PR-F1 is strictly greater than control's preregistered PR-F1;
- all collapse and memorization vetoes pass;
- every training and audit Sinkhorn plan converges without a cap hit;
- the matched Laplace arm's normalized kernel rows remain numerically valid;
- both correction arms execute exactly the number of events implied by their
  frozen cadence and the 5,000-update continuation;
- relative to the matched Laplace continuation, Sinkhorn PR-F1 is no more
  than 0.02 lower and effective rank is at least 90% as large.

PR-F1 is the single preregistered improvement axis; FID, KID, precision, and
recall remain reported diagnostics and cannot substitute after the run.
The Laplace comparison is a practical competitiveness check, not a claim that
the two corrections minimize identical population energies.

## Preflight and artifact boundary

Before full training, a real-GPU preflight must:

- hash-check B0, B2, S0, protocol, and executable sources;
- prove all arms load byte-identical B0 state dicts;
- prove the first flow batch is identical across arms;
- run at least two correction events for Laplace and Sinkhorn;
- record solver health, gradient ratios/cosines, memory, and time;
- inspect no generated image or quality metric.

The full runner writes one immutable unit shard after all three continuations
and evaluations complete. A restart may reuse only a shard compatible with
the corrected freeze. The invalid v1 freeze, log, and launcher are retained
only as an audit trail and are never accepted by v2.

## Claim boundary

`PROVISIONAL-GO` means the correction deserves the later geometry and
one-step MeanFlow stages. It does not establish encoder-independent
state-of-the-art generation, ImageNet scaling, one-step inference, or
superiority to the published drifting model.
