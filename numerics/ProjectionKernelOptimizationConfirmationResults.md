# Projection/kernel optimization confirmation results

## Verdict

The frozen active-32, exact-atlas, `M = 128` conditioned-transport arm passed
the predeclared low-dimensional confirmation against the matched current
paper-neural comparator.

This is the strongest defensible result:

> On 16 newly generated synthetic neural-generator targets in dimensions
> 2, 4, 8, and 16, the frozen `M = 128` arm improved both ED2 and held-out
> SW1 relative to the current optimized paper-model port at equal
> generator-example count and equal exact kernel-pair count. It was also
> faster in this CPU implementation, even after its direction/atlas setup was
> charged. This does not establish an image-scale or universal advantage.

The protocol was frozen in
`ProjectionKernelOptimizationConfirmationProtocol.md`. The complete
machine-readable analysis is
`projection_kernel_confirmation_analysis.json`, with SHA-256
`99f7c4050e63e04db015b4af84d38cd9edc54d9b53a526ba0c3e5815b84e64e7`
recorded in its adjacent sidecar.

## Repaired comparison

The earlier development comparison borrowed the paper row from a historical
runner that used a different field implementation and timing context. It was
valid as historical development provenance but not as a wall-clock
comparison. The repaired runner now:

- executes the current optimized paper field in the same process;
- uses the same frozen target, initialization, latent, and evaluation data;
- gives every arm 20,480 generator-example evaluations;
- randomizes arm execution order independently for each target;
- separates online training from direction/atlas setup;
- uses one registered `tau = 1` entry point for dense and representative
  fields;
- runs the field and actual-runner `M = B` equivalence gates;
- records source, registry, Git, result-payload, and KLL-state hashes; and
- independently recomputes ED2 and held-out SW1 from saved outputs.

The four earlier bandwidth-confounded directories are now visibly quarantined
with an `-INVALID-TAU` suffix. A process-timeout directory is similarly marked
`-INVALID-TIMEOUT`.

## Frozen artifacts

The new registry has SHA-256
`3a148bfa4c427fc26e44c95c43b213efda13f79b87d329f37c5f7f0ccff44c05`.

| Setting | Artifact | Deep audit |
|---|---|---|
| pre-outcome primary `M = 128` | `conditioned_transport_runs/20260722-201212-consumed` | pass |
| canonical reporting-only primary rerun | `conditioned_transport_runs/20260722-202659-consumed` | pass |
| dense active-32, `M = 512` | `conditioned_transport_runs/20260722-201428-consumed` | pass |
| aggressive active-32, `M = 64` | `conditioned_transport_runs/20260722-201628-consumed` | pass |

The first primary artifact proved the protocol was frozen before endpoints
were inspected. Its aggregate summary encoded undefined support metrics as
non-standard JSON `NaN`; the underlying rows and deep audit were valid. After
fixing only that finite-value aggregation, the canonical rerun reproduced all
16 exact-hybrid and all 16 paper output arrays bit-for-bit. It supplies the
reported timing and strict JSON payload.

Every listed artifact contains 96 rows and 96 saved outputs. Maximum discrepancies
when recomputing the saved endpoints were below `1.98e-6` for ED2 and
`1.28e-9` for SW1. The small ED2 tolerance accounts for saving outputs and
references as float32 after evaluating the original float64 reference.

## Primary quality result

Ratios are paired within target; lower than one favors the frozen arm.
Confidence intervals are target-level percentile-bootstrap intervals with
20,000 resamples and frozen seed `2026081709`.

| Metric | Geometric ratio vs paper | 95% interval | Paired median ratio | Wins |
|---|---:|---:|---:|---:|
| ED2 | **0.3305** | **[0.2520, 0.4313]** | 0.3863 | 15/16 |
| held-out SW1 | **0.5711** | **[0.4969, 0.6545]** | 0.6479 | 16/16 |

Both upper confidence limits are below one and both win counts exceed the
predeclared 12/16 threshold. The one ED2 loss was the 16D balanced GMM
(`1.037x`); its held-out SW1 still improved (`0.918x`). Thus the result is
strong but not an every-target, every-metric dominance claim.

Raw medians were:

| Arm | ED2 | held-out SW1 |
|---|---:|---:|
| active-32, `M = 128` exact hybrid | **0.011872** | **0.086622** |
| matched optimized paper | 0.047305 | 0.149031 |

## Cost result

| Cost scope | Paired median ratio vs paper | Interpretation |
|---|---:|---|
| generator-example evaluations | 1.000 | exactly matched |
| exact kernel pairs | 1.000 | exactly matched |
| online training wall | **0.6150** | about 38% lower |
| setup plus online training wall | **0.7636** | about 24% lower |
| target-example accesses | 1.500 | conditioned transport uses more |

The setup-inclusive scope charges conditioned-direction construction and the
exact target-atlas build, but excludes target generation and final evaluation
for both arms. The wall result is a CPU implementation measurement, not a
hardware-independent complexity theorem. Peak memory was not measured and no
memory-superiority claim is made.

The primary run also executed two dense field-audit calls per local arm.
Those diagnostic pairs are excluded from the algorithmic pair ledger but their
time remains in the measured training wall, making the reported wall advantage
conservative for the deployed `M = 128` path.

The target-access disadvantage remains real: the candidate reads the whole
20,480-point pool to build its persistent atlas and another 10,240 examples
for local fields. Its persistent exact-atlas payload has additional storage
and it performs projection/sorting work absent from the paper comparator.

## Did compression itself help?

Against the separately launched dense active-32 arm:

| Setting vs dense | ED2 geometric / median | SW1 geometric / median | Online wall median | Pair ratio |
|---|---:|---:|---:|---:|
| `M = 128` | **0.9486 / 0.9979** | **0.9769 / 0.9990** | **0.9937** | **0.25** |
| `M = 64` | 0.9984 / 1.0299 | 1.0020 / 1.0219 | 1.0072 | **0.125** |

`M = 128` was essentially quality-neutral by the paired median and slightly
better by the geometric mean, while removing 75% of local kernel pairs. Its
measured wall gain over dense was only about 0.6%, so the tree construction
and other fixed costs absorb most of the pair-count reduction on this CPU
test. That difference is too small to distinguish confidently from launch
noise; the robust result versus dense is the exact 4x pair reduction, not a
wall-speed claim. `M = 64` removed more pairs but was neither faster nor more
accurate by the typical paired result. It remains an ablation, not the
promoted setting.

The direct `M = 128` field audits had median relative-L2 error `0.0804` and
minimum cosine `0.8915`. Median row/column mass relative-L2 errors were
`0.0389 / 0.1087`; the largest target-level means were `0.3295 / 0.3116`.
These diagnostics explain why compression is not treated as exact below
`M = B`.

## Rare modes and representative semantics

Both the candidate and paper comparator had minimum mixture-mode coverage
`0.5`; neither universally prevented mode loss. The candidate's median
rare-component mass error was `0.0240`, compared with `0.0500` for the paper
comparator, but both missed the rare component on the 16D rare-GMM cell.

The projection tree is balanced by sample occupancy. At `M = 128`, every leaf
contains exactly four observations. That prevents empty or oversized
representative leaves, but it is not a semantic guarantee: the tree never sees
mixture labels and can merge a rare component with a neighboring common
region. Claims of a guaranteed minimum allocation to every unknown mode have
therefore been removed.

## Remaining limits

- The experiment is synthetic and low-dimensional; it does not validate
  images, learned encoder features, or dimensions above 16.
- Each target cell has one frozen realization. The target-level bootstrap
  quantifies heterogeneity across these 16 cells, not all possible training
  randomness.
- Equal generator-example and kernel-pair counts do not mean equal total
  operations. The full projection, sorting, storage, target-access, and wall
  ledgers must remain visible.
- The endpoint advantage is for this neural-generator benchmark and current
  repository paper port, not a reproduction of the paper's ImageNet metrics.
- Peak memory and GPU throughput remain unmeasured.

The next scientifically useful step is a separately frozen high-dimensional
or real-feature study, not more selection on either the development or this
confirmation registry.
