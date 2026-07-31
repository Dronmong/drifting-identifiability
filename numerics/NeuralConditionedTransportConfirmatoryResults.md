# Neural conditioned-transport confirmatory results

**Status:** preregistered confirmation passed  
**Authoritative artifact:**
`neural_conditioned_confirmatory_runs/20260722-172943-confirmatory/`  
**V3 registry SHA-256:**
`452574697ed8f1ebac28db6c2dec94f239f13f4b5067e21d67f08d419d4806a4`  
**V3 source/atlas freeze SHA-256:**
`37cd199eeb8c9b1179c9a637cd20922a70e52c2dde3a22e0e6d6f2ed38589746`

## Outcome

Both preregistered result families passed every gate:

```text
deterministic exact conditioned transport primary     PASS (9/9 gates)
frozen KLL conditioned transport retention arm        PASS (5/5 gates)
divergences across all five algorithms                0
```

The audit contains all `640` unique target/initialization/arm cells, `640`
saved outputs, and `64` disjoint evaluation references.

## Primary comparisons

The independent unit is one of 64 new target instances. Concentrated and broad
initializations were first reduced within target. Intervals are the frozen
5,000-draw bootstrap, stratified over the 16 dimension/family cells.

| Exact primary comparison | Geometric-mean ratio | 95% interval |
|---|---:|---:|
| ED2 vs paper neural | **0.3189** | **[0.2916, 0.3492]** |
| held-out SW1 vs paper neural | **0.5517** | **[0.5277, 0.5772]** |
| ED2 vs baseline envelope | **0.4069** | **[0.3703, 0.4459]** |
| held-out SW1 vs baseline envelope | **0.6421** | **[0.6137, 0.6711]** |

The baseline envelope is the best target-wise value among paper neural,
minibatch sliced-Wasserstein, and previous small KLL-RSR. Ratios below one
favor conditioned transport.

The exact primary beat paper on both metrics in `62/64` targets and beat the
stronger baseline envelope on both in `61/64`. The frozen gates required only
`48/64` and `40/64`, respectively.

## Absolute endpoint scale

These are medians over all 128 target/initialization cells. They are descriptive
only; promotion used target-reduced ratios above.

| Arm | Median ED2 | Median held-out SW1 |
|---|---:|---:|
| paper neural | 0.04366 | 0.14738 |
| minibatch SW | 0.04731 | 0.14820 |
| previous KLL small RSR | 0.04237 | 0.14274 |
| exact conditioned + local | 0.01646 | 0.08210 |
| frozen KLL conditioned + local | **0.01628** | **0.08191** |

## Dimension robustness

Target-reduced medians show that the former 16D reversal is absent:

| Dimension | Paper ED2 / SW1 | Exact conditioned ED2 / SW1 |
|---:|---:|---:|
| 2 | 0.03377 / 0.16610 | **0.00735 / 0.07329** |
| 4 | 0.04230 / 0.15895 | **0.01046 / 0.07428** |
| 8 | 0.04172 / 0.12648 | **0.01947 / 0.08369** |
| 16 | 0.07674 / 0.14997 | **0.02743 / 0.08726** |

All 16 dimension/family cells favored exact conditioned transport on ED2. The
least favorable cell was 16D balanced GMM, with ratio `0.8899`; the frozen
robustness ceiling was `1.10`. The other 15 cell ratios ranged from `0.1271`
to `0.6822`.

The gain also survived initialization. Exact/paper geometric ratios were
`0.3123 / 0.5442` for ED2/SW1 under concentrated initialization and
`0.3245 / 0.5591` under broad initialization.

## KLL retention

The KLL arm used payloads serialized and hashed before model training. Against
the deterministic exact primary:

| KLL/exact comparison | Ratio | 95% interval |
|---|---:|---:|
| ED2 | **0.9697** | **[0.9443, 0.9943]** |
| held-out SW1 | **0.9922** | **[0.9810, 1.0034]** |

Every KLL retention gate passed. The worst cell-level ED2 ratio was `1.1377`,
below the frozen `1.25` ceiling. KLL also beat paper on both primary metrics in
`62/64` targets. This establishes retention for this one frozen sketch
realization; it does not erase the KLL compaction variability observed during
development.

## Rare-mode result

Across the 16 fresh rare-GMM targets:

| Arm | Median mode coverage | Median rare-mass error |
|---|---:|---:|
| paper neural | 0.50 | 0.02519 |
| exact conditioned + local | **1.00** | **0.01002** |
| frozen KLL conditioned + local | **1.00** | **0.00999** |

The exact rare-mass error is about `39.8%` of paper's, comfortably below the
preregistered `80%` ceiling. Thus the aggregate quality gain was not purchased
by dropping the 2%, 5%, or 10% minority components.

## Cost interpretation

All arms used exactly 20,480 generator-example evaluations and 320 generator
forward calls. Candidate arms used 160 Adam updates and 10,240 unique latents,
matching previous RSR accounting. Ordinary paper/minibatch baselines used 320
updates and 20,480 unique latents.

The new algorithm is not cheaper than paper:

- it accesses 30,720 target examples when the persistent atlas and local
  batches are both counted, versus 20,480 for paper;
- its local field uses 10,485,760 kernel pairs per run, four times paper's
  2,621,440; and
- it additionally pays direction-dependent projection and sorting work.

Median CPU training time was approximately `0.53 s` for either conditioned
arm, `0.37 s` for paper, `0.54 s` for minibatch SW, and `0.27 s` for small
KLL-RSR. These small CPU timings are implementation-specific. The confirmed
claim is quality at matched generator-example count, not speed or total-FLOP
superiority.

## Failed pre-runs and trust boundary

V1 and V2 produced no reported endpoint result and are excluded. V1 caught a
roundoff-sensitive atlas replay check; V2 caught a heterogeneous CSV-schema
bug after training but before result serialization. Both were abandoned under
the frozen contamination rule, repaired, logged in
`NeuralConditionedTransportFailures.md`, and replaced with a new master seed.
Before v3, the repaired atlas check passed all 64 abandoned v1 targets and a
full heterogeneous 640-row artifact fixture passed end-to-end.

The scoped conclusion is therefore:

> On the preregistered 2D/4D/8D/16D synthetic neural-generator benchmark,
> conditioned transport-then-amortize gives a robust general quality
> improvement over the repository paper-field port and over the target-wise
> best competing baseline at matched generator-example count. A frozen KLL
> atlas retains the improvement.

This does **not** establish improvement over the original paper's image
benchmarks, encoder-space metrics, training FLOPs, memory, or wall-clock speed.

