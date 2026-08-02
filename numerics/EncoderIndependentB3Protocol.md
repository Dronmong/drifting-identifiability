# Stage B3 protocol — matched encoder-free drifting reference

**Status: IMPLEMENTED; hash-bound GPU preflight GO. Long run not executed.**

B3 is a reference measurement for the repository's R11-corrected, raw-pixel,
smooth-Laplace, one-step drifting proxy. It is not the paper's complete model
and it is not a competitive gate. Its purpose is to put that proxy and the
frozen B0/B1/B2 bridge checkpoints on the same evaluation instruments.

## 1. Question and scope

Earlier drifting results and B0/B1/B2 results used different reference
allocations. B3 asks:

> Under identical reference images, sample counts, metric implementations, and
> drift-audit roles, how does the one-step drifting proxy compare with the
> already-frozen B0, B1, and B2 models?

The answer is descriptive. There is no `PASS` category. Three training units
give a coarse consistency check, not high-powered inference.

Phase 30 found **very low**, not literally always-zero, recall: the w64/p64
median was 0.004 (0.009/0.000 by seed), and w128/p256 reached 0.044 in one seed
but 0.000 in the other. B3 must not rewrite those values as exact zeros.

## 2. Arms and the capacity repair

Both arms use the same field cloud and the same exact-memory trainer:

| arm | generator | parameters | field cloud | backward microbatch |
|---|---|---:|---:|---:|
| `B3-native` | `OneStepGenerator`, width 64 | 146,691 | 256 | 256 |
| `B3-capacity` | `OneStepGenerator`, width 368 | 3,864,003 | 256 | 256 |

The frozen bridge has 3,893,443 parameters, so the capacity arm is within one
percent. Width and field-cloud size are no longer changed together.

The memory repair is exact for this objective. At each update:

1. evaluate all 256 latent samples under `no_grad`;
2. compute the full R11 teacher from that full cloud;
3. re-evaluate the same 256 latent samples in one full-cloud backward pass;
4. backpropagate the full-cloud **sum divided by 256**;
5. take one Adam update.

The teacher is detached. `OneStepGenerator` has GroupNorm, no BatchNorm,
dropout, or stochastic layer, so each sample's output is independent of a
chunk partition. The selected 256-sample backward pass is exactly the
full-cloud mean-loss gradient. The generic exact chunk-capable trainer remains
covered by a regression that compares loss, every parameter gradient, and one
complete optimizer update against the full-batch path.

## 3. Frozen training recipe

| item | value |
|---|---:|
| units | 600, 601, 602 |
| steps | 30,000 |
| checkpoints | 10,000 / 20,000 / 30,000 |
| geometry | raw pixels |
| kernel | `smooth_laplace` |
| legacy calibration target | 0.05 |
| field direction | paper |
| field normalization | RMS |
| R11 teacher | scalar second-moment correction |
| drift step η | 0.5 |
| positives | 64 |
| optimizer | Adam |
| peak learning rate | 2e-3 |
| schedule | cosine to zero |

Only step 30,000 is primary. The earlier checkpoints diagnose whether rank
compression accumulates; they cannot replace the final result after inspection.

The nominal ESS value is a legacy calibration label. B3 records the calibrated
bandwidth, realized rectangular ESS, and dead-row fraction at declared
checkpoints. It must not report 0.05 as the realized neighborhood ESS.

## 4. Randomness and calibration

Every stochastic role has a distinct labelled seed stream. Within a unit, the
two B3 arms share:

- target calibration indices and the resulting kernel;
- target minibatch indices at every step;
- latent samples at every step;
- evaluation latents and drift-audit latents.

Model initialization uses the same labelled seed but naturally produces
different tensors for the two widths. Diagnostic and bootstrap streams are
separate from training. The artifact records the seed derivation manifest,
calibration indices digest, kernel parameters, and source hashes.

## 5. Evaluation instruments

The two B2.5 instruments are reused **separately and never pooled**:

1. `in_domain_development_reused`: official CIFAR-10 test images. This source
   was adaptively used in B1 and is development-only.
2. `shifted_disjoint`: the hash-bound CINIC-10 ImageNet-only B2.5 pool. It has
   zero decoded-pixel overlap with complete CIFAR-10 and zero path/pixel
   overlap with B2's earlier external pool. Once B3 uses it, it remains a
   development source rather than a fresh confirmation source.

The same allocation object supplies the reference, matched-real control,
probe, positive, and real-real-floor roles for every B3 arm and every frozen
B0/B1/B2 checkpoint. Existing checkpoints are re-evaluated; they are not
retrained.

## 6. Metrics and comparison rules

Each source reports recall, precision, KID, density, coverage, indicative
fixed-sample FID, raw-pixel effective rank, normalized-Laplace drift energy at
the frozen B2 bandwidth, wall time, peak memory, parameter count, and forward
counts. The normalized-Laplace energy is a common diagnostic axis; it is not
the bi-softmax objective that B3 trains.

The two B3 arms use the same evaluation latents, so paired generated-sample
intervals are valid for their contrast. B3 and bridge models have different
latent spaces and samplers. Cross-architecture intervals therefore resample
their generated sets **independently** while sharing the same reference
subsample. Generated-index pairing is forbidden for those comparisons.

Report raw unit values and medians. With three units, do not attach a strong
significance claim to small differences.

## 7. Phase-30 compatibility and checkpoint integrity

The preflight includes a deterministic equation-level regression against the
Phase-30 update: the legacy inline full-cloud teacher/loss, the new full-cloud
helper, and the new microbatch path must agree. It also records which files in
the old Phase-30 provenance have changed; source-hash drift is not silently
treated as exact historical replay.

B3 saves raw (non-EMA) checkpoints, matching Phase 30. Every checkpoint records
its unit, arm, width, step, protocol/preflight hash, state dictionary, and
SHA-256. Loading rejects any mismatch.

## 8. Preflight and execution order

The long run is forbidden until preflight returns GO. The required order is:

1. run unit tests, including full-vs-microbatch loss/gradient/update equality;
2. verify B2.5 source provenance and disjoint allocations;
3. verify and load frozen B0/B1/B2 artifacts;
4. freeze the three unit-specific target-only calibrations;
5. run a short native and capacity smoke test;
6. measure capacity-arm peak reserved memory and throughput on the actual GPU;
7. hash protocol, executable sources, data, prerequisites, allocations, and
   preflight artifact;
8. only then train units 600–602.

The prior seven-hour estimate is retired. Runtime and headroom must come from
the corrected width-368/cloud-256 full-backward preflight.

The optimization audit's 100-update RTX 4050 Laptop GPU probe found:

| arm | reserved memory | seconds/update | projected hours per 30k unit |
|---|---:|---:|---:|
| `B3-native` | ~0.71 GB | ~0.054 | ~0.45 h |
| `B3-capacity` | ~5.24 GB | ~0.81 | ~6.77 h |

The six-run training projection is therefore about **21.65 GPU-hours**, before
evaluation. The hash-bound preflight repeats a 100-update probe; the final
artifact reports measured wall time.
The hash-bound preflight artifact and its authoritative SHA-256 sidecar live
under `stage_b3/b3_preflight.json`.

The runner atomically saves optimizer, model, target-RNG, history, elapsed-time,
and peak-memory recovery state every 1,000 updates. Restarting the same unit
continues from the last verified recovery point; completed final artifacts are
never overwritten. Deterministic CUDA execution requires and records
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

After the checked preflight is present, the complete three-unit experiment and
aggregation can be handed off as one command from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File `
  numerics/encoder_independent_drifting/stage_b3/run_overnight.ps1
```

This launcher does not shorten the experiment. It runs both arms for all
30,000 updates on units 600, 601, and 602, then aggregates the results.

## 9. Licensed conclusions

B3 may support matched descriptive statements about this repository's
one-step drifting proxy versus B0/B1/B2. It does not establish that the paper's
model fails, that a bridge wins generally, or that encoder independence alone
causes any observed difference. The in-domain and shifted results retain their
separate development scopes.
