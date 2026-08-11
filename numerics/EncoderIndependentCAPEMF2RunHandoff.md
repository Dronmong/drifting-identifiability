# CAP-EMF-2 production-run handoff

> **Historical handoff.** The current concentrated campaign is fully specified
> by [`EncoderIndependentCAPEMF2ASFDProtocol.md`](EncoderIndependentCAPEMF2ASFDProtocol.md):
> one ordered-uniform foundation to 750k, gated ASFD continuation to 800k, then
> fixed evaluation. Its budget ceiling and post-foundation reserve are explicit
> admission inputs rather than the USD 50/USD 10 constants below.

**Last updated:** 2026-08-08  
**Canonical executable protocol:** `numerics/EncoderIndependentCAPEMF2ScreenProtocol.md`  
**Audit record:** `numerics/EncoderIndependentCAPEMF2CheckpointAudit.md`  
**Current state:** implementation and protocol audited; no download, cloud
training, or paid process is active

This document is the handoff for the next operator or agent. It explains the
entire experiment, its exact order, what is already complete, what must be
regenerated, when money may be spent, and what claims the result may support.
The canonical protocol contains the exact PowerShell commands. Do not rewrite
those commands from memory or create a second variant. If this handoff and the
canonical protocol ever disagree, stop, reconcile them, rerun the audit, and
regenerate all source-bound evidence.

## 1. Objective

CAP-EMF-2 is a developmental test of an encoder-free, one-call drifting model
on unconditional 32x32 CIFAR-10. It asks two ordered questions:

1. Can the Euler Mean Flow local finite difference be made numerically faithful
   to its exact directional derivative on the real trained model?
2. Once that numerical mechanism passes, does a better distribution over the
   two time endpoints improve image quality and stability relative to a
   concurrently trained legacy sampler?

This is not a general image-generation claim. It is one matched seed on the
CIFAR-10 training distribution, with no test-set selection. A successful run
would be evidence that the repaired package is worth confirming, not a final
benchmark result.

## 2. Frozen model and training mechanism

The experiment uses the same scientific architecture in every arm:

- raw RGB pixels; no separately trained semantic feature encoder;
- unconditional CIFAR-10, all 50,000 training images and all ten classes;
- a patch-2 U-ViT with AdaLN-Zero conditioning and a local pixel refiner;
- width 384, depth 12, 8 attention heads, 256 image tokens, approximately
  37.7 million parameters;
- direct-`x` Euler Mean Flow training under the repository's reversed clock;
- exactly one model call during generation;
- AdamW optimization at learning rate `1e-4`, effective batch 64,
  horizontal flips, gradient clipping, and EMA decay `0.9999`;
- matched deterministic seed `0` unless a later confirmatory protocol is
  separately designed.

The candidate numerical repair is
`local_1000_d0002_fp32`: sinusoidal scalar embedding scale 1000, local step
`delta = 0.0002`, and matched dense stopped evaluations with TF32 disabled for
the subtractive finite-difference path. It is a candidate, not an assumption.
The complete production-GPU quotient/target/gradient matrix must certify it
before training may begin.

## 3. The three matched arms

Only the time-pair sampler changes:

| Arm | Time-pair construction | Role |
|---|---|---|
| `legacy` | CAP logit-normal `t`, then `r=tU`, sampled floor `.01` | concurrent historical control |
| `ordered_logitnormal` | max/min of two iid CAP logit-normal draws | lower coefficient tail, weaker natural inference-corner coverage |
| `ordered_uniform` | max/min of two iid uniform draws | full triangular support and stronger inference-corner coverage |

All arms use the corrected fixed-count 50% diagonal mixture, independent RNG
streams, the same data order, augmentations, initialization, noise, optimizer,
model, numerical candidate, health gates, and evaluation populations. The
screen compares complete sampler packages; it does not separately identify
the causal contribution of ordering versus removal of the sampled-`r` floor.

## 4. Budget and storage contract

The operator-authorized all-in ceiling is now **USD 50**. This is a maximum,
not a target. The preflight must recompute

```text
conservative training = C150 + 2*C300
authorized upper cost = 1.15 * conservative training + $5
required condition    = authorized upper cost <= $50
```

Here `C150` and `C300` are cumulative one-arm costs measured by the genuine
production-GPU benchmark. The formula exactly prices three arms through 150k
and two arms through 300k:

```text
3*C150 + 2*(C300-C150) = C150 + 2*C300.
```

The USD 5 reserve covers startup, admission, transfer, storage/egress, and
evaluation work not represented by the training loop benchmark. The 15%
contingency applies to the conservative training projection. Consequently the
conservative training component itself must be no more than approximately
USD 39.13. A lower observed cost does not authorize extra arms, updates, seeds,
or model capacity.

Before rental:

- configure a provider-side hard spend/runtime cap of at most USD 50;
- provision one genuinely off-instance durable filesystem;
- provision approximately 200 GiB unless the measured storage preflight
  requires more;
- confirm the volume survives deletion of the GPU instance;
- account for storage and egress inside the USD 5 reserve.

The Python wall stop is secondary protection. It cannot prevent provider
billing while an instance is unreachable or stalled.

## 5. Trusted artifact layout

Choose one unused `$runTag`. The canonical protocol creates:

```text
$storageRoot/
  cap2-workspace-$runTag/
    evidence/
    evidence/production_gates/
    runs/
      cap2_legacy/
      cap2_ordered_logitnormal/
      cap2_ordered_uniform/
  cap2-durable-$runTag/
    benchmark/
    legacy/
    ordered_logitnormal/
    ordered_uniform/
```

The common workspace preserves the complete cross-arm authorization graph and
original relative paths. Each arm mirror is a second immutable rolling
recovery log. They serve different purposes and neither may be replaced by a
temporary instance disk.

Fresh run tags are mandatory. Never reuse an old immutable namespace, output
JSON name, or sidecar. All evaluation requirements, including the positive
control environment, must be installed before any baseline or calibration
artifact is generated.

## 6. Current readiness state

Already complete:

- source-level math and reversed-clock EMF implementation audit;
- recovery, terminal-finalization, storage, metric-leaf, and selection repairs;
- 170/170 CAP/CAP2 tests;
- a post-protocol 55/55 artifact/preflight/recovery subset;
- Ruff format/lint, Python compilation, CLI help, and diff checks;
- independent command, chronology, and release-adversary reviews.

Not complete and not reusable from the earlier attempt:

- frozen source commit for the paid attempt;
- fresh sampler and gate-calibration artifacts;
- fresh metric reference, real/real calibration, baseline, and positive control;
- production-GPU numerical admission;
- production checkpoint forensics;
- genuine disk-resume benchmark;
- aggregate compute/storage preflight;
- every 50k, 150k, and 300k training/evaluation artifact.

The source manifest and protocol changed during repair, so prior source-bound
JSON evidence is stale by design. Do not edit hashes or copy decisions into new
files.

## 7. Complete chronological run protocol

### Phase 0 — freeze before spending

1. Review the dirty worktree and preserve unrelated user work.
2. Commit or otherwise freeze the exact intended CAP/CAP2 source and protocol.
3. Record the commit hash and choose a new `$runTag` once.
4. Create and attest the durable workspace and run-tagged mirror root using the
   variable block in section 5 of the canonical protocol.
5. Install both `requirements-eval.txt` and
   `requirements-positive-control.txt` before generating any metric evidence.
6. Do not launch training in this phase.

Any source or protocol edit after evidence generation invalidates the evidence
and requires a fresh run tag.

### Phase 1 — regenerate local evidence

Run canonical sections A, C, and D:

1. extract and hash the common 50,000-image CIFAR-10 train KID reference;
2. generate the direct disjoint real/real CleanFID/CleanKID calibration;
3. regenerate the historical CAP-EMF-1 baseline in the frozen environment;
4. regenerate and validate the external positive control;
5. run the source-bound two-million-draw sampler audit;
6. calibrate the two-sided health gate from independent real subsets.

The baseline, control, calibration, and all candidates must use identical
package versions, image quantization, KID population, KID seed, metric batch
128, feature batch 128, workers 0, and compatible device/numerical provenance.

### Phase 2 — production-GPU admission without training

Provision the intended production GPU and run
`stage_cap2.production_readiness`, which performs canonical sections E, B, F,
and G in this order:

1. **Numerical admission:** compare the finite difference, assembled target,
   and parameter gradient with exact references across every declared time and
   input stratum. All checks must pass on the actual GPU.
2. **Checkpoint forensics:** record the preserved model's raw/clipped
   components, phase behavior, spectrum, and `(t,h)` responses without editing
   it.
3. **Benchmark:** exercise the full training loop, both production health
   paths, checkpoint/snapshot/recovery I/O, a real disk reload, and the next
   optimizer/EMA update. Measure runtime and artifact sizes.
4. **Preflight:** revalidate every source-bound input and require both compute
   and storage projections to fit the USD 50/volume contracts.

The readiness entry point has no training path. If it emits `NO_GO`, stop. Do
not raise thresholds or the budget simply to force admission. A different GPU
or provider requires rerunning hardware-bound evidence.

### Phase 3 — three arms to 50k

Provision and probe one fresh mirror namespace per arm. Run `run_screen` for
`legacy`, `ordered_logitnormal`, and `ordered_uniform`, each to exactly 50,000
updates using the same preflight, seed, effective batch, and durable workspace.

For every arm:

1. require a complete 50k result and durable recovery commit;
2. rerun the full numerical matrix on the exact 50k raw checkpoint;
3. build `early_admission_50000.json`;
4. stop that arm if its early certificate is not `GO`.

No arm may jump directly from fresh initialization to 150k.

### Phase 4 — valid arms to 150k

Continue each valid arm in the same output directory and recovery stream, with
its immutable 50k early-admission certificate. At 150k:

1. evaluate the declared EMA checkpoint on 50,000 generated samples;
2. retain all sequential PNGs, their manifest, the generated clean-Inception
   feature archive, the shared reference archive, and the fixed grid;
3. rerun numerical admission on the exact raw checkpoint that would continue;
4. create one immutable promotion certificate per arm.

For `legacy`, `--allow-valid-legacy-control` may make the command return
success when its separate control-continuation record is `GO`. It does not
rewrite a quality `NO_GO` and never waives integrity, numerical, health,
collapse, or evaluation checks.

### Phase 5 — select one ordered arm

The selection command consumes all three promotion records. It always retains
the concurrently trained legacy arm if its control-continuation certificate is
valid. It selects at most one ordered arm, and only if that arm:

- has an individual promotion `GO`;
- beats concurrent legacy CleanFID beyond the observed real/real FID margin;
- beats concurrent legacy CleanKID beyond the observed real/real KID margin.

If both ordered arms qualify but are indistinguishable within both calibrated
margins, select neither. Auxiliary 2,048-sample KID/precision/recall metrics may
not break the tie or rescue an arm. If selection is `NO_GO`, stop the campaign
at 150k and report that result.

### Phase 6 — paired continuation to 300k

Mechanically reload and revalidate the selection. Continue exactly two arms:

1. `legacy`, as the concurrent control;
2. the verified ordered winner.

Both continue from their own exact 150k optimizer, raw model, EMA, RNG,
counter, health, checkpoint, and snapshot state. No restart-from-weights and no
new seed is allowed. The runner refuses horizons beyond 300k.

### Phase 7 — terminal evaluation and paired verdict

For both 300k arms:

1. run a fresh production-GPU numerical readmission on the raw checkpoint;
2. run the 50,000-sample development evaluation on the EMA checkpoint in the
   exact frozen metric environment;
3. preserve the complete PNG and clean-feature populations;
4. build `cap2_final_verdict_300k.json` with both durable mirror roots;
5. invoke `final_verdict --revalidate` to reload references and independently
   recompute the retained CleanFID/KID leaves.

A terminal `GO` requires an intact 150k authorization ladder, strict recovery
identity, numerical readmission, health/collapse validity, historical and 150k
quality retention, and a margin-separated ordered win over concurrent legacy
on both primary standard metrics.

## 8. Recovery procedure

Normal resume uses the same arm root, workspace, mirror namespace, profile,
seed, preflight, and horizon command. Never rename a recovered arm to a sibling
path because promotion and selection references are relative to the common
layout.

If direct resume reports a stale/future/orphaned or immutable collision:

1. remount and probe the common workspace;
2. move the damaged arm tree to the run-tagged damaged directory;
3. restore the greatest complete recovery commit from that arm's mirror into
   the original empty arm path;
4. verify the mirror before adding any unmirrored files;
5. copy back `eval_150k_pngs` and `eval_150k_grid.png` from the move-aside tree
   when present and absent from the restored tree;
6. revalidate any existing concurrent selection;
7. resume only after every check succeeds.

Incomplete future transactions may be quarantined and recomputed. A conflicting
complete committed pair must fail closed. If the common durable workspace is
missing after remount, abort: the per-arm mirror is not a complete backup for
the authorization graph.

## 9. Stop conditions

Stop immediately on any of the following:

- numerical admission or readmission `NO_GO`;
- source, protocol, environment, hardware, metric-population, or hash mismatch;
- benchmark resume-continuity failure;
- aggregate cost above USD 50 or inadequate durable capacity/headroom;
- provider hard cap unavailable;
- non-finite training update or binding health/collapse failure;
- failed 50k early admission;
- no ordered arm selected at 150k;
- broken recovery, missing shared workspace, or immutable artifact conflict;
- terminal paired verdict `NO_GO`.

Do not reinterpret a stopped run using visual samples or auxiliary metrics.
Preserved `NO_GO` evidence is a valid experimental result.

## 10. Result interpretation

The strongest permissible successful statement is:

> On one matched developmental seed for unconditional 32x32 CIFAR-10, the
> selected encoder-free, one-call ordered-time CAP-EMF-2 arm passed the frozen
> numerical, health, collapse, recovery, and metric-integrity gates and beat a
> concurrently trained legacy arm on retained 50k-sample CleanFID and CleanKID
> beyond the predeclared real/real discrepancy margins at 300k updates.

Do not claim replication, CIFAR-10 test performance, high-resolution
scalability, broad semantic competitiveness, proof that encoders are
unnecessary, or superiority to the original paper from this screen alone.

If the ordered arm wins, the next step is a separately frozen multi-seed
confirmation. If no ordered arm wins and the patch-head/refiner pathology
persists, the next intervention is decoder/stability architecture—not another
reinterpretation of sampler metrics.

## 11. Next-agent checklist

- [ ] Read this handoff, the canonical protocol, and the checkpoint audit.
- [ ] Inspect `git status`; preserve unrelated user work.
- [ ] Freeze/commit the intended source and record its hash.
- [ ] Confirm the protocol contains `--max-total-cost 50` and
      `--nontraining-reserve 5` in both preflight paths.
- [ ] Choose a fresh run tag and provision durable off-instance storage.
- [ ] Set a provider hard cap no greater than USD 50.
- [ ] Regenerate local source-bound evidence in the frozen metric environment.
- [ ] Run production readiness; do not train unless its aggregate result is
      `GO`.
- [ ] Follow the 50k → 150k → selection → paired 300k ladder exactly.
- [ ] Preserve and revalidate the final paired verdict.
