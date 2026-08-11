# CAP-EMF-2 numerical repair and matched sampler screen

> **Superseded for the concentrated 750k -> ASFD run.** This document remains
> the historical matched-screen specification. The executable run card is
> [`EncoderIndependentCAPEMF2ASFDProtocol.md`](EncoderIndependentCAPEMF2ASFDProtocol.md).
> Do not mix commands, budget assumptions, or selection claims between them.

**Status:** implementation checkpoint audited; all source-bound evidence must be
regenerated before production-GPU admission
**Purpose:** repair the two evidenced CAP-EMF-1 defects before another paid run
**Maximum scope:** developmental units up to 300,000 updates; no ASFD and no
full confirmation
**Budget ceiling:** USD 50 total, including a USD 5 non-training reserve and
15% contingency; this is a hard maximum, not a spending target

## 1. Questions

CAP-EMF-2 separates two defects that CAP-EMF-1 bundled:

1. Is the stopped finite difference local and numerically aligned with its
   exact directional JVP on the trained model?
2. Once a numerical setting passes, which ordered time-pair distribution best
   controls the coefficient tail while covering the one-step inference corner?

No image-quality trial may start until question 1 passes on the production GPU.

## 2. Implemented repairs

- The time-embedding scale is explicit and its maximum phase move is recorded.
- `delta`, sampled `r`, and denominator clamping are separate knobs.
- Stopped evaluation supports:
  - the historical active-row gather;
  - matched full-batch evaluation;
  - matched full-batch stopped evaluation with TF32 disabled.
- The earlier ordered-iid sampler has been restored in logit-normal and uniform
  forms.
- The 50% diagonal mixture uses an independent RNG, an exact per-batch count,
  and the unsorted first endpoint. This preserves the intended endpoint
  marginal instead of biasing diagonal rows toward the sorted maximum.
- Every example contributes to `t/r/h`, coefficient, target, weighted-loss,
  and output-gradient sufficient statistics.
- Absolute-`t` endpoint bins now include `.98`, `.99`, and `.995`.
- Health records separate the patch-head base, refiner residual, and final
  output for raw and EMA weights.
- Gates are two-sided and include raw saturation.
- Recovery preserves true final-window clipping counters.
- A recovery file cannot resume into a different arm profile.
- Standard evaluation writes and hashes all generated PNGs, uses CleanFID's
  published CIFAR-10 train FID moments, and uses one hash-sealed extraction of
  all 50,000 train images as the common KID reference population.
- CAP2 artifacts use an explicit source manifest that includes executable
  runners.
- Every authorization artifact is written atomically with a SHA-256 sidecar;
  checkpoints, snapshots, recovery, run identity, and promotions are bound to
  the exact profile, seed, hardware, and numerical environment.
- Numerical admission, the full-loop benchmark, and screen training are bound
  to the same predeclared GPU model; generic `cuda` availability is not enough.
- A read-only checkpoint-forensics artifact records raw/clipped component,
  patch-phase, spectrum, and `(t,h)` response diagnostics before training.
- A disjoint train/train CleanFID calibration supplies a predeclared observed
  discrepancy margin rather than an assumed noise floor. It is one
  deterministic discrepancy point,
  not a variance estimate or confidence interval.
- CAP-EMF-1 and CAP2 use the same fixed auxiliary feature evaluator, reference
  subset, memorization audit, generation count, and seeds.
- Every checkpoint records the same fixed 2,048-row exact inference endpoint
  `(t,r,h)=(1,0,1)` under both raw and EMA weights. Naturally sampled endpoint
  occupancy remains diagnostic; it is not an impossible shared arm gate.
- Recovery rehearsal now stops, reloads model/optimizer/EMA/RNG state from disk,
  and performs the next update. Paid recovery files are synchronously committed
  to explicitly attested, instance-independent storage as immutable versions.
- The preflight freezes a worst-case aggregate screen budget, not merely a
  per-arm estimate, and the runner enforces a conservative wall-time stop only
  after publishing a verified recovery.
- Every development evaluation retains the exact 50,000 generated PNG
  population, the full CleanFID feature population, and the exact KID reference
  archive so the terminal scores can be independently revalidated.

## 3. Numerical candidates

| Name | Embedding scale | `delta` | Stopped path | Status |
|---|---:|---:|---|---|
| `legacy_1000_d01` | 1000 | .01 | dense TF32 | known negative control |
| `local_1000_d0002_fp32` | 1000 | .0002 | dense stopped FP32 | predeclared candidate; full GPU admission pending |
| `smooth_100_d001_fp32` | 100 | .001 | dense stopped FP32 | architecture candidate; must first train a short checkpoint |

The preserved checkpoint supports *testing* `local_1000_d0002_fp32`, but does
not yet certify it: the older narrow audit passed only 5 of 9 checks. A narrow
exploratory sweep made `.0001` look better, but `.0001` is not promoted unless
it independently passes the complete matrix below. The checkpoint cannot
certify `smooth_100_d001_fp32`, because
changing the embedding changes the function represented by the trained model.

Numerical GO requires, in every declared stratum:

- quotient/exact-JVP minimum cosine at least `.98`;
- quotient relative RMS error at most `.15`;
- assembled-target cosine at least `.995` and relative RMS error at most `.10`;
- parameter-gradient cosine at least `.95`;
- parameter-gradient relative L2 error at most `.20` and norm ratio in
  `[.85, 1.15]`;
- all 63 homogeneous batches from three repeats, three input sources, and seven
  time strata,
  including exact `(t,r)=(1,0)` and low-`t` stress, pass;
- all nine heterogeneous 4-row gradient batches and all nine exact production-
  shape 16-row batches pass, with half diagonal and half active rows;
- CUDA execution with gradient comparison enabled.

The seventh stratum is explicitly below the numerical floor (`t=.01,r=0`).
It guards a repaired clock-substitution detail: in the reversed Eq. 18 factor
`(t-r-delta)_+ t/r`, only the denominator `r` is clamped. The multiplicative
`t` is a numerator; clamping it would overweight precisely the rows that also
receive the largest `1/t^2` regression weight.

CPU audits are mechanical only and always emit `NO_GO`.

## 4. Sampler arms

All arms use the numerical candidate admitted in section 3.

| Arm | Construction | Sampled `r` floor |
|---|---|---:|
| `legacy` | CAP logit-normal `t`, then `r=tU` | .01 |
| `ordered_logitnormal` | two iid CAP logit-normal draws, max/min | 0 |
| `ordered_uniform` | two iid uniform draws, max/min | 0 |

Every arm retains the same 50% exact diagonal mixture. Ordered arms clamp only
the coefficient denominator, not the sampled condition.

The three-arm screen is a historical-control comparison, not a complete
factorial. The `legacy` arm retains both CAP's conditional sampler and its
sampled-`r` floor, while the ordered arms change both. Therefore an ordered-arm
win supports the repaired package but cannot by itself attribute the gain
uniquely to ordering versus removal of the sampled floor.

The source-bound two-million-draw audit was regenerated after the corrected
diagonal semantics and expanded source boundary. The canonical artifact in
step C returned `GO` with the following audited values:

| Arm | q99 coefficient | `P(c>7)` | `P(t>.95,h>.90)` |
|---|---:|---:|---:|
| legacy | 21.3881 | 4.2219% | 0.00985% |
| ordered logit-normal | 1.4290 | 0.00245% | 0% observed |
| ordered uniform | 22.3627 | 4.05225% | 0.37055% |

This makes the tradeoff explicit. Ordered logit-normal almost eliminates the
large-coefficient tail but misses the long inference corner. Ordered uniform
restores the corner at the price of a coefficient tail as heavy as CAP's. It is
not assumed in advance that either property dominates after the local-difference
repair; that is the reason both arms remain in the controlled screen.

Within the declared package comparison, model initialization, data order, noise seeds,
augmentation, batch, optimizer, learning rate, adaptive loss, numerical
candidate, refiner, and EMA fixed.

## 5. Mandatory admission sequence

All commands below run from the repository root. Cloud commands should use the
same pinned CUDA/PyTorch environment intended for training.

Choose one fresh run tag once and retain these variables in the PowerShell
session. Never point a new audit at a previously consumed immutable filename:

```powershell
$runTag = "20260808_a"  # replace once for the actual attempt
$storageRoot = "X:\"  # one durable filesystem for workspace + arm mirrors
$workspace = "X:\cap2-workspace-$runTag"
$mirrorRoot = Join-Path $storageRoot "cap2-durable-$runTag"
New-Item -ItemType Directory -Force -Path $workspace | Out-Null
New-Item -ItemType Directory -Force -Path $mirrorRoot | Out-Null
python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror provision `
  --mirror-dir $workspace `
  --storage-id "provider-volume-id/workspace-$runTag" `
  --i-attest-instance-independent-storage
python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror probe `
  --mirror-dir $workspace
$evidence = "$workspace/evidence"
$gates = "$evidence/production_gates"
$runs = "$workspace/runs"
$kidReference = "$evidence/cifar10_train_clean_features.npz"
$metricCalibration = "$evidence/metric_calibration.json"
$baselineStandard = "$evidence/baseline_cleanfid.json"
$positiveControlStandard = "$evidence/positive_control_cleanfid.json"
$samplerAudit = "$evidence/sampler_audit.json"
$gateCalibration = "$evidence/gate_calibration.json"
$preflight = "$gates/cap2_preflight.json"
New-Item -ItemType Directory -Force -Path $evidence, $gates, $runs | Out-Null
```

The workspace is not a temporary pod path. It is the layout-preserving source
of the shared preflight, all three 150k promotions and evaluation leaves, the
selection, and both promoted run trees. Every `run_screen` invocation verifies
its attestation and live round trip and refuses paths outside it. The per-arm
mirror remains a second transaction log for rolling recovery; it does not
replace the common authorization workspace.

### A. Freeze the metric reference, historical baseline, and positive control

Install the pinned evaluation dependency (currently the verified PyPI release
`clean-fid==0.1.35`) in the evaluation environment, then run:

```powershell
python -m pip install -r numerics/encoder_independent_drifting/stage_cap2/requirements-eval.txt
python -m pip install -r numerics/encoder_independent_drifting/stage_cap2/requirements-positive-control.txt
```

Install both requirement sets before creating **any** metric artifact. The
positive-control dependencies pin NumPy and Pillow; installing them later would
change the numerical environment after baseline/calibration creation and make
preflight correctly reject the comparison.

First partition all 50,000 training images into two disjoint halves, compute
the one real/real discrepancy point, and seal the full clean-Inception feature
population in dataset-index order. CleanFID 0.1.35 publishes the CIFAR-10 FID
moments but not a train KID feature archive, so this local, immutable cache is
mandatory:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.metric_calibration `
  --data-root C:\path\to\cifar10 `
  --samples-per-side 25000 `
  --left-dir "$evidence/real_left_pngs" `
  --right-dir "$evidence/real_right_pngs" `
  --kid-reference-features-out $kidReference `
  --metric-batch 128 --metric-workers 0 `
  --out $metricCalibration
```

The two halves must be exactly 25,000 images: together they cover every train
index once. The `.npz`, its SHA sidecar, and the calibration JSON are one
artifact family. `--metric-workers 0` is the safe Windows setting; Linux may
use the default worker count. This is one finite-sample discrepancy point, not
a variance estimate or confidence interval.

Then standardize the preserved historical baseline against that exact KID
population:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.standard_metrics `
  --checkpoint numerics/encoder_independent_drifting/stage_cap/checkpoints/cap_emf1_step650000_ema.pt `
  --device cuda `
  --data-root C:\path\to\cifar10 `
  --png-dir "$evidence/baseline_pngs" `
  --batch 128 --metric-batch 128 --feature-batch 128 `
  --kid-reference-features $kidReference `
  --generated-features "$evidence/baseline_clean_features.npz" `
  --metric-workers 0 `
  --out $baselineStandard
```

This is report-only and does not open the local CIFAR-10 test split.
The unused legacy-TensorFlow FID pass is off by default; it can be requested
explicitly for diagnosis but is not part of admission.

The frozen positive control is NVIDIA's official class-conditional CIFAR-10
StyleGAN2-ADA model. Its repository revision and checkpoint hash are fixed by
`stage_cap2/positive_control.py`; the 50,000 samples are deterministic and
exactly balanced by the rule `class = seed mod 10`. Prepare an external checkout
and network file, then generate the immutable source record:

```powershell
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git C:\path\to\stylegan2-ada-pytorch
git -C C:\path\to\stylegan2-ada-pytorch checkout d72cc7d041b42ec8e806021a205ed9349f87c6a4
Invoke-WebRequest `
  -Uri https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl `
  -OutFile C:\path\to\cifar10-stylegan2-ada.pkl

python -m numerics.encoder_independent_drifting.stage_cap2.positive_control `
  --stylegan-repo C:\path\to\stylegan2-ada-pytorch `
  --network C:\path\to\cifar10-stylegan2-ada.pkl `
  --out-dir "$evidence/positive_control_pngs" `
  --provenance-out "$evidence/positive_control_source.json" `
  --device cuda --batch 100
```

The generator verifies the official checkpoint SHA-256
`f8952c74e23da2186d147ad871c48780bd59500ee37c301201081ee8e0cb32f1`
before loading the pickle. It publishes through a staging directory and binds
the exact PNG manifest to the source record. Evaluate that folder through the
identical standard pipeline:

```powershell
$citation = (Get-Content "$evidence/positive_control_source.json" -Raw | ConvertFrom-Json).citation
python -m numerics.encoder_independent_drifting.stage_cap2.standard_metrics `
  --existing-png-dir "$evidence/positive_control_pngs" `
  --external-source-citation $citation `
  --external-source-provenance "$evidence/positive_control_source.json" `
  --kid-reference-features $kidReference `
  --metric-batch 128 --feature-batch 128 `
  --generated-features "$evidence/positive_control_clean_features.npz" `
  --metric-workers 0 `
  --device cuda `
  --data-root C:\path\to\cifar10 `
  --out $positiveControlStandard
```

The positive control is mandatory: it checks that the exact metric installation
can distinguish a known stronger sample set from the preserved CAP baseline.
Calibration, baseline, and positive-control evaluation must run in the same
recorded Python/PyTorch/torchvision/CleanFID numerical environment. Preflight
checks both that equality and the shared KID-reference archive hashes.

The frozen local run completed on 2026-08-07:

| Population | CleanFID | CleanKID | Precision | Recall |
|---|---:|---:|---:|---:|
| disjoint real/train halves | 2.1363 | 0.0000156 | -- | -- |
| preserved CAP-EMF-1 EMA | 128.8825 | 0.113611 | 0.3833 | 0.3149 |
| pinned StyleGAN2-ADA control | 3.1933 | 0.000661 | 0.7144 | 0.6836 |

The control beats the baseline on both mandatory standardized metrics by a
large margin, so the stack can distinguish a known capable generator. The
common KID archive SHA-256 is
`621febf97915d384724e8bf464d3b44757b5167286c66d2d15273981f62706db`.
These are train-reference development metrics, not held-out test claims.

### B. Freeze the saved-checkpoint mechanism audit

This repeats the base-head/refiner/final decomposition on fixed train-only
latents, retaining raw and clipped results separately and mapping a predeclared
two-dimensional `(t,h)` response grid:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.checkpoint_forensics `
  --checkpoint numerics/encoder_independent_drifting/stage_cap/checkpoints/cap_emf1_step650000_ema.pt `
  --device cuda --expected-gpu-name "RTX 4090" `
  --data-root C:\path\to\cifar10 `
  --samples 2048 --grid-samples 256 --batch 16 `
  --out "$gates/checkpoint_forensics.json"
```

`RTX 4090` is an example. Replace it everywhere with a substring of the exact
provider GPU model selected for the screen. Forensics must run in that same
GPU/PyTorch/CUDA/cuDNN/cuBLAS environment: the preflight checks those fields
against numerical admission rather than accepting a local substitute.

### C. Run the sampler audit

The source-bound two-million-draw artifact was regenerated after the code
audit and returned `GO`. The command below reproduces it at a new unused path;
canonical artifacts are immutable once consumed.

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.sampler_audit `
  --count 2000000 `
  --out $samplerAudit
```

### D. Calibrate the two-sided health gate

The audited gate artifact was regenerated from twelve globally disjoint pairs
of 2,048 CIFAR-10 training images and returned `GO`. The command below
reproduces it at a new unused path after the source audit is frozen.

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.gate_calibration `
  --data-root C:\path\to\cifar10 `
  --samples 2048 --repeats 12 `
  --out $gateCalibration
```

Only disjoint CIFAR-10 training subsets are used.

### E. Run the production numerical admission

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission `
  --checkpoint numerics/encoder_independent_drifting/stage_cap/checkpoints/cap_emf1_step650000_ema.pt `
  --candidate local_1000_d0002_fp32 `
  --device cuda --expected-gpu-name "RTX 4090" `
  --data-root C:\path\to\cifar10 `
  --batch 4 --repeats 3 --include-gradient `
  --out "$gates/numerical_admission.json"
```

No sampler training is allowed unless this artifact says `GO`.

### F. Benchmark the complete loop

The benchmark includes data access, training, EMA, component health, a real
checkpoint, and recovery serialization:

First mount a provider volume or bucket-backed filesystem that survives GPU
instance deletion. Provision and probe a fresh benchmark namespace; the command
does not create or silently substitute a local directory:

```powershell
New-Item -ItemType Directory -Force -Path "$mirrorRoot/benchmark" | Out-Null
python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror provision `
  --mirror-dir "$mirrorRoot/benchmark" `
  --storage-id "provider-volume-id/$runTag/benchmark" `
  --i-attest-instance-independent-storage

python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror probe `
  --mirror-dir "$mirrorRoot/benchmark"
```

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.benchmark `
  --arm ordered_uniform `
  --numerical local_1000_d0002_fp32 `
  --device cuda --expected-gpu-name "RTX 4090" `
  --data-root C:\path\to\cifar10 `
  --steps 2000 --micro-batch 16 `
  --hourly-rate 0.75 `
  --durable-mirror-dir "$mirrorRoot/benchmark" `
  --i-confirm-durable-mirror `
  --out "$gates/benchmark.json"
```

The projected price is tied to the actual device and batch split. The 2,000
updates exercise ordinary logging at production cadence and deliberately time
one ordinary 512-sample health event plus one checkpoint 2,048-sample raw/EMA
health event. Those two event costs are subtracted from the base loop and added
back at their true 2k and 50k production cadences.
The report does **not** multiply a one-off checkpoint/recovery event by every
2,000 updates:
it separates measured non-I/O loop time from checkpoint, snapshot, and
recovery serialization, then adds those events back at their declared
production cadences. It also retains the naive raw-loop extrapolation as a
conservative upper bound. Neither number is a confidence interval. Provider
startup, final evaluation, and upload time remain separate reserves.
The benchmark must use a fresh durable namespace: immutable mirror collisions
fail closed rather than overwriting prior evidence.

### G. Freeze the CAP2 preflight

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.preflight `
  --numerical-admission "$gates/numerical_admission.json" `
  --sampler-audit $samplerAudit `
  --gate-calibration $gateCalibration `
  --benchmark "$gates/benchmark.json" `
  --baseline-standard $baselineStandard `
  --positive-control-standard $positiveControlStandard `
  --metric-calibration $metricCalibration `
  --checkpoint-forensics "$gates/checkpoint_forensics.json" `
  --max-total-cost 50 `
  --nontraining-reserve 5 `
  --contingency-fraction 0.15 `
  --durable-storage-root $storageRoot `
  --artifact-storage-reserve-gib 20 `
  --storage-contingency-fraction 0.20 `
  --out $preflight
```

Any source or protocol change after this step invalidates the preflight.
The preflight also rejects truncated numerical matrices, old diagonal
semantics, missing or anonymous positive controls, non-pinned CleanFID,
mismatched baseline/control/calibration KID-reference hashes, mismatched
CUDA/cuDNN/cuBLAS environments, incomplete checkpoint forensics,
non-atomic/stale inputs, and a benchmark that did not exercise real
checkpoint, snapshot, recovery I/O, and both production-sized health paths.
It re-derives numerical threshold decisions, gate bounds, and projected event
counts from the serialized underlying observations rather than trusting a
top-level `GO` string.
It also requires the conservative cost of three arms through 150k plus two
arms through 300k, contingency, and the non-training reserve to remain within
the declared total ceiling. Never raise the ceiling merely to force a `GO`.
The same preflight now derives the complete immutable recovery/checkpoint/
snapshot footprint from measured benchmark bytes. It admits only a shared
durable filesystem whose total and currently free capacity exceed that
projection after a 20 GiB evaluation-artifact reserve and 20% contingency.
The expected campaign footprint is roughly 150 GiB on the current model;
provisioning a 200 GiB volume is the practical safe choice, but the measured
preflight value—not that rule of thumb—is authoritative. Storage and egress
charges must remain inside the declared nontraining dollar reserve.
Before starting the production gates, also set the provider account/project
spend cap (or an equivalent automatic instance shutdown) to at most the same
declared ceiling. The in-process wall stop protects the run state, but no
Python process can bound billing while a provider instance is stalled, idle,
or unreachable.

Once sections A, C, and D exist, the recommended production entry point
performs E, B, F, and G in exactly that order and has no code path to training.
It stops at the first failed gate. The standalone B/E/F/G commands above expose
the component operations for audit; run either those commands or this entry
point, never both into the same immutable `$gates` directory. Supply the
provider's real hourly price:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.production_readiness `
  --checkpoint numerics/encoder_independent_drifting/stage_cap/checkpoints/cap_emf1_step650000_ema.pt `
  --expected-gpu-name "RTX 4090" --hourly-rate 0.75 --micro-batch 16 `
  --data-root C:\path\to\cifar10 `
  --output-dir $gates `
  --sampler-audit $samplerAudit `
  --gate-calibration $gateCalibration `
  --baseline-standard $baselineStandard `
  --positive-control-standard $positiveControlStandard `
  --metric-calibration $metricCalibration `
  --max-total-cost 50 --nontraining-reserve 5 --contingency-fraction 0.15 `
  --durable-storage-root $storageRoot `
  --artifact-storage-reserve-gib 20 `
  --storage-contingency-fraction 0.20 `
  --durable-mirror-dir "$mirrorRoot/benchmark" `
  --i-confirm-durable-mirror `
  --i-have-authorized-production-gates
```

## 6. Staged screen

Every arm must first stop at 50k. A fresh process may not jump directly to
100k or 150k, because the preserved CAP-EMF-1 EMA does not establish numerical
fidelity for newly trained CAP2 raw weights.

Provision one fresh, attested durable namespace per arm (for example,
`$mirrorRoot/ordered_uniform`). Reuse that arm's namespace across its
50k/150k/300k continuations, but never share a namespace between arms.

```powershell
foreach ($arm in "legacy", "ordered_logitnormal", "ordered_uniform") {
  New-Item -ItemType Directory -Force -Path "$mirrorRoot/$arm" | Out-Null
  python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror provision `
    --mirror-dir "$mirrorRoot/$arm" `
    --storage-id "provider-volume-id/$runTag/$arm" `
    --i-attest-instance-independent-storage
  python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror probe `
    --mirror-dir "$mirrorRoot/$arm"
}
```

After instance replacement, first remount and probe `$workspace`; its shared
evidence and original relative layout must still be intact. Use the same
move-aside/restore procedure after **any** interrupted checkpoint transaction
for which a direct resume reports a future, stale, orphaned, or immutable
checkpoint/preview collision. Those files were published before the rolling
recovery commit and are intentionally not self-deleting. Never rename an arm
to a `_restored` sibling, because immutable promotion/selection references are
relative to that layout. Move the damaged tree aside and restore the latest
committed mirror into the same original empty arm path. The rolling mirror
deliberately excludes the large unsealed 150k PNG directory and its
presentation grid, so preserve those from the move-aside tree as well. Then
verify both the recovery stream and any already-issued concurrent selection:

```powershell
$arm = "ordered_uniform"  # replace with the arm being recovered
$armRoot = "$runs/cap2_$arm"
$damagedParent = Join-Path $storageRoot "cap2-damaged-$runTag"
$damagedRoot = Join-Path $damagedParent `
  ("{0}-{1}" -f $arm, (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $damagedParent | Out-Null
Move-Item -LiteralPath $armRoot -Destination $damagedRoot

python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror restore `
  --mirror-dir "$mirrorRoot/$arm" `
  --output-dir $armRoot

python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror verify `
  --mirror-dir "$mirrorRoot/$arm" `
  --output-dir $armRoot

foreach ($leaf in "eval_150k_pngs", "eval_150k_grid.png") {
  $saved = Join-Path $damagedRoot $leaf
  $restored = Join-Path $armRoot $leaf
  if ((Test-Path -LiteralPath $saved) -and
      -not (Test-Path -LiteralPath $restored)) {
    Copy-Item -Recurse -LiteralPath $saved -Destination $restored
  }
}

$selection = "$runs/cap2_selection_150k_to_300k.json"
if (Test-Path -LiteralPath $selection) {
  python -m numerics.encoder_independent_drifting.stage_cap2.selection `
    --revalidate $selection | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "restored arm does not revalidate against the issued selection"
  }
}
```

The restore operation selects the greatest complete recovery commit and omits
checkpoint/preview/result artifacts from an interrupted future step. Complete
committed history remains immutable; incomplete future transactions can be
recomputed without poisoning the namespace. A missing or corrupt 150k
evaluation leaf must fail selection revalidation; do not silently replace it.
If the common `$workspace` itself is unavailable after the durable volume is
remounted, stop the campaign: the per-arm rolling mirror is not a complete
backup for the shared authorization workspace.

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.run_screen `
  --arm ordered_uniform `
  --preflight $preflight `
  --updates 50000 --device cuda `
  --data-root C:\path\to\cifar10 `
  --output-dir "$runs/cap2_ordered_uniform" `
  --durable-workspace-dir $workspace `
  --i-confirm-durable-workspace `
  --durable-mirror-dir "$mirrorRoot/ordered_uniform" `
  --i-confirm-durable-mirror `
  --i-have-authorized-the-screen-run
```

Run equivalent commands for `legacy` and `ordered_logitnormal`. For each arm,
re-admit the exact 50k raw checkpoint and freeze the continuation certificate:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission `
  --checkpoint "$runs/cap2_ordered_uniform/checkpoints/cap2_ordered_uniform_step50000_raw.pt" `
  --candidate local_1000_d0002_fp32 `
  --device cuda --expected-gpu-name "RTX 4090" `
  --data-root C:\path\to\cifar10 `
  --batch 4 --repeats 3 --include-gradient `
  --out "$runs/cap2_ordered_uniform/readmission_50000_raw.json"

python -m numerics.encoder_independent_drifting.stage_cap2.early_admission `
  --preflight $preflight `
  --result-50k "$runs/cap2_ordered_uniform/result_50000.json" `
  --checkpoint-50k-raw "$runs/cap2_ordered_uniform/checkpoints/cap2_ordered_uniform_step50000_raw.pt" `
  --readmission-50k-raw "$runs/cap2_ordered_uniform/readmission_50000_raw.json" `
  --out "$runs/cap2_ordered_uniform/early_admission_50000.json"
```

Only then continue the same recovery stream to 150k:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.run_screen `
  --arm ordered_uniform `
  --preflight $preflight `
  --data-root C:\path\to\cifar10 `
  --updates 150000 --device cuda `
  --output-dir "$runs/cap2_ordered_uniform" `
  --durable-workspace-dir $workspace `
  --i-confirm-durable-workspace `
  --durable-mirror-dir "$mirrorRoot/ordered_uniform" `
  --i-confirm-durable-mirror `
  --early-admission "$runs/cap2_ordered_uniform/early_admission_50000.json" `
  --i-have-authorized-the-screen-run
```

At 150k, the declared final EMA--not a hand-picked raw or intermediate
checkpoint--receives the fixed 50k-sample development evaluation. Run this on
the same local machine, package versions, numerical settings, metric batch,
and workers used for the frozen historical baseline; do not compare a cloud
candidate evaluation with a local baseline:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.development_evaluation `
  --unit "$runs/cap2_ordered_uniform/result_150000.json" `
  --device cuda --data-root C:\path\to\cifar10 `
  --png-dir "$runs/cap2_ordered_uniform/eval_150k_pngs" `
  --grid "$runs/cap2_ordered_uniform/eval_150k_grid.png" `
  --generation-batch 128 --metric-batch 128 --feature-batch 128 `
  --kid-reference-features $kidReference `
  --metric-workers 0 `
  --out "$runs/cap2_ordered_uniform/eval_150k.json"
```

Then rerun the complete numerical matrix on the exact **raw** checkpoint that
would be optimized after continuation. The EMA remains the quality-evaluation
checkpoint. Construct one individual eligibility certificate for every arm:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission `
  --checkpoint "$runs/cap2_ordered_uniform/checkpoints/cap2_ordered_uniform_step150000_raw.pt" `
  --candidate local_1000_d0002_fp32 `
  --device cuda --expected-gpu-name "RTX 4090" `
  --data-root C:\path\to\cifar10 `
  --batch 4 --repeats 3 --include-gradient `
  --out "$runs/cap2_ordered_uniform/readmission_150k_raw.json"

python -m numerics.encoder_independent_drifting.stage_cap2.promotion `
  --preflight $preflight `
  --result-150k "$runs/cap2_ordered_uniform/result_150000.json" `
  --checkpoint-150k-raw "$runs/cap2_ordered_uniform/checkpoints/cap2_ordered_uniform_step150000_raw.pt" `
  --checkpoint-150k-ema "$runs/cap2_ordered_uniform/checkpoints/cap2_ordered_uniform_step150000_ema.pt" `
  --readmission "$runs/cap2_ordered_uniform/readmission_150k_raw.json" `
  --development-evaluation "$runs/cap2_ordered_uniform/eval_150k.json" `
  --out "$runs/cap2_ordered_uniform/promotion_150k_to_300k.json"
```

Run the equivalent command for all arms. For `legacy` only, append
`--allow-valid-legacy-control`: this changes the process exit status only when
the immutable quality decision is `NO_GO` but its separately recomputed
`control_continuation` decision is `GO`. It does not rewrite the quality verdict
or waive any integrity, numerical, health, collapse, or evaluation check.

After all three individual records exist, build the mandatory concurrent
selection. It retains `legacy` and selects at most one ordered arm:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.selection `
  --legacy-promotion "$runs/cap2_legacy/promotion_150k_to_300k.json" `
  --ordered-logitnormal-promotion "$runs/cap2_ordered_logitnormal/promotion_150k_to_300k.json" `
  --ordered-uniform-promotion "$runs/cap2_ordered_uniform/promotion_150k_to_300k.json" `
  --out "$runs/cap2_selection_150k_to_300k.json"
```

Only arms named by a `GO` selection may continue. The ordered arm must have its
own individual `GO`. The concurrent `legacy` arm may have an individual
quality `NO_GO` only when its separately recomputed `control_continuation` is
`GO`; that exemption covers exactly the two historical-improvement checks and
does not waive numerical, integrity, health, collapse, or evaluation checks:

```powershell
$selection = "$runs/cap2_selection_150k_to_300k.json"
$selectionRecord = python -m numerics.encoder_independent_drifting.stage_cap2.selection `
  --revalidate $selection | ConvertFrom-Json
$orderedArm = $selectionRecord.ordered_winner
if ($selectionRecord.decision -ne "GO" -or
    $orderedArm -notin "ordered_logitnormal", "ordered_uniform") {
  throw "verified selection has no valid ordered winner"
}

foreach ($arm in "legacy", $orderedArm) {
  $armRoot = "$runs/cap2_$arm"
  python -m numerics.encoder_independent_drifting.stage_cap2.run_screen `
    --arm $arm `
    --preflight $preflight `
    --data-root C:\path\to\cifar10 `
    --updates 300000 --device cuda `
    --output-dir $armRoot `
    --durable-workspace-dir $workspace `
    --i-confirm-durable-workspace `
    --durable-mirror-dir "$mirrorRoot/$arm" `
    --i-confirm-durable-mirror `
    --promotion "$armRoot/promotion_150k_to_300k.json" `
    --selection $selection `
    --i-have-authorized-the-screen-run `
    --i-have-authorized-the-300k-promotion
  if ($LASTEXITCODE -ne 0) { throw "300k continuation failed for $arm" }
}
```

Use the same `--output-dir` for a promoted continuation. Recovery identity is
stable across horizons, while immutable results are written as
`result_150000.json` and `result_300000.json`. The promoted horizon resets only
its final-window clipping counters; total counters, model, optimizer, EMA, RNG
streams, histories, checkpoints, and pending objective-ledger rows continue.

The runner refuses any horizon beyond 300k. A full confirmation must receive a
new protocol after the screen is interpreted.

### Terminal 300k readmission, evaluation, and verdict

Training completion is not a result by itself. After the selected ordered arm
and concurrent legacy control both reach 300k, identify the selected ordered
arm mechanically and run a fresh raw-state numerical admission for each arm on
the production GPU:

```powershell
$selection = "$runs/cap2_selection_150k_to_300k.json"
$selectionRecord = python -m numerics.encoder_independent_drifting.stage_cap2.selection `
  --revalidate $selection | ConvertFrom-Json
$orderedArm = $selectionRecord.ordered_winner
if ($selectionRecord.decision -ne "GO" -or
    $orderedArm -notin "ordered_logitnormal", "ordered_uniform") {
  throw "verified selection has no valid ordered winner"
}
$legacyRoot = "$runs/cap2_legacy"
$orderedRoot = "$runs/cap2_$orderedArm"

python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission `
  --checkpoint "$legacyRoot/checkpoints/cap2_legacy_step300000_raw.pt" `
  --candidate local_1000_d0002_fp32 `
  --device cuda --expected-gpu-name "RTX 4090" `
  --data-root C:\path\to\cifar10 `
  --batch 4 --repeats 3 --include-gradient `
  --out "$legacyRoot/readmission_300k_raw.json"

python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission `
  --checkpoint "$orderedRoot/checkpoints/cap2_${orderedArm}_step300000_raw.pt" `
  --candidate local_1000_d0002_fp32 `
  --device cuda --expected-gpu-name "RTX 4090" `
  --data-root C:\path\to\cifar10 `
  --batch 4 --repeats 3 --include-gradient `
  --out "$orderedRoot/readmission_300k_raw.json"
```

Evaluate the final EMA checkpoints on the same local machine and exact metric
environment used for the baseline/control/calibration artifacts. The explicit
feature paths retain the full generated clean-Inception populations:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.development_evaluation `
  --unit "$legacyRoot/result_300000.json" `
  --device cuda --data-root C:\path\to\cifar10 `
  --png-dir "$legacyRoot/eval_300k_pngs" `
  --grid "$legacyRoot/eval_300k_grid.png" `
  --generation-batch 128 --metric-batch 128 --metric-workers 0 --feature-batch 128 `
  --kid-reference-features $kidReference `
  --generated-features "$legacyRoot/eval_300k_clean_features.npz" `
  --out "$legacyRoot/eval_300k.json"

python -m numerics.encoder_independent_drifting.stage_cap2.development_evaluation `
  --unit "$orderedRoot/result_300000.json" `
  --device cuda --data-root C:\path\to\cifar10 `
  --png-dir "$orderedRoot/eval_300k_pngs" `
  --grid "$orderedRoot/eval_300k_grid.png" `
  --generation-batch 128 --metric-batch 128 --metric-workers 0 --feature-batch 128 `
  --kid-reference-features $kidReference `
  --generated-features "$orderedRoot/eval_300k_clean_features.npz" `
  --out "$orderedRoot/eval_300k.json"
```

Finally build the paired verdict in the same frozen evaluation environment
(machine, package versions, device/numerical settings, metric batch/workers)
used to create the baseline, calibration, and candidate evaluations, with read
access to both durable mirror roots. Then independently revalidate the written
verdict. The second command reloads all references and recomputes the retained
CleanFID/KID scores; it is not a print-only check:

```powershell
$finalVerdict = "$runs/cap2_final_verdict_300k.json"
python -m numerics.encoder_independent_drifting.stage_cap2.final_verdict `
  --selection $selection `
  --arm-artifacts legacy `
    "$legacyRoot/result_300000.json" `
    "$legacyRoot/checkpoints/cap2_legacy_step300000_raw.pt" `
    "$legacyRoot/checkpoints/cap2_legacy_step300000_ema.pt" `
    "$legacyRoot/readmission_300k_raw.json" `
    "$legacyRoot/eval_300k.json" `
    "$mirrorRoot/legacy" `
  --arm-artifacts $orderedArm `
    "$orderedRoot/result_300000.json" `
    "$orderedRoot/checkpoints/cap2_${orderedArm}_step300000_raw.pt" `
    "$orderedRoot/checkpoints/cap2_${orderedArm}_step300000_ema.pt" `
    "$orderedRoot/readmission_300k_raw.json" `
    "$orderedRoot/eval_300k.json" `
    "$mirrorRoot/$orderedArm" `
  --out $finalVerdict

python -m numerics.encoder_independent_drifting.stage_cap2.final_verdict `
  --revalidate $finalVerdict
```

A `GO` here means only a paired, one-seed, CIFAR-10-train-reference
developmental win at the declared 300k horizon. A `NO_GO` is still a valid,
integrity-preserved experiment and must not be overwritten or reinterpreted
through auxiliary metrics.

## 7. Promotion criteria

An arm is individually eligible at 150k only when it jointly has:

- numerical admission still satisfied on its checkpoint;
- CleanFID and CleanKID improve beyond the recorded direct real/real
  discrepancy;
- repository-backend KID and relative precision/recall are reported diagnostics
  at this 2,048-sample size; only the predeclared absolute precision, recall,
  F1, duplicate, and exact-copy collapse vetoes are gates;
- effective rank, every Haar band, moment, variance, and saturation inside the
  calibrated two-sided gate;
- stable base-head, residual, and final high-frequency trajectories;
- the fixed exact `(t,r,h)=(1,0,1)` probe contains all 2,048 sealed rows under
  raw and EMA weights at 100k and 150k, with finite summaries and late error
  no more than four times the early value (a coarse one-sided non-explosion
  check, not a convergence claim); naturally sampled corner occupancy is
  diagnostic because its expectation differs radically by sampler arm;
- the complete numerical matrix still passes on the 150k raw checkpoint;
- genuine final-window clipping below its threshold;
- no non-finite update;
- one model call at inference;
- an intact fixed, uncurated grid. Visual interpretation is report-only and
  may veto a run, but it never rescues a failed quantitative gate.

The cross-arm certificate additionally requires the ordered candidate to beat
the concurrent `legacy` arm on CleanFID and CleanKID beyond the same observed
real/real margins. If both ordered arms are indistinguishable within the
standard-metric margins, neither is selected from auxiliary noise.

No CIFAR-10 test metric may select the arm.

## 8. Interpretation rules

- If ordered logit-normal wins but endpoint coverage remains weak, coefficient
  control helped but the inference boundary remains open.
- If ordered uniform wins, the paper-default full-triangle coverage becomes the
  foundation setting.
- If neither ordered arm helps, investigate the patch decoder before another
  long run.
- If the base head remains pathological while the final is acceptable, do not
  remove or penalize the refiner. Test an overlapping/anti-aliased decoder that
  makes the base itself healthy.
- If late drift remains after sampler repair, test LR decay or reconstructible
  EMA horizons as a separate factor.
- ASFD remains blocked until a foundation arm passes the full capability gate.

## 9. Claim boundary

CAP2 is still a one-call, encoder-free **foundation** experiment. It does not
by itself establish an encoder-independent drifting model, image-generation
competitiveness, replication, or generalization beyond unconditional 32x32
CIFAR-10.
