# CAP-EMF-2 750k foundation -> ASFD -> final evaluation

**Status:** executable run card; this file does not itself authorize spending<br>
**Provider hard cap:** explicitly selected before any paid GPU work; reauthorize
at a higher value only if a genuine benchmark rejects the initial envelope<br>
**Scientific unit:** one all-class CIFAR-10 foundation and one ASFD continuation<br>
**Inference:** exactly one generator call

## 1. Exact sequence

```text
freeze source and regenerate source-bound evidence
  -> production GPU admission and full-loop benchmark
  -> ordered-uniform CAP foundation, updates 0..50,000
  -> raw-state numerical admission at the planned pause
  -> exact same-recovery continuation, updates 50,001..750,000
  -> fixed 650k and 750k train-reference evaluations
  -> quantitative capability gate plus uncurated-grid review
  -> target-only frozen-feature qualification
  -> durable role-separated feature banks
  -> outcome-blind coefficient calibration and 500-update measured preflight
  -> exact recovery fork, ASFD updates 750,001..800,000
  -> fixed 50,000-sample evaluation and immutable comparison report
```

This is the concentrated-budget design. It does not train several foundation
arms. Consequently the final 750k-vs-800k comparison answers whether the final
model improved, but cannot attribute that difference uniquely to ASFD rather
than to 50k additional optimizer updates. A matched raw-only continuation is a
later causal ablation, not hidden inside this run.

## 2. Frozen mechanism

- Official CIFAR-10 training split, all 50,000 images and all ten classes.
- Width-384 patch-2 U-ViT, direct-x Euler Mean Flow, ordered-uniform sampler.
- 750,000 foundation updates at effective batch 64.
- The 750k EMA is the only final foundation and the frozen ASFD teacher. The
  650k EMA is a prospectively declared stability check, never selectable.
- The ASFD online model begins from the exact 750k **raw recovery**: online
  weights, Adam, EMA and every primary RNG stream. It does not restart from EMA.
- 50,000 continuation updates; one correction event every ten updates.
- Each correction is the unprojected sum of a freshly calibrated spectral
  anchor, multi-radius raw Laplace field energy, and frozen-self-feature
  Laplace field energy. Independent caps are 0.15, 0.10 and 0.10 of the primary
  gradient, followed by the foundation's ordinary global clip.
- The uncapped components are gradients of explicit nonnegative losses, but
  the independent live norm caps are state/batch dependent. Therefore the
  realized ASFD update is a stabilized multi-loss heuristic, not the gradient
  of one fixed population energy and not itself covered by the formal
  identifiability converse.
- Four role-separated feature views per image for each train/fresh x
  positive/probe bank. Qualification images are excluded. Float16 stores
  frozen descriptors; kernels and energies use float32.
- Teacher, banks, kernels and spectral anchor are training-only and discarded
  at inference.

## 3. Global safety and budget rules

1. Commit the full tree before generating evidence. Any source edit invalidates
   every downstream artifact.
2. Provision non-nested, off-instance authorization-workspace and immutable-
   mirror roots. Attest and live-probe both roots. Use 200 GiB unless the
   measured storage preflight asks for more.
3. Set an explicit provider-side hard spend/runtime cap. USD 75 is the current
   conservative ceiling after the budget was relaxed; it is not a spending
   target. If the genuine benchmark requires more, raise both the declared
   ceiling and the provider cap deliberately **before update one**. Python
   projections are admission evidence, not a billing control.
4. The initial campaign explicitly reserves a configurable amount for
   post-foundation ASFD training and
   20 GiB for banks, in addition to measured continuation recovery/snapshots.
   If this does not fit, stop before update one.
5. Keep the CIFAR-10 test split sealed. This one-seed proof of concept uses the
   published train reference and cannot become a confirmation claim.

## 4. Path setup

```powershell
$repo = "C:\path\to\drifting_identifiability"
$env:PYTHONPATH = $repo
# Required by torch.use_deterministic_algorithms(True) for CUDA GEMMs.  Set it
# before launching any Python process; setting it after torch import is too late.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$storageRoot = "X:\"  # one provider volume; capacity is measured once here
$workspace = "$storageRoot\cap_asfd_workspace"
$mirror = "$storageRoot\cap_asfd_mirror"
$gates = "$workspace\gates"
$foundationRun = "$workspace\foundation"
$asfdRun = "$workspace\asfd"
$dataRoot = "$workspace\cifar"
$evidence = "$workspace\evidence"
$kidReference = "$evidence\cifar10_train_clean_features.npz"
$metricCalibration = "$evidence\metric_calibration.json"
$baselineStandard = "$evidence\baseline_cleanfid.json"
$positiveControlStandard = "$evidence\positive_control_cleanfid.json"
$samplerAudit = "$evidence\sampler_audit.json"
$gateCalibration = "$evidence\gate_calibration.json"
$admissionCheckpoint = "$evidence\checkpoints\cap_emf1_step650000_ema.pt"
$admissionCheckpointSha = "b55b2a62bfc44e546f347cb348b8e7e63aef6686d8a97527f6d4d232a5023f49"
$expectedGpu = "PROVIDER GPU NAME SUBSTRING"
$hourlyRate = 0.00  # replace with the all-in provider rate
$microBatch = 16    # benchmarked value; lower before preflight if admission OOMs
$maxTotalCost = 75.00  # hard ceiling, not a target; freeze before update one
$asfdReserve = 25.00   # protected continuation allocation; not a model-size knob
New-Item -ItemType Directory -Force -Path $evidence, "$evidence\checkpoints", $gates | Out-Null
```

Provision CPython 3.11.15, then install the single frozen production/evaluation
environment before generating any evidence:

```powershell
python -m pip install -r numerics\encoder_independent_drifting\stage_cap2\requirements-production-cu126.txt
```

The preserved 650k admission checkpoint is intentionally not stored in Git.
Upload the local file
`numerics/encoder_independent_drifting/stage_cap/checkpoints/cap_emf1_step650000_ema.pt`
to `$admissionCheckpoint`, then fail closed on its immutable bytes:

```powershell
if (-not (Test-Path -LiteralPath $admissionCheckpoint -PathType Leaf)) {
  throw "missing uploaded admission checkpoint: $admissionCheckpoint"
}
$actualAdmissionSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $admissionCheckpoint).Hash.ToLowerInvariant()
if ($actualAdmissionSha -ne $admissionCheckpointSha) {
  throw "admission checkpoint SHA mismatch"
}
```

The sampler audit, gate calibration, baseline, positive control and metric
calibration below must be freshly regenerated from the frozen commit. Reusing
earlier source-bound JSON is forbidden. The admission checkpoint is the
preserved historical 650k EMA; verify its immutable checkpoint metadata rather
than pretending it was trained by this source tree.

Before the production command, regenerate the source-bound local evidence by
following Sections 5A, 5C and 5D of the historical screen protocol at fresh
paths. Substitute `$admissionCheckpoint` for the historical in-repository
checkpoint path. Section 5B is deliberately omitted because
`production_readiness` runs checkpoint forensics once on the production GPU:

- 25k/25k disjoint train calibration and the complete 50k KID feature archive;
- 50k samples from `$admissionCheckpoint` through `stage_cap2.standard_metrics`;
- the pinned StyleGAN2-ADA positive-control population and the same metrics;
- the two-million-draw sampler audit; and
- the twelve-pair, 2,048-sample train-only gate calibration.

Those commands remain canonical in
[`EncoderIndependentCAPEMF2ScreenProtocol.md`](EncoderIndependentCAPEMF2ScreenProtocol.md#5-mandatory-admission-sequence).
They are prerequisites, not an alternative campaign. Create four separate
directories on the same durable volume, then attest and live-probe each exact
namespace used by an executable:

```powershell
New-Item -ItemType Directory -Force -Path `
  $workspace, $mirror, "$mirror\foundation", "$mirror\asfd" | Out-Null
foreach ($item in @(
  @{Path=$workspace; Id="provider-volume/workspace"},
  @{Path=$mirror; Id="provider-volume/benchmark"},
  @{Path="$mirror\foundation"; Id="provider-volume/foundation"},
  @{Path="$mirror\asfd"; Id="provider-volume/asfd"}
)) {
  python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror provision `
    --mirror-dir $item.Path --storage-id $item.Id `
    --i-attest-instance-independent-storage
  python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror probe `
    --mirror-dir $item.Path
}
```

The roots are distinct and non-nested relative to their corresponding run
directories, while `$storageRoot` sees the combined workspace, immutable
mirrors, banks and evaluation evidence priced by the storage preflight.

## 5. Production admission (no training path)

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.production_readiness `
  --checkpoint $admissionCheckpoint `
  --checkpoint-sha256 $admissionCheckpointSha `
  --expected-gpu-name $expectedGpu --hourly-rate $hourlyRate `
  --micro-batch $microBatch `
  --data-root $dataRoot --output-dir "$gates\production" `
  --sampler-audit $samplerAudit `
  --gate-calibration $gateCalibration `
  --baseline-standard $baselineStandard `
  --positive-control-standard $positiveControlStandard `
  --metric-calibration $metricCalibration `
  --max-total-cost $maxTotalCost --nontraining-reserve 10 `
  --post-foundation-training-reserve $asfdReserve `
  --contingency-fraction 0.15 `
  --campaign ordered_750_foundation `
  --durable-mirror-dir $mirror --durable-storage-root $storageRoot `
  --artifact-storage-reserve-gib 20 --storage-contingency-fraction 0.20 `
  --i-confirm-durable-mirror --i-have-authorized-production-gates
```

Stop unless every gate, the aggregate budget with the explicitly frozen ASFD
reserve, and the complete storage projection are `GO`.

## 6. Single 750k foundation with fail-closed 50k admission pause

The planned horizon is 750k from the first update. The pause does not create a
50k model, change the recovery identity, or open another arm.

```powershell
$preflight = "$gates\production\cap2_preflight.json"
$foundationAdmission = "$gates\foundation_50k_early_admission.json"
$foundationReadmission = "$gates\foundation_50k_raw_readmission.json"

# Phase A: this invocation is mechanically unable to cross update 50,000.
python -m numerics.encoder_independent_drifting.stage_cap2.run_screen `
  --arm ordered_uniform --preflight $preflight --updates 750000 `
  --device cuda --data-root $dataRoot --output-dir $foundationRun `
  --durable-mirror-dir "$mirror\foundation" --i-confirm-durable-mirror `
  --durable-workspace-dir $workspace --i-confirm-durable-workspace `
  --pause-for-early-admission --i-have-authorized-the-screen-run
if ($LASTEXITCODE -ne 0) { throw "foundation failed before the 50k pause" }

# Audit the trained raw state on the same production GPU and numerical mode.
python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission `
  --checkpoint "$foundationRun\checkpoints\cap2_ordered_uniform_step50000_raw.pt" `
  --candidate local_1000_d0002_fp32 --device cuda --batch 4 --repeats 3 `
  --data-root $dataRoot --expected-gpu-name $expectedGpu --include-gradient `
  --out $foundationReadmission
if ($LASTEXITCODE -ne 0) { throw "50k raw numerical readmission did not return GO" }

# This immutable GO binds the preflight, partial result, raw checkpoint and
# readmission. A NO_GO exits nonzero and cannot unlock Phase B.
python -m numerics.encoder_independent_drifting.stage_cap2.early_admission `
  --preflight $preflight --result-50k "$foundationRun\result_50000.json" `
  --checkpoint-50k-raw "$foundationRun\checkpoints\cap2_ordered_uniform_step50000_raw.pt" `
  --readmission-50k-raw $foundationReadmission --out $foundationAdmission
if ($LASTEXITCODE -ne 0) { throw "foundation continuation was not authorized" }

# Phase B: exact optimizer, EMA and RNG recovery; no new model is initialized.
python -m numerics.encoder_independent_drifting.stage_cap2.run_screen `
  --arm ordered_uniform --preflight $preflight --updates 750000 `
  --device cuda --data-root $dataRoot --output-dir $foundationRun `
  --early-admission $foundationAdmission `
  --durable-mirror-dir "$mirror\foundation" --i-confirm-durable-mirror `
  --durable-workspace-dir $workspace --i-confirm-durable-workspace `
  --i-have-authorized-the-screen-run
```

The runner publishes every 50k checkpoint pair, every 25k raw snapshot and
every 5k immutable recovery commit. After instance loss, restore into a clean
foundation directory; omitting `--recovery-step` selects the greatest complete
numeric recovery commit:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror restore `
  --mirror-dir "$mirror\foundation" --output-dir $foundationRun
python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror verify `
  --mirror-dir "$mirror\foundation" --output-dir $foundationRun
```

Then rerun the command for the active phase: before 50k use Phase A; once the
50k result and GO exist, use Phase B with the same `$foundationAdmission`.
Every Phase-B restart revalidates that immutable admission before loading the
same recovery, so it resumes rather than starting a second model.

## 7. Foundation evaluation and authorization

```powershell
$unit = "$foundationRun\result_750000.json"
python -m numerics.encoder_independent_drifting.stage_cap2.development_evaluation `
  --unit $unit --step 650000 --device cuda --data-root $dataRoot `
  --png-dir "$workspace\evidence\foundation_650k_pngs" `
  --grid "$workspace\evidence\foundation_650k_grid.png" `
  --kid-reference-features $kidReference `
  --generated-features "$workspace\evidence\foundation_650k_features.npz" `
  --out "$workspace\evidence\foundation_650k_eval.json"
python -m numerics.encoder_independent_drifting.stage_cap2.development_evaluation `
  --unit $unit --step 750000 --device cuda --data-root $dataRoot `
  --png-dir "$workspace\evidence\foundation_750k_pngs" `
  --grid "$workspace\evidence\foundation_750k_grid.png" `
  --kid-reference-features $kidReference `
  --generated-features "$workspace\evidence\foundation_750k_features.npz" `
  --out "$workspace\evidence\foundation_750k_eval.json"
```

Run checkpoint-specific numerical admission on the final **raw** 750k
checkpoint, using the same production GPU and admitted candidate:

```powershell
$raw750 = "$foundationRun\checkpoints\cap2_ordered_uniform_step750000_raw.pt"
python -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission `
  --checkpoint $raw750 --candidate local_1000_d0002_fp32 `
  --device cuda --expected-gpu-name $expectedGpu --data-root $dataRoot `
  --batch 4 --repeats 3 --include-gradient `
  --out "$gates\raw_750k_readmission.json"
```

Bind an actual human decision to the fixed uncurated final grid:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.foundation_visual_review `
  --evaluation "$workspace\evidence\foundation_750k_eval.json" `
  --decision PASS --reviewer "REVIEWER NAME" `
  --acknowledgement "I reviewed the fixed uncurated grid without selecting samples" `
  --out "$gates\foundation_visual_review.json"

python -m numerics.encoder_independent_drifting.stage_cap2.foundation_gate `
  --preflight $preflight --result $unit `
  --recovery "$foundationRun\checkpoints\recovery.pt" `
  --raw-readmission "$gates\raw_750k_readmission.json" `
  --evaluation-650k "$workspace\evidence\foundation_650k_eval.json" `
  --evaluation-750k "$workspace\evidence\foundation_750k_eval.json" `
  --visual-review "$gates\foundation_visual_review.json" `
  --mirror-root "$mirror\foundation" `
  --out "$gates\foundation_gate.json"
```

Human review cannot rescue any quantitative failure. A `NO_GO` stops ASFD.

## 8. Target-only qualification and durable banks

```powershell
python -m numerics.encoder_independent_drifting.stage_asfd.qualify `
  --foundation-gate "$gates\foundation_gate.json" `
  --device cuda --data-root $dataRoot --batch 32 `
  --out "$gates\asfd_qualification.json"
python -m numerics.encoder_independent_drifting.stage_asfd.feature_bank `
  --qualification "$gates\asfd_qualification.json" `
  --device cuda --data-root $dataRoot --batch 32 `
  --output-dir "$workspace\asfd_feature_banks"
```

Qualification uses target-train images only. The bank metadata records actual
indices, actual flip bits, feature-noise/flip seeds, shapes, dtypes and hashes.
Every shard is re-hashed on load.

## 9. Measured ASFD preflight

```powershell
python -m numerics.encoder_independent_drifting.stage_asfd.preflight `
  --foundation-gate "$gates\foundation_gate.json" `
  --qualification "$gates\asfd_qualification.json" `
  --feature-bank "$workspace\asfd_feature_banks\feature_bank.json" `
  --work-dir "$workspace\asfd_preflight_work" `
  --device cuda --data-root $dataRoot `
  --out "$gates\asfd_preflight.json"
```

This executes exactly 50 outcome-blind coefficient events and a 500-update/
50-correction exact-recovery fork. It verifies finite gradients, the frozen
teacher's live input Jacobian, one-call inference, runtime and memory. Its 50k
projection with 15% contingency must fit the reserve frozen before foundation
update one. If it does not, this campaign stops: the already-trained foundation
is bound to its original CAP2 preflight, so a larger post-hoc budget cannot
reauthorize it. This is why the initial reserve is deliberately generous.

## 10. ASFD continuation

```powershell
python -m numerics.encoder_independent_drifting.stage_asfd.continuation `
  --preflight "$gates\asfd_preflight.json" `
  --output-dir $asfdRun --durable-mirror-dir "$mirror\asfd" `
  --durable-workspace-dir $workspace --durable-storage-root $storageRoot `
  --device cuda --data-root $dataRoot `
  --i-confirm-durable-mirror --i-confirm-durable-workspace `
  --i-have-authorized-asfd-continuation
```

The online/Adam/EMA/primary-stream state is preserved exactly. Extension RNGs,
finite spectral-bank refreshes, abort state and diagnostics are in every
recovery. A crash after publishing a snapshot/checkpoint but before recovery is
idempotent. The continuation must match the exact GPU/runtime measured by its
500-update preflight, rechecks the attested workspace and live storage, and
stops only after a mirrored recovery if the measured 50k wall-time envelope is
exceeded. Changing source, teacher, bank, coefficient, threshold or horizon
requires a new experiment.

After instance loss, restore into a clean ASFD directory and rerun the same
continuation command:

```powershell
python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror restore `
  --mirror-dir "$mirror\asfd" --output-dir $asfdRun
python -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror verify `
  --mirror-dir "$mirror\asfd" --output-dir $asfdRun
```

## 11. Final evaluation and report

```powershell
python -m numerics.encoder_independent_drifting.stage_asfd.evaluation `
  --result "$asfdRun\asfd_result.json" --device cuda --data-root $dataRoot `
  --png-dir "$workspace\evidence\asfd_800k_pngs" `
  --grid "$workspace\evidence\asfd_800k_grid.png" `
  --kid-reference-features $kidReference `
  --generated-features "$workspace\evidence\asfd_800k_features.npz" `
  --out "$workspace\evidence\asfd_800k_eval.json"
python -m numerics.encoder_independent_drifting.stage_asfd.final_visual_review `
  --foundation-evaluation "$workspace\evidence\foundation_750k_eval.json" `
  --asfd-evaluation "$workspace\evidence\asfd_800k_eval.json" `
  --decision PASS --reviewer "REVIEWER NAME" `
  --acknowledgement "I reviewed both fixed uncurated grids without selecting or replacing samples" `
  --out "$workspace\evidence\asfd_final_visual_review.json"
python -m numerics.encoder_independent_drifting.stage_asfd.final_report `
  --foundation-gate "$gates\foundation_gate.json" `
  --continuation "$asfdRun\asfd_result.json" `
  --asfd-evaluation "$workspace\evidence\asfd_800k_eval.json" `
  --visual-review "$workspace\evidence\asfd_final_visual_review.json" `
  --out "$workspace\evidence\asfd_final_report.json"
```

The report recomputes retained 50k CleanFID/KID evidence, requires identical
reference/environment bindings, applies the same precision/recall and
memorization collapse vetoes as the foundation gate, and binds a review of both
fixed uncurated grids. Integrity `PASS` means the evidence is valid. Only
`quality_decision: PROMISING_IMPROVEMENT` records a result beyond the frozen
real/real discrepancy margin; a merely lower floating-point value is not a win.

## 12. Claim boundary

Allowed if observed: one-call inference; no external pretrained training
encoder; self-derived frozen teacher discarded at inference; final 800k model
better or worse than its frozen 750k foundation on the recorded train-reference
metrics; one-foundation proof of concept.

Forbidden: causal superiority over an ordinary 50k continuation; replication;
sealed-test/general image-generation claims; claiming a finite bank or batch is
measure determining; post-hoc checkpoint, layer, `t_f`, bandwidth or coefficient
selection from final image quality.
Also forbidden: claiming that the finite spectral bank is globally
characteristic, that dynamic gradient caps preserve a fixed objective, or that
"no external encoder" means representation-independent training.
