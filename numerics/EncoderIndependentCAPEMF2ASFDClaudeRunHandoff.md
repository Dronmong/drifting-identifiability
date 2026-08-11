# Claude operator handoff: RunPod CAP-ASFD experiment

## Role and authority

You are operating and monitoring one paid RunPod experiment. The canonical
instructions are in
[`EncoderIndependentCAPEMF2ASFDRunPodProtocol.md`](EncoderIndependentCAPEMF2ASFDRunPodProtocol.md).
Read that document completely before connecting. This handoff explains how to
apply it; if the two documents ever disagree, stop and report the discrepancy.

The SSH address, private key, RunPod API token and any account credentials must
be supplied out of band. Never paste them into a repository file, command log,
chat response or generated artifact. Do not change RunPod billing settings or
add funds without the user's explicit approval.

The approved scientific unit is deliberately concentrated:

1. one foundation model, declared for 750,000 updates;
2. a mandatory numerical pause at update 50,000;
3. exact optimizer/EMA/RNG recovery from 50,000 to 750,000;
4. a quantitative and user-reviewed visual capability gate;
5. only if that gate passes, exact recovery from 750,000 to 800,000 with the
   frozen-self-feature ASFD correction;
6. one fixed 50,000-sample final evaluation and an uncurated paired visual
   review.

There are no competing training arms and no `all` command. The 2,000 admission
updates are disposable benchmarking work, not another scientific model.

## Claim boundary

This is a one-seed, all-class CIFAR-10 development experiment for one-call
generation without an external or separately trained encoder. It is not a
causal ASFD ablation, a replication, a held-out generalization result,
representation independence, an ImageNet comparison or a theorem about the
dynamically capped correction. Report negative results honestly and do not
reinterpret a failed gate as success.

## Frozen release and provider configuration

Use the exact release commit supplied by the user after this handoff is
committed and pushed. Before doing anything provider-side, verify:

```bash
git rev-parse HEAD
git status --short
```

The commit must equal `RUNPOD_RELEASE_COMMIT`, and status must be empty. Never
edit, patch, pull, rebase, install an unpinned alternative, or switch branches
after `prepare`. Any source change requires abandoning the consumed campaign
and starting a fresh admission.

Required provider layout:

- RunPod **Secure Cloud**, not Community Cloud;
- one **on-demand RTX 4090**, not spot/interruptible;
- one **200 GB Network Volume** mounted at `/workspace`;
- Ubuntu 22.04 RunPod PyTorch template with SSH enabled;
- at least 30 GB container disk;
- no second GPU or autoscaling worker.

`/workspace/runpod_operator.env` must contain the exact release commit, Network
Volume ID and console-displayed Pod hourly price. The default campaign ceiling
is USD 75 with USD 25 reserved for ASFD. This is a fail-closed ceiling, not a
spending target. The user has said additional budget may be considered, but you
must never raise the ceiling silently. Once `prepare` seals this file, any
change invalidates the campaign and requires a deliberately fresh setup.

## Admission checkpoint

The local checkpoint

```text
numerics/encoder_independent_drifting/stage_cap/checkpoints/
cap_emf1_step650000_ema.pt
```

must be uploaded to

```text
/workspace/uploads/cap_emf1_step650000_ema.pt
```

and must hash to

```text
b55b2a62bfc44e546f347cb348b8e7e63aef6686d8a97527f6d4d232a5023f49
```

Do not substitute a similarly named checkpoint.

## Exact chronological procedure

Use a persistent `tmux` session so an SSH disconnect does not kill a foreground
phase. Run one command at a time. Preserve the complete terminal output.

### 1. Provider inspection and bootstrap

```bash
set -Eeuo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace
cd /workspace/drifting-identifiability || exit 1
bash numerics/encoder_independent_drifting/stage_cap2/runpod_bootstrap.sh
```

Stop if the GPU is not an RTX 4090, the Network Volume is absent, or bootstrap
does not verify the exact pinned environment. Bootstrap must not train.

### 2. Provider, source, storage and checkpoint sealing

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh prepare
```

This must verify the release, GPU, real Volume capacity, Volume identity,
historical checkpoint hash and all durable-root probes. Stop on any warning
that becomes an error; do not bypass or hand-edit an attestation.

### 3. Fresh evidence

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh evidence
```

This downloads CIFAR-10 and the pinned StyleGAN2-ADA positive control,
regenerates metric calibration and checks the preserved baseline. It must not
open the CIFAR-10 test split.

### 4. Measured production admission

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh admission
```

This performs numerical forensics and the real 2,000-update benchmark. Before
training, report to the user:

- exact GPU and displayed/all-in hourly rate;
- benchmark update throughput and projected wall time;
- projected compute, storage, reserve, contingency and total cost;
- free storage;
- every admission decision and warning.

Do not start Phase A unless the command exits zero,
`gates/production/cap2_preflight.json` says `GO`, and the user accepts the
measured projection. A failed projection is not overrideable. If the user wants
a larger ceiling, create a fresh campaign; do not edit the sealed configuration.

### 5. Foundation Phase A: updates 0 through 50,000

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-phase-a
```

The command must stop at exactly 50,000 while retaining a declared 750,000
horizon. Confirm that the result, raw/EMA checkpoints and optimizer/RNG recovery
are durably mirrored.

### 6. Mandatory 50k raw readmission

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-admit-50k
```

Report its decision and diagnostics. Continue only on a source-bound `GO`.
Never restart or initialize a replacement model at this boundary.

### 7. Foundation Phase B: exact recovery to 750,000

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-phase-b
```

This must resume the same raw model, Adam state, EMA and RNG streams. Monitor
without modifying the run directory.

### 8. Foundation evaluation and user decision

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-evaluate
```

Provide the user the fixed uncurated 750k grid and the 650k/750k quantitative
results. Do not curate, regenerate or replace samples. Claude must not choose
`PASS` on the user's behalf. After the user explicitly replies `PASS` or `FAIL`:

```bash
export CAP_ASFD_REVIEWER="USER-SUPPLIED NAME"
export CAP_ASFD_FOUNDATION_DECISION="USER-SUPPLIED PASS OR FAIL"
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-review
```

A subjective pass cannot rescue a quantitative, integrity, coverage,
memorization or noncollapse failure. If the foundation gate is not `GO`, stop;
ASFD is not authorized for an incapable foundation.

### 9. ASFD qualification and measured preflight

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh asfd-prepare
```

This freezes and qualifies the foundation's internal features, constructs four
role-separated feature banks, calibrates coefficients without final-outcome
access and benchmarks the actual 500-update correction graph. Report the
feature health diagnostics, measured runtime, projected 50k ASFD wall time,
hard wall and cost to the user. Continue only on `GO` and user acceptance.

### 10. Exact ASFD continuation: updates 750,001 through 800,000

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh asfd-run
```

This resumes the exact 750k raw model/Adam/EMA/RNG recovery. The continuation
binds the preflight GPU/runtime/rate and enforces a mirrored recovery-first hard
wall. A hard-wall marker is terminal; never delete it or resume around it.

### 11. Final evidence and user decision

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh final-evaluate
```

Provide both fixed uncurated grids and all final metrics to the user. Do not
curate samples. After the user explicitly supplies `PASS` or `FAIL`:

```bash
export CAP_ASFD_REVIEWER="USER-SUPPLIED NAME"
export CAP_ASFD_FINAL_DECISION="USER-SUPPLIED PASS OR FAIL"
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh final-review
```

Report the final machine decision exactly. `PROMISING_IMPROVEMENT` requires a
prospectively meaningful FID/KID improvement, no disallowed metric regression,
coverage/noncollapse/memorization passes, complete artifact integrity and no
visual veto. A tiny metric change alone is not success.

## Monitoring requirements

During a running phase, inspect rather than mutate:

```bash
nvidia-smi
df -h /workspace
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh status
```

At useful intervals report update count, elapsed time, throughput, estimated
completion, GPU utilization/memory, free disk and the last health diagnostics.
Do not run a second training process, change batch size, change seeds, edit the
operator configuration, delete artifacts or manually copy checkpoints over an
active run.

An SSH disconnect is not a training failure if `tmux` and the process remain
alive. Reconnect and attach. If a command exits nonzero, stop and preserve:

- the exact command and exit code;
- the last terminal output;
- `runpod_pipeline.sh status`;
- GPU and disk state;
- the local run/recovery and durable mirror paths.

Do not blindly rerun, modify JSON, remove a halt marker or choose an older
checkpoint. Diagnose first. Normal recovery must use the greatest verified
mirrored recovery and the same phase command.

## Pod replacement and billing

The Network Volume is the persistent boundary; the Pod is disposable. A
Network-Volume Pod bills while idle and normally must be terminated rather
than stopped. If waiting materially for a user review, first verify that the
latest recovery/evidence is mirrored, then terminate the Pod and later attach
the same Volume to a new on-demand RTX 4090. Re-check the exact release and
sealed configuration on replacement.

Keep auto-pay disabled. Never allow the funded balance to reach zero while the
Network Volume contains the only copy. Do not delete the Network Volume until
the final report, grids, feature archives, recovery and final raw/EMA
checkpoints have two independently verified copies.

## Required final report to the user

Return:

1. release commit, GPU/runtime identity and actual provider rate;
2. measured total wall time and estimated provider cost;
3. every gate decision, including any veto;
4. the 650k and 750k foundation metrics;
5. the 800k ASFD metrics and calibrated deltas from the frozen 750k foundation;
6. fixed uncurated grid locations;
7. coverage, noncollapse, precision/recall and memorization results;
8. retained checkpoint/recovery/evidence paths and SHA verification state;
9. the exact final classification and the scientific claim boundary above.

Do not call the experiment successful merely because it completed.
