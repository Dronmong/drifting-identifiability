# RunPod execution protocol: one 750k CAP foundation -> ASFD -> evaluation

**Status:** executable provider run card; it does not itself authorize spending<br>
**Provider:** RunPod Secure Cloud, on-demand RTX 4090<br>
**Storage:** one 200 GB Network Volume mounted at `/workspace`<br>
**Scientific unit:** one model and its exact recovery continuation<br>
**Inference:** exactly one generator call

This is the Linux/RunPod realization of the audited CAP-ASFD experiment. The
scientific configuration remains the one described in
[`EncoderIndependentCAPEMF2ASFDProtocol.md`](EncoderIndependentCAPEMF2ASFDProtocol.md).
The two source-bound shell scripts are:

- `stage_cap2/runpod_bootstrap.sh`: environment installation only;
- `stage_cap2/runpod_pipeline.sh`: fail-closed phase orchestration.

There is deliberately no `all` command. Production admission, the 50k raw
readmission, the 750k human grid review and the final paired grid review remain
separate boundaries.

## 1. Before creating a Pod

1. Push the release commit reported in the handoff to the repository remote.
2. In RunPod, create a **200 GB Network Volume** in a Secure Cloud datacenter
   that currently offers an on-demand RTX 4090.
3. Record the Network Volume ID and the exact displayed Pod hourly price.
4. Disable auto-pay. Enable a low-balance warning with enough reserve that the
   Network Volume cannot reach a zero-funded state.
5. Do not select a spot/interruptible Pod. Do not use Community Cloud: Network
   Volumes for Pods are a Secure Cloud facility.

The advertised Pod price is not the rate supplied to admission. The pipeline
adds the conservative hourly equivalent of the 200 GB Network Volume to the
displayed Pod price. The genuine 2,000-update benchmark remains authoritative.

## 2. Deploy the Pod

Deploy one on-demand RTX 4090 Pod with:

- the 200 GB Network Volume attached at `/workspace`;
- at least 30 GB container disk;
- an Ubuntu 22.04 RunPod PyTorch template with SSH enabled;
- no additional GPU and no autoscaling worker.

The template's bundled Python and Torch are not trusted. The bootstrap creates
a separate CPython 3.11.15 environment containing the pinned CUDA 12.6 wheels.
The production gate records the live driver, CUDA, cuDNN, cuBLAS and GPU.

Connect over SSH and confirm:

```bash
set -Eeuo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace
test -n "${RUNPOD_POD_ID:-}" || echo "warning: RUNPOD_POD_ID is not exported"
```

Stop if the GPU is not an RTX 4090 or `/workspace` is not the attached Network
Volume. A marketed 200 GB volume should expose roughly 186 GiB; the setup
requires at least 180 GiB total and 170 GiB initially free.

## 3. Checkout and immutable operator configuration

```bash
set -Eeuo pipefail
cd /workspace || exit 1
git clone https://github.com/Dronmong/drifting-identifiability.git
cd /workspace/drifting-identifiability || exit 1
git checkout --detach RELEASE_COMMIT_FROM_HANDOFF
git status --short
```

`git status` should be empty. Create the non-secret provider configuration;
replace all three placeholders with the exact release, Volume ID and displayed
Pod price. The total ceiling is a gross usage ceiling, not a spending target.

```bash
cat > /workspace/runpod_operator.env <<'EOF'
export RUNPOD_RELEASE_COMMIT="RELEASE_COMMIT_FROM_HANDOFF"
export RUNPOD_NETWORK_VOLUME_ID="NETWORK_VOLUME_ID_FROM_CONSOLE"
export RUNPOD_POD_HOURLY_RATE="DISPLAYED_POD_USD_PER_HOUR"
export RUNPOD_EXPECTED_GPU="RTX 4090"
export RUNPOD_NETWORK_VOLUME_GIB="200"
export RUNPOD_STORAGE_USD_PER_GIB_MONTH="0.07"
export CAP_ASFD_MICRO_BATCH="16"
export CAP_ASFD_MAX_TOTAL_COST="95"
export CAP_ASFD_ASFD_RESERVE="25"
EOF
chmod 600 /workspace/runpod_operator.env
source /workspace/runpod_operator.env
```

The `prepare` phase seals this file's SHA-256. Changing the price, release,
volume identity, batch split or budget after evidence begins fails closed. If
the production benchmark rejects the envelope, abandon this fresh campaign and
deliberately create a new one; do not edit a consumed authorization.

**Why the ceiling is 95 and not 75.** The gate is
`conservative_training * 1.15 + 10 + ASFD_RESERVE <= MAX_TOTAL_COST`, and
`conservative_raw_loop_upper_cost` deliberately projects the 2,000-update
benchmark rate — which crams one recovery, one snapshot, one checkpoint pair
and two health evaluations into those 2,000 updates — across all 750,000. At
CAP-EMF-1's measured 0.2289 s/update that lands near 0.31 s/update, giving
about USD 49 conservative and an authorized upper near USD 91. A ceiling of 75
would fail closed on a run whose realistic cost is USD 42–45. The ceiling is a
fail-closed bound, not a forecast; the measured projection at admission remains
authoritative and the reserve still protects ASFD.

## 4. Bootstrap without training

```bash
cd /workspace/drifting-identifiability || exit 1
bash numerics/encoder_independent_drifting/stage_cap2/runpod_bootstrap.sh
```

Expected result: CPython 3.11.15, Torch 2.7.1+cu126, torchvision
0.22.1+cu126, NumPy 1.26.4, Pillow 12.2.0 and a visible CUDA RTX 4090.
Bootstrap never invokes a training entry point.

## 5. Upload and seal the historical admission checkpoint

From the local project, upload

```text
numerics/encoder_independent_drifting/stage_cap/checkpoints/
cap_emf1_step650000_ema.pt
```

to this exact Network Volume path:

```text
/workspace/uploads/cap_emf1_step650000_ema.pt
```

Use the RunPod web file transfer, SCP coordinates shown for the Pod, or the
Network Volume S3 API. The pipeline refuses any bytes whose SHA-256 is not:

```text
b55b2a62bfc44e546f347cb348b8e7e63aef6686d8a97527f6d4d232a5023f49
```

Then run:

```bash
cd /workspace/drifting-identifiability || exit 1
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh prepare
```

This checks the source commit, exact GPU substring, real Volume capacity,
checkpoint bytes, four distinct durable namespaces and live write/read/delete
probes. It copies the checkpoint into immutable evidence. It does not train.

## 6. Fresh evidence and production admission

The following phase downloads CIFAR-10 and the pinned NVIDIA StyleGAN2-ADA
control, constructs the disjoint real/train metric calibration, evaluates the
preserved baseline and positive control, and regenerates the sampler and health
gate calibrations. It never opens the CIFAR-10 test split.

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh evidence
```

Run the complete no-training production admission:

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh admission
```

Admission performs numerical forensics and a 2,000-update full-loop benchmark,
then combines compute, storage, non-training reserve, ASFD reserve and 15%
contingency. Stop unless it exits zero and

```text
/workspace/cap_asfd_workspace/gates/production/cap2_preflight.json
```

has `decision: GO`. This is the first point at which the measured cost is known.
The 2,000 benchmark updates are disposable admission work, not a second model.

## 7. One foundation with a mandatory 50k pause

Only after manually accepting the production projection:

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-phase-a
```

The runner is mechanically unable to cross update 50,000. It retains the
750,000-update declared horizon and publishes the exact optimizer, EMA and RNG
recovery. Audit that trained raw state:

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-admit-50k
```

Only a source-bound `GO` unlocks the exact continuation:

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-phase-b
```

No model is reinitialized between these commands.

## 8. Foundation evidence and human gate

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-evaluate
```

Download and inspect the fixed uncurated grid printed by the command. Do not
select, replace or rerun samples. Then record the honest decision:

```bash
export CAP_ASFD_REVIEWER="YOUR NAME"
export CAP_ASFD_FOUNDATION_DECISION="PASS"  # or FAIL
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh foundation-review
```

A visual `PASS` cannot override quantitative capability, integrity, coverage or
memorization failures. A `FAIL` writes the review and leaves the gate closed.

## 8.1 Secondary: declared long-EMA windows

Optional, and deliberately a separate command so it cannot reach the foundation
gate:

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh posthoc-ema
```

The declared 0.9999 EMA spans about 10,000 updates, roughly 1.3% of the run,
and the right horizon is not knowable in advance. In CAP-EMF-1 a 200,000-update
uniform average of raw snapshots moved FID-50k from 112.94 to 83.65 — a 26%
improvement for no additional GPU time — and the snapshots were then lost with
the Pod, so the finding could not be reproduced. Here they are durably mirrored.

Three properties keep this from becoming a post-hoc selection on the primary
metric:

- the windows are **predeclared** (4, 8 and 16 trailing snapshots, i.e. 100k,
  200k and 400k updates) and **all** are reported; there is no best-window
  search and the command cannot request a different set;
- metrics use the **training** reference, the same one the headline FID uses;
  this module has no code path that can open the sealed test split;
- the artifact records `eligible_for_selection: false` alongside the primary
  checkpoint it must not displace.

The declared 0.9999 EMA remains the result of the experiment. Run this after
`foundation-evaluate`, or after `final-evaluate`, or not at all.

## 9. Frozen-feature qualification and ASFD

Only a complete foundation `GO` can enter this stage:

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh asfd-prepare
```

This performs target-only feature qualification, constructs four role-separated
durable feature banks, calibrates correction coefficients without outcome
access, and measures the actual 500-update ASFD graph. Review its runtime and
hard-wall projection, then explicitly launch:

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh asfd-run
```

This resumes the exact raw 750k model/Adam/EMA/RNG recovery for updates
750,001..800,000. The dynamic correction caps are stabilization heuristics, not
the gradient of one fixed identifiable population objective.

## 10. Final evaluation

```bash
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh final-evaluate
```

Download both printed uncurated grids. Record a veto-only decision:

```bash
export CAP_ASFD_REVIEWER="YOUR NAME"
export CAP_ASFD_FINAL_DECISION="PASS"  # or FAIL
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh final-review
```

The final report calls the result `PROMISING_IMPROVEMENT` only if FID/KID move
beyond the prospectively frozen real/real margin, neither metric regresses past
that margin, the final population passes coverage/memorization/noncollapse
checks, every retained artifact revalidates and the fixed visual comparison is
not vetoed.

## 11. Pod replacement and recovery

The Network Volume survives Pod termination. A Pod with a Network Volume cannot
be stopped in place; terminate it and attach the same Volume to a new on-demand
RTX 4090 Pod. Checkout the same release commit and source the sealed operator
configuration. The venv and evidence remain on `/workspace`.

If a local run tree is damaged, first move it aside without deleting it:

```bash
mv /workspace/cap_asfd_workspace/foundation \
  "/workspace/cap_asfd_workspace/foundation-damaged-$(date -u +%Y%m%dT%H%M%SZ)"
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh restore-foundation
```

or:

```bash
mv /workspace/cap_asfd_workspace/asfd \
  "/workspace/cap_asfd_workspace/asfd-damaged-$(date -u +%Y%m%dT%H%M%SZ)"
bash numerics/encoder_independent_drifting/stage_cap2/runpod_pipeline.sh restore-asfd
```

Then rerun only the active phase command. Terminal results are idempotently
reopened and revalidated rather than silently trusted.

## 12. Billing and cleanup

- Keep auto-pay disabled and maintain a low-balance warning.
- Do not allow the account to reach zero: RunPod may eventually delete an
  unfunded Network Volume even though Pod compute has stopped.
- A Network Volume Pod continues billing compute while idle. Proceed directly
  between automatic phases; terminate the Pod when pausing for an extended
  human decision.
- Download the final report, grids, feature archives, recovery and final raw/EMA
  checkpoints before deleting the Network Volume.
- Delete the Pod first and the Network Volume only after two independent copies
  of the retained result exist.

RunPod account limits are not a per-campaign budget. The source-bound cost gate,
ASFD measured hard wall, disabled auto-pay and deliberately limited prepaid
balance jointly constrain this developmental run.

## 13. Claim boundary

This remains one one-seed, all-class CIFAR-10 development experiment. A success
would show that one one-call model trained without an external or separately
trained encoder improved its frozen 750k foundation under the declared gates.
It would not prove causal ASFD benefit without a later matched raw continuation,
representation independence, replication, held-out generalization, ImageNet
parity or a theorem about the dynamically capped optimizer.
