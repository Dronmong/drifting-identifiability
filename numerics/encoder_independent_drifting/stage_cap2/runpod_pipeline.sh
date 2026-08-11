#!/usr/bin/env bash
# Fail-closed RunPod orchestration for one 750k foundation and one ASFD
# continuation. There is deliberately no `all` command: admission and both
# human visual reviews are mandatory phase boundaries.

set -Eeuo pipefail

on_error() {
  local line="$1"
  local code="$2"
  printf 'RunPod CAP-ASFD command failed at line %s (exit %s).\n' \
    "$line" "$code" >&2
}
trap 'on_error "$LINENO" "$?"' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"

OPERATOR_CONFIG="${CAP_ASFD_RUNPOD_CONFIG:-/workspace/runpod_operator.env}"
if [[ -f "$OPERATOR_CONFIG" ]]; then
  # This file contains only non-secret operator declarations. `prepare` seals
  # its exact bytes before any source-bound evidence is generated.
  # shellcheck source=/dev/null
  source "$OPERATOR_CONFIG"
fi

VOLUME_ROOT="${RUNPOD_VOLUME_ROOT:-/workspace}"
WORKSPACE="${CAP_ASFD_WORKSPACE:-$VOLUME_ROOT/cap_asfd_workspace}"
MIRROR="${CAP_ASFD_MIRROR:-$VOLUME_ROOT/cap_asfd_mirror}"
VENV="${RUNPOD_VENV:-$VOLUME_ROOT/cap_asfd_venv}"
PYTHON="${RUNPOD_PYTHON:-$VENV/bin/python}"
UPLOAD_CHECKPOINT="${RUNPOD_ADMISSION_UPLOAD:-$VOLUME_ROOT/uploads/cap_emf1_step650000_ema.pt}"

GATES="$WORKSPACE/gates"
FOUNDATION_RUN="$WORKSPACE/foundation"
ASFD_RUN="$WORKSPACE/asfd"
DATA_ROOT="$WORKSPACE/cifar"
EVIDENCE="$WORKSPACE/evidence"
EXTERNAL="$WORKSPACE/external"
ADMISSION_CHECKPOINT="$EVIDENCE/checkpoints/cap_emf1_step650000_ema.pt"
ADMISSION_CHECKPOINT_SHA="b55b2a62bfc44e546f347cb348b8e7e63aef6686d8a97527f6d4d232a5023f49"

KID_REFERENCE="$EVIDENCE/cifar10_train_clean_features.npz"
METRIC_CALIBRATION="$EVIDENCE/metric_calibration.json"
BASELINE_STANDARD="$EVIDENCE/baseline_cleanfid.json"
POSITIVE_CONTROL_STANDARD="$EVIDENCE/positive_control_cleanfid.json"
SAMPLER_AUDIT="$EVIDENCE/sampler_audit.json"
GATE_CALIBRATION="$EVIDENCE/gate_calibration.json"
CAP2_PREFLIGHT="$GATES/production/cap2_preflight.json"
FOUNDATION_READMISSION="$GATES/foundation_50k_raw_readmission.json"
FOUNDATION_ADMISSION="$GATES/foundation_50k_early_admission.json"
FOUNDATION_GATE="$GATES/foundation_gate.json"
ASFD_QUALIFICATION="$GATES/asfd_qualification.json"
ASFD_PREFLIGHT="$GATES/asfd_preflight.json"

EXPECTED_GPU="${RUNPOD_EXPECTED_GPU:-RTX 4090}"
MICRO_BATCH="${CAP_ASFD_MICRO_BATCH:-16}"
MAX_TOTAL_COST="${CAP_ASFD_MAX_TOTAL_COST:-75}"
ASFD_RESERVE="${CAP_ASFD_ASFD_RESERVE:-25}"
NETWORK_VOLUME_GIB="${RUNPOD_NETWORK_VOLUME_GIB:-200}"
STORAGE_USD_PER_GIB_MONTH="${RUNPOD_STORAGE_USD_PER_GIB_MONTH:-0.07}"
MIN_TOTAL_BYTES=$((180 * 1024 * 1024 * 1024))
MIN_FREE_BYTES=$((170 * 1024 * 1024 * 1024))

export PYTHONPATH="$REPO_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

usage() {
  cat <<'EOF'
Usage: runpod_pipeline.sh COMMAND

Provider and admission commands:
  prepare                 verify release/GPU/volume, attest paths, seal upload
  evidence                generate fresh source-bound train-only evidence
  admission               run production readiness and 2,000-step benchmark

Single-model foundation commands:
  foundation-phase-a      exact model updates 0..50k, then mandatory pause
  foundation-admit-50k    raw numerical readmission and immutable continuation GO
  foundation-phase-b      exact recovery updates 50,001..750k
  foundation-evaluate     fixed 650k/750k evaluations and raw readmission
  foundation-review       record human PASS/FAIL and build capability gate

ASFD and final commands:
  asfd-prepare             qualify frozen features, build banks, measured smoke
  asfd-run                 exact raw recovery continuation 750,001..800k
  final-evaluate           fixed 50k-sample final evaluation
  final-review             record human PASS/FAIL and build immutable report

Recovery and inspection:
  restore-foundation      restore greatest durable foundation recovery
  restore-asfd            restore greatest durable ASFD recovery
  status                  show retained phase artifacts, GPU and disk state

There is intentionally no command that crosses every gate automatically.
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 2
}

require_var() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "Required environment variable is unset: $name"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file is missing: $path"
}

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

assert_release_tree() {
  require_var RUNPOD_RELEASE_COMMIT
  [[ -d "$REPO_ROOT/.git" ]] || die "Repository is not a Git checkout: $REPO_ROOT"
  local live_commit
  live_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  [[ "$live_commit" == "$RUNPOD_RELEASE_COMMIT" ]] || \
    die "Release commit mismatch: $live_commit != $RUNPOD_RELEASE_COMMIT"

  mapfile -t bound_paths < <(
    "$PYTHON" -c \
      'from numerics.encoder_independent_drifting.stage_cap2.artifacts import source_manifest; print(chr(10).join(source_manifest()))'
  )
  bound_paths+=("numerics/EncoderIndependentCAPEMF2ASFDRunPodProtocol.md")
  ((${#bound_paths[@]} > 3)) || die "Source manifest was unexpectedly empty"
  git -C "$REPO_ROOT" diff --quiet HEAD -- "${bound_paths[@]}" || \
    die "Release-bound source differs from the checked-out commit"
  git -C "$REPO_ROOT" diff --cached --quiet -- "${bound_paths[@]}" || \
    die "Release-bound source has staged changes"
  local path
  for path in "${bound_paths[@]}"; do
    git -C "$REPO_ROOT" ls-files --error-unmatch -- "$path" >/dev/null 2>&1 || \
      die "Release-bound path is not tracked: $path"
  done
}

assert_provider() {
  require_var RUNPOD_NETWORK_VOLUME_ID
  require_var RUNPOD_POD_HOURLY_RATE
  [[ -d "$VOLUME_ROOT" ]] || die "RunPod Network Volume is not mounted: $VOLUME_ROOT"
  [[ -x "$PYTHON" ]] || die "Pinned Python is missing; run runpod_bootstrap.sh"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"

  local gpu
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  [[ "$gpu" == *"$EXPECTED_GPU"* ]] || \
    die "Wrong GPU: expected substring '$EXPECTED_GPU', found '$gpu'"

  local total free
  read -r total free < <(
    df -PB1 "$VOLUME_ROOT" | awk 'NR==2 {print $2, $4}'
  )
  [[ "$total" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || \
    die "Could not read Network Volume capacity"

  local workspace_real mirror_real root_real
  mkdir -p -- "$WORKSPACE" "$MIRROR"
  workspace_real="$(realpath -m -- "$WORKSPACE")"
  mirror_real="$(realpath -m -- "$MIRROR")"
  root_real="$(realpath -m -- "$VOLUME_ROOT")"
  [[ "$workspace_real" == "$root_real"/* ]] || die "Workspace is outside volume"
  [[ "$mirror_real" == "$root_real"/* ]] || die "Mirror is outside volume"
  [[ "$workspace_real" != "$mirror_real" ]] || die "Workspace equals mirror"
  [[ "$workspace_real" != "$mirror_real"/* ]] || die "Workspace is inside mirror"
  [[ "$mirror_real" != "$workspace_real"/* ]] || die "Mirror is inside workspace"

  local sealed_config_sha="$WORKSPACE/runpod_operator.env.sha256"
  if [[ -f "$sealed_config_sha" ]]; then
    require_file "$OPERATOR_CONFIG"
    local expected_config_sha
    expected_config_sha="$(awk 'NR==1 {print $1}' "$sealed_config_sha")"
    [[ "$(sha256_file "$OPERATOR_CONFIG")" == "$expected_config_sha" ]] || \
      die "RunPod operator configuration changed after prepare"
  fi
}

assert_initial_capacity() {
  local total free
  read -r total free < <(
    df -PB1 "$VOLUME_ROOT" | awk 'NR==2 {print $2, $4}'
  )
  ((total >= MIN_TOTAL_BYTES)) || \
    die "Network Volume is smaller than the 180 GiB safety minimum"
  ((free >= MIN_FREE_BYTES)) || \
    die "Network Volume has less than 170 GiB free before admission"
}

all_in_hourly_rate() {
  "$PYTHON" - "$RUNPOD_POD_HOURLY_RATE" "$NETWORK_VOLUME_GIB" \
    "$STORAGE_USD_PER_GIB_MONTH" <<'PY'
import math
import sys

pod, gib, monthly = map(float, sys.argv[1:])
if not all(math.isfinite(x) and x >= 0 for x in (pod, gib, monthly)) or pod <= 0:
    raise SystemExit("RunPod rates and volume size must be finite and nonnegative")
print(f"{pod + gib * monthly / (30.0 * 24.0):.12f}")
PY
}

provision_root() {
  local path="$1"
  local label="$2"
  mkdir -p -- "$path"
  if [[ ! -f "$path/.cap2-durable-root.json" ]]; then
    "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror \
      provision --mirror-dir "$path" \
      --storage-id "runpod-network-volume/$RUNPOD_NETWORK_VOLUME_ID/$label" \
      --i-attest-instance-independent-storage
  fi
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror \
    probe --mirror-dir "$path"
  "$PYTHON" - "$path/.cap2-durable-root.json" \
    "runpod-network-volume/$RUNPOD_NETWORK_VOLUME_ID/$label" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("storage_id") != sys.argv[2]:
    raise SystemExit(
        f"durable-root storage identity changed: {payload.get('storage_id')!r} "
        f"!= {sys.argv[2]!r}"
    )
PY
}

prepare() {
  assert_provider
  assert_initial_capacity
  assert_release_tree
  require_file "$OPERATOR_CONFIG"
  mkdir -p -- "$EVIDENCE/checkpoints" "$GATES" "$EXTERNAL" \
    "$MIRROR/foundation" "$MIRROR/asfd"
  provision_root "$WORKSPACE" workspace
  provision_root "$MIRROR" benchmark
  provision_root "$MIRROR/foundation" foundation
  provision_root "$MIRROR/asfd" asfd

  require_file "$UPLOAD_CHECKPOINT"
  local uploaded_sha
  uploaded_sha="$(sha256_file "$UPLOAD_CHECKPOINT")"
  [[ "$uploaded_sha" == "$ADMISSION_CHECKPOINT_SHA" ]] || \
    die "Uploaded admission checkpoint SHA mismatch"
  if [[ ! -f "$ADMISSION_CHECKPOINT" ]]; then
    cp -- "$UPLOAD_CHECKPOINT" "$ADMISSION_CHECKPOINT"
  fi
  [[ "$(sha256_file "$ADMISSION_CHECKPOINT")" == "$ADMISSION_CHECKPOINT_SHA" ]] || \
    die "Sealed admission checkpoint SHA mismatch"

  local config_sha_path="$WORKSPACE/runpod_operator.env.sha256"
  local config_sha
  config_sha="$(sha256_file "$OPERATOR_CONFIG")"
  if [[ -f "$config_sha_path" ]]; then
    [[ "$(awk 'NR==1 {print $1}' "$config_sha_path")" == "$config_sha" ]] || \
      die "RunPod operator configuration differs from its sealed digest"
  else
    local temporary_config_sha="$config_sha_path.partial"
    printf '%s  %s\n' "$config_sha" "$(basename "$OPERATOR_CONFIG")" > \
      "$temporary_config_sha"
    mv -- "$temporary_config_sha" "$config_sha_path"
  fi

  printf 'Provider and durable storage checks passed. No training was launched.\n'
  printf 'Conservative all-in hourly rate: $%s\n' "$(all_in_hourly_rate)"
  printf 'Next command: %s evidence\n' "$0"
}

evidence() {
  assert_provider
  assert_release_tree
  require_file "$ADMISSION_CHECKPOINT"
  [[ "$(sha256_file "$ADMISSION_CHECKPOINT")" == "$ADMISSION_CHECKPOINT_SHA" ]] || \
    die "Admission checkpoint changed"

  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.metric_calibration \
    --data-root "$DATA_ROOT" --samples-per-side 25000 \
    --left-dir "$EVIDENCE/real_left_pngs" \
    --right-dir "$EVIDENCE/real_right_pngs" \
    --kid-reference-features-out "$KID_REFERENCE" \
    --metric-batch 128 --metric-workers 0 --out "$METRIC_CALIBRATION"

  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.standard_metrics \
    --checkpoint "$ADMISSION_CHECKPOINT" --device cuda --data-root "$DATA_ROOT" \
    --png-dir "$EVIDENCE/baseline_pngs" --batch 128 --metric-batch 128 \
    --feature-batch 128 --kid-reference-features "$KID_REFERENCE" \
    --generated-features "$EVIDENCE/baseline_clean_features.npz" \
    --metric-workers 0 --out "$BASELINE_STANDARD"

  local stylegan_repo="$EXTERNAL/stylegan2-ada-pytorch"
  local stylegan_network="$EXTERNAL/cifar10-stylegan2-ada.pkl"
  local stylegan_revision="d72cc7d041b42ec8e806021a205ed9349f87c6a4"
  local stylegan_sha="f8952c74e23da2186d147ad871c48780bd59500ee37c301201081ee8e0cb32f1"
  if [[ ! -d "$stylegan_repo/.git" ]]; then
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git "$stylegan_repo"
    git -C "$stylegan_repo" checkout --detach "$stylegan_revision"
  fi
  [[ "$(git -C "$stylegan_repo" rev-parse HEAD)" == "$stylegan_revision" ]] || \
    die "StyleGAN2-ADA checkout revision changed"
  if [[ ! -f "$stylegan_network" ]]; then
    curl --fail --location \
      https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl \
      --output "$stylegan_network"
  fi
  [[ "$(sha256_file "$stylegan_network")" == "$stylegan_sha" ]] || \
    die "StyleGAN2-ADA network SHA mismatch"

  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.positive_control \
    --stylegan-repo "$stylegan_repo" --network "$stylegan_network" \
    --out-dir "$EVIDENCE/positive_control_pngs" \
    --provenance-out "$EVIDENCE/positive_control_source.json" \
    --device cuda --batch 100
  local citation
  citation="$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["citation"])' \
    "$EVIDENCE/positive_control_source.json")"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.standard_metrics \
    --existing-png-dir "$EVIDENCE/positive_control_pngs" \
    --external-source-citation "$citation" \
    --external-source-provenance "$EVIDENCE/positive_control_source.json" \
    --kid-reference-features "$KID_REFERENCE" --metric-batch 128 \
    --feature-batch 128 \
    --generated-features "$EVIDENCE/positive_control_clean_features.npz" \
    --metric-workers 0 --device cuda --data-root "$DATA_ROOT" \
    --out "$POSITIVE_CONTROL_STANDARD"

  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.sampler_audit \
    --count 2000000 --out "$SAMPLER_AUDIT"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.gate_calibration \
    --data-root "$DATA_ROOT" --samples 2048 --repeats 12 \
    --out "$GATE_CALIBRATION"
  printf 'Fresh source-bound evidence complete. No foundation training was launched.\n'
}

admission() {
  assert_provider
  assert_release_tree
  local required
  for required in "$ADMISSION_CHECKPOINT" "$KID_REFERENCE" \
    "$METRIC_CALIBRATION" "$BASELINE_STANDARD" "$POSITIVE_CONTROL_STANDARD" \
    "$SAMPLER_AUDIT" "$GATE_CALIBRATION"; do
    require_file "$required"
  done
  local hourly
  hourly="$(all_in_hourly_rate)"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.production_readiness \
    --checkpoint "$ADMISSION_CHECKPOINT" \
    --checkpoint-sha256 "$ADMISSION_CHECKPOINT_SHA" \
    --expected-gpu-name "$EXPECTED_GPU" --hourly-rate "$hourly" \
    --micro-batch "$MICRO_BATCH" --data-root "$DATA_ROOT" \
    --output-dir "$GATES/production" --sampler-audit "$SAMPLER_AUDIT" \
    --gate-calibration "$GATE_CALIBRATION" --baseline-standard "$BASELINE_STANDARD" \
    --positive-control-standard "$POSITIVE_CONTROL_STANDARD" \
    --metric-calibration "$METRIC_CALIBRATION" --max-total-cost "$MAX_TOTAL_COST" \
    --nontraining-reserve 10 --post-foundation-training-reserve "$ASFD_RESERVE" \
    --contingency-fraction 0.15 --campaign ordered_750_foundation \
    --durable-mirror-dir "$MIRROR" --durable-storage-root "$VOLUME_ROOT" \
    --artifact-storage-reserve-gib 20 --storage-contingency-fraction 0.20 \
    --i-confirm-durable-mirror --i-have-authorized-production-gates
  require_file "$CAP2_PREFLIGHT"
  printf 'Production admission returned GO. Review its projection before Phase A.\n'
}

foundation_phase_a() {
  assert_provider
  assert_release_tree
  require_file "$CAP2_PREFLIGHT"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.run_screen \
    --arm ordered_uniform --preflight "$CAP2_PREFLIGHT" --updates 750000 \
    --device cuda --data-root "$DATA_ROOT" --output-dir "$FOUNDATION_RUN" \
    --durable-mirror-dir "$MIRROR/foundation" --i-confirm-durable-mirror \
    --durable-workspace-dir "$WORKSPACE" --i-confirm-durable-workspace \
    --pause-for-early-admission --i-have-authorized-the-screen-run
  printf 'Foundation paused at 50,000. Do not start Phase B before readmission.\n'
}

foundation_admit_50k() {
  assert_provider
  assert_release_tree
  local raw50="$FOUNDATION_RUN/checkpoints/cap2_ordered_uniform_step50000_raw.pt"
  local result50="$FOUNDATION_RUN/result_50000.json"
  require_file "$raw50"
  require_file "$result50"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission \
    --checkpoint "$raw50" --candidate local_1000_d0002_fp32 --device cuda \
    --batch 4 --repeats 3 --data-root "$DATA_ROOT" \
    --expected-gpu-name "$EXPECTED_GPU" --include-gradient \
    --out "$FOUNDATION_READMISSION"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.early_admission \
    --preflight "$CAP2_PREFLIGHT" --result-50k "$result50" \
    --checkpoint-50k-raw "$raw50" --readmission-50k-raw "$FOUNDATION_READMISSION" \
    --out "$FOUNDATION_ADMISSION"
  printf 'The exact 50k recovery is authorized for continuation.\n'
}

foundation_phase_b() {
  assert_provider
  assert_release_tree
  require_file "$FOUNDATION_ADMISSION"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.run_screen \
    --arm ordered_uniform --preflight "$CAP2_PREFLIGHT" --updates 750000 \
    --device cuda --data-root "$DATA_ROOT" --output-dir "$FOUNDATION_RUN" \
    --early-admission "$FOUNDATION_ADMISSION" \
    --durable-mirror-dir "$MIRROR/foundation" --i-confirm-durable-mirror \
    --durable-workspace-dir "$WORKSPACE" --i-confirm-durable-workspace \
    --i-have-authorized-the-screen-run
}

foundation_evaluate() {
  assert_provider
  assert_release_tree
  local result750="$FOUNDATION_RUN/result_750000.json"
  local raw750="$FOUNDATION_RUN/checkpoints/cap2_ordered_uniform_step750000_raw.pt"
  require_file "$result750"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.development_evaluation \
    --unit "$result750" --step 650000 --device cuda --data-root "$DATA_ROOT" \
    --png-dir "$EVIDENCE/foundation_650k_pngs" \
    --grid "$EVIDENCE/foundation_650k_grid.png" \
    --kid-reference-features "$KID_REFERENCE" \
    --generated-features "$EVIDENCE/foundation_650k_features.npz" \
    --out "$EVIDENCE/foundation_650k_eval.json"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.development_evaluation \
    --unit "$result750" --step 750000 --device cuda --data-root "$DATA_ROOT" \
    --png-dir "$EVIDENCE/foundation_750k_pngs" \
    --grid "$EVIDENCE/foundation_750k_grid.png" \
    --kid-reference-features "$KID_REFERENCE" \
    --generated-features "$EVIDENCE/foundation_750k_features.npz" \
    --out "$EVIDENCE/foundation_750k_eval.json"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.numerical_admission \
    --checkpoint "$raw750" --candidate local_1000_d0002_fp32 --device cuda \
    --expected-gpu-name "$EXPECTED_GPU" --data-root "$DATA_ROOT" \
    --batch 4 --repeats 3 --include-gradient \
    --out "$GATES/raw_750k_readmission.json"
  printf 'Inspect this fixed grid before the next command:\n  %s\n' \
    "$EVIDENCE/foundation_750k_grid.png"
}

foundation_review() {
  assert_provider
  assert_release_tree
  require_var CAP_ASFD_REVIEWER
  require_var CAP_ASFD_FOUNDATION_DECISION
  [[ "$CAP_ASFD_FOUNDATION_DECISION" == PASS || \
     "$CAP_ASFD_FOUNDATION_DECISION" == FAIL ]] || \
    die "CAP_ASFD_FOUNDATION_DECISION must be PASS or FAIL"
  local ack='I reviewed the fixed uncurated grid without selecting samples'
  local review_code=0
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.foundation_visual_review \
    --evaluation "$EVIDENCE/foundation_750k_eval.json" \
    --decision "$CAP_ASFD_FOUNDATION_DECISION" --reviewer "$CAP_ASFD_REVIEWER" \
    --acknowledgement "$ack" --out "$GATES/foundation_visual_review.json" || \
    review_code=$?
  if [[ "$CAP_ASFD_FOUNDATION_DECISION" == PASS && "$review_code" -ne 0 ]]; then
    return "$review_code"
  fi
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.foundation_gate \
    --preflight "$CAP2_PREFLIGHT" --result "$FOUNDATION_RUN/result_750000.json" \
    --recovery "$FOUNDATION_RUN/checkpoints/recovery.pt" \
    --raw-readmission "$GATES/raw_750k_readmission.json" \
    --evaluation-650k "$EVIDENCE/foundation_650k_eval.json" \
    --evaluation-750k "$EVIDENCE/foundation_750k_eval.json" \
    --visual-review "$GATES/foundation_visual_review.json" \
    --mirror-root "$MIRROR/foundation" --out "$FOUNDATION_GATE"
}

asfd_prepare() {
  assert_provider
  assert_release_tree
  require_file "$FOUNDATION_GATE"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_asfd.qualify \
    --foundation-gate "$FOUNDATION_GATE" --device cuda --data-root "$DATA_ROOT" \
    --batch 32 --out "$ASFD_QUALIFICATION"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_asfd.feature_bank \
    --qualification "$ASFD_QUALIFICATION" --device cuda --data-root "$DATA_ROOT" \
    --batch 32 --output-dir "$WORKSPACE/asfd_feature_banks"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_asfd.preflight \
    --foundation-gate "$FOUNDATION_GATE" --qualification "$ASFD_QUALIFICATION" \
    --feature-bank "$WORKSPACE/asfd_feature_banks/feature_bank.json" \
    --work-dir "$WORKSPACE/asfd_preflight_work" --device cuda \
    --data-root "$DATA_ROOT" --out "$ASFD_PREFLIGHT"
  printf 'ASFD measured preflight returned GO. Review its wall/cost projection.\n'
}

asfd_run() {
  assert_provider
  assert_release_tree
  require_file "$ASFD_PREFLIGHT"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_asfd.continuation \
    --preflight "$ASFD_PREFLIGHT" --output-dir "$ASFD_RUN" \
    --durable-mirror-dir "$MIRROR/asfd" --durable-workspace-dir "$WORKSPACE" \
    --durable-storage-root "$VOLUME_ROOT" --device cuda --data-root "$DATA_ROOT" \
    --i-confirm-durable-mirror --i-confirm-durable-workspace \
    --i-have-authorized-asfd-continuation
}

final_evaluate() {
  assert_provider
  assert_release_tree
  require_file "$ASFD_RUN/asfd_result.json"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_asfd.evaluation \
    --result "$ASFD_RUN/asfd_result.json" --device cuda --data-root "$DATA_ROOT" \
    --png-dir "$EVIDENCE/asfd_800k_pngs" --grid "$EVIDENCE/asfd_800k_grid.png" \
    --kid-reference-features "$KID_REFERENCE" \
    --generated-features "$EVIDENCE/asfd_800k_features.npz" \
    --out "$EVIDENCE/asfd_800k_eval.json"
  printf 'Inspect both fixed grids before final-review:\n  %s\n  %s\n' \
    "$EVIDENCE/foundation_750k_grid.png" "$EVIDENCE/asfd_800k_grid.png"
}

final_review() {
  assert_provider
  assert_release_tree
  require_var CAP_ASFD_REVIEWER
  require_var CAP_ASFD_FINAL_DECISION
  [[ "$CAP_ASFD_FINAL_DECISION" == PASS || "$CAP_ASFD_FINAL_DECISION" == FAIL ]] || \
    die "CAP_ASFD_FINAL_DECISION must be PASS or FAIL"
  local ack='I reviewed both fixed uncurated grids without selecting or replacing samples'
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_asfd.final_visual_review \
    --foundation-evaluation "$EVIDENCE/foundation_750k_eval.json" \
    --asfd-evaluation "$EVIDENCE/asfd_800k_eval.json" \
    --decision "$CAP_ASFD_FINAL_DECISION" --reviewer "$CAP_ASFD_REVIEWER" \
    --acknowledgement "$ack" --out "$EVIDENCE/asfd_final_visual_review.json"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_asfd.final_report \
    --foundation-gate "$FOUNDATION_GATE" --continuation "$ASFD_RUN/asfd_result.json" \
    --asfd-evaluation "$EVIDENCE/asfd_800k_eval.json" \
    --visual-review "$EVIDENCE/asfd_final_visual_review.json" \
    --out "$EVIDENCE/asfd_final_report.json"
}

restore_foundation() {
  assert_provider
  assert_release_tree
  [[ ! -e "$FOUNDATION_RUN" ]] || \
    die "Move the damaged foundation directory aside before an immutable restore"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror \
    restore --mirror-dir "$MIRROR/foundation" --output-dir "$FOUNDATION_RUN"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror \
    verify --mirror-dir "$MIRROR/foundation" --output-dir "$FOUNDATION_RUN"
}

restore_asfd() {
  assert_provider
  assert_release_tree
  [[ ! -e "$ASFD_RUN" ]] || \
    die "Move the damaged ASFD directory aside before an immutable restore"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror \
    restore --mirror-dir "$MIRROR/asfd" --output-dir "$ASFD_RUN"
  "$PYTHON" -m numerics.encoder_independent_drifting.stage_cap2.durable_mirror \
    verify --mirror-dir "$MIRROR/asfd" --output-dir "$ASFD_RUN"
}

status_report() {
  assert_provider
  printf 'release=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  printf 'workspace=%s\nmirror=%s\n' "$WORKSPACE" "$MIRROR"
  df -h -- "$VOLUME_ROOT"
  local path
  for path in "$CAP2_PREFLIGHT" "$FOUNDATION_ADMISSION" \
    "$FOUNDATION_RUN/result_750000.json" "$FOUNDATION_GATE" \
    "$ASFD_PREFLIGHT" "$ASFD_RUN/asfd_result.json" \
    "$EVIDENCE/asfd_final_report.json"; do
    if [[ -f "$path" ]]; then printf 'present %s\n' "$path"; else printf 'absent  %s\n' "$path"; fi
  done
}

command_name="${1:-help}"
case "$command_name" in
  help|-h|--help) usage ;;
  prepare) prepare ;;
  evidence) evidence ;;
  admission) admission ;;
  foundation-phase-a) foundation_phase_a ;;
  foundation-admit-50k) foundation_admit_50k ;;
  foundation-phase-b) foundation_phase_b ;;
  foundation-evaluate) foundation_evaluate ;;
  foundation-review) foundation_review ;;
  asfd-prepare) asfd_prepare ;;
  asfd-run) asfd_run ;;
  final-evaluate) final_evaluate ;;
  final-review) final_review ;;
  restore-foundation) restore_foundation ;;
  restore-asfd) restore_asfd ;;
  status) status_report ;;
  *) usage >&2; die "Unknown command: $command_name" ;;
esac
