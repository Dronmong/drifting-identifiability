#!/usr/bin/env bash
# Bootstrap the single frozen CAP-ASFD environment on a RunPod Secure Cloud Pod.
# This script installs software only. It never launches training.

set -Eeuo pipefail

on_error() {
  local line="$1"
  local code="$2"
  printf 'RunPod bootstrap failed at line %s (exit %s). No training was launched.\n' \
    "$line" "$code" >&2
}
trap 'on_error "$LINENO" "$?"' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
VOLUME_ROOT="${RUNPOD_VOLUME_ROOT:-/workspace}"
VENV="${RUNPOD_VENV:-$VOLUME_ROOT/cap_asfd_venv}"
PYTHON_VERSION="3.11.15"
UV_VERSION="0.8.14"
REQUIREMENTS="$SCRIPT_DIR/requirements-production-cu126.txt"

usage() {
  cat <<'EOF'
Usage: runpod_bootstrap.sh

Install and verify the pinned CAP-ASFD production environment on a RunPod Pod
whose persistent Network Volume is mounted at /workspace. This command never
launches training.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ ! -d "$VOLUME_ROOT" ]]; then
  printf 'Expected the RunPod Network Volume at %s, but it is not mounted.\n' \
    "$VOLUME_ROOT" >&2
  exit 2
fi
if [[ ! -f "$REQUIREMENTS" ]]; then
  printf 'Pinned production requirements are missing: %s\n' "$REQUIREMENTS" >&2
  exit 2
fi
if [[ "$(id -u)" -ne 0 ]]; then
  printf 'Bootstrap must run as root inside the RunPod Pod.\n' >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl git libgl1 libglib2.0-0 tmux
rm -rf /var/lib/apt/lists/*

if ! command -v uv >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -LsSf \
    "https://astral.sh/uv/${UV_VERSION}/install.sh" | \
    env UV_INSTALL_DIR=/usr/local/bin sh
fi

installed_uv="$(uv --version | awk '{print $2}')"
if [[ "$installed_uv" != "$UV_VERSION" ]]; then
  printf 'Expected uv %s, found %s. Refusing an unpinned bootstrap.\n' \
    "$UV_VERSION" "$installed_uv" >&2
  exit 2
fi

uv python install "$PYTHON_VERSION"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi

actual_python="$("$VENV/bin/python" -c 'import platform; print(platform.python_version())')"
if [[ "$actual_python" != "$PYTHON_VERSION" ]]; then
  printf 'Existing venv has Python %s, expected %s. Remove %s and retry.\n' \
    "$actual_python" "$PYTHON_VERSION" "$VENV" >&2
  exit 2
fi

uv pip install --python "$VENV/bin/python" --requirement "$REQUIREMENTS"

export PYTHONPATH="$REPO_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$VENV/bin/python" - <<'PY'
import os
import platform

import numpy
import PIL
import torch
import torchvision

expected = {
    "python": "3.11.15",
    "torch": "2.7.1+cu126",
    "torchvision": "0.22.1+cu126",
    "numpy": "1.26.4",
    "pillow": "12.2.0",
}
actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
}
if actual != expected:
    raise SystemExit(f"pinned environment mismatch: {actual!r} != {expected!r}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the RunPod container")
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise SystemExit("CUBLAS_WORKSPACE_CONFIG is not frozen")
print("Pinned environment verified:", actual)
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA runtime:", torch.version.cuda)
PY

printf '\nBootstrap complete. No training was launched.\n'
printf 'Python: %s\n' "$VENV/bin/python"
printf 'Next: export the variables in the RunPod protocol and run:\n'
printf '  bash %s/runpod_pipeline.sh prepare\n' "$SCRIPT_DIR"
