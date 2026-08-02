#!/usr/bin/env bash
# B2.5 driver, mirroring stage_b3/run_overnight.ps1's safety properties:
# skip completed units, refuse half-written artifacts, never overwrite the
# aggregate. Safe to re-invoke after an interruption.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
UV=(uv run --python 3.12
    --extra-index-url https://download.pytorch.org/whl/cu126
    --index-strategy unsafe-best-match
    --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126
    --with numpy --with scipy --with pillow python -m)
D=numerics/encoder_independent_drifting/stage_b25
for U in 500 501 502; do
  R="$D/b25_unit_$U.json"; S="$R.sha256"
  if [ -f "$R" ] && [ -f "$S" ]; then echo "Skipping completed B2.5 unit $U"; continue; fi
  if [ -f "$R" ] || [ -f "$S" ]; then echo "Incomplete artifact for unit $U; inspect before resuming" >&2; exit 1; fi
  echo "=== B2.5 unit $U ==="
  "${UV[@]}" numerics.encoder_independent_drifting.stage_b25.run_unit --unit "$U" --device cuda
done
A="$D/b25_development.json"
if [ -f "$A" ] || [ -f "$A.sha256" ]; then echo "B2.5 aggregate exists; refusing to overwrite" >&2; exit 1; fi
"${UV[@]}" numerics.encoder_independent_drifting.stage_b25.aggregate
echo "B2.5 development run and aggregation completed."
