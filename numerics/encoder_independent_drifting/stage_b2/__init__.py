"""Audited Stage-B2 implementation, isolated from the frozen B1 manifest."""

import os

# CUDA deterministic matmul requires this to be present before the first
# cuBLAS operation.  Setting it at the stage-package boundary keeps the command
# lines replayable and makes ``torch.use_deterministic_algorithms(True)`` honest.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .core import (
    B2Config,
    b2_config,
    calibrate_laplace_bandwidth,
    laplace_drift_energy,
    laplace_mean_shift_field,
)

__all__ = [
    "B2Config",
    "b2_config",
    "calibrate_laplace_bandwidth",
    "laplace_drift_energy",
    "laplace_mean_shift_field",
]
