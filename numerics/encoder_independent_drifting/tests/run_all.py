"""Run every encoder-independent drifting unit test (plan section 9, P0.1-P0.4).

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.tests.run_all
"""

from __future__ import annotations

import sys

import torch

from . import (
    test_b1,
    test_crossfit_controller,
    test_f1_k200,
    test_f3b,
    test_fixed_features,
    test_kernel_gradients,
    test_phase2,
    test_positive_kernel_mixture,
    test_reforms,
    test_reproducibility,
    test_spectral_anchor,
)
from .harness import run_module

MODULES = (
    ("spectral anchor (P0.1)", test_spectral_anchor),
    ("fixed features (P0.2)", test_fixed_features),
    ("kernel gradients (P0.3)", test_kernel_gradients),
    ("positive kernel mixture (P0.4)", test_positive_kernel_mixture),
    ("cross-fit controller (P0.4)", test_crossfit_controller),
    ("reproducibility", test_reproducibility),
    ("diagnosis reforms (R1, R4)", test_reforms),
    ("phase 2 (CIFAR target and arms)", test_phase2),
    ("F1 K=200 confirmation gate", test_f1_k200),
    ("F3B prescribed bridge", test_f3b),
    ("B1 paired spectral anchor", test_b1),
)


def main() -> None:
    torch.set_num_threads(1)
    failures = 0
    for name, module in MODULES:
        failures += run_module(name, vars(module))
    print()
    if failures:
        print(f"PHASE 0 UNIT TESTS: {failures} FAILED")
        sys.exit(1)
    print("PHASE 0 UNIT TESTS: all passed")


if __name__ == "__main__":
    main()
