"""Encoder-independent kernel drifting (`SACKGD`).

Implementation of ``numerics/EncoderIndependentKernelDriftingResearchPlan.md``.

The package is deliberately isolated: it does not import the flagship
runner, the coherent-transport route planner, the PQST controller, or any
neural transport teacher (plan section 8).  Nothing here trains or loads a
pretrained feature encoder for the training objective; see
``reference_encoder.py`` for the clearly labelled reference arm.

Run from the repository root, e.g.::

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.tests.run_all

Nothing in this package proves anything.  Population statements about the
spectral anchor are ideal-expectation statements; every empirical quantity
is a finite random-feature approximation and is reported as such.
"""

from __future__ import annotations

__all__ = [
    "adaptive_mixture",
    "collision_suite",
    "config",
    "datasets",
    "diagnostics",
    "fixed_features",
    "kernel_gradient",
    "kernels",
    "metrics",
    "models",
    "objectives",
    "spectral_anchor",
]
