"""Developmental repairs for the failed Stage-S3 one-step pixel foundation.

The package is deliberately separate from :mod:`stage_pmf`: the latter is a
frozen, completed experiment whose source hashes must remain reproducible.
Nothing in this package authorizes a long or test-set-visible run.
"""

from .config import S3R_ARMS, profile
from .model import RepairedPixelMeanFlowTransformer
from .objectives import alpha_flow_loss, emf_x1_loss, pmf_loss

__all__ = [
    "S3R_ARMS",
    "RepairedPixelMeanFlowTransformer",
    "alpha_flow_loss",
    "emf_x1_loss",
    "pmf_loss",
    "profile",
]
