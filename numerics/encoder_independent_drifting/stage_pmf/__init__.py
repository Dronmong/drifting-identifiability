"""Local Stage-S3 pixel MeanFlow foundation.

This package is intentionally isolated from the older F3B velocity bridge.
It implements direct denoised-image prediction, the full MeanFlow time
triangle, a stopped JVP training target, and exactly one network evaluation at
inference.  No full S3 run is authorized merely by importing this package.
"""

from .config import INITIAL_UNITS, profile
from .model import PixelMeanFlowTransformer
from .objective import meanflow_loss, one_step_sample, sample_time_triangle

__all__ = [
    "INITIAL_UNITS",
    "PixelMeanFlowTransformer",
    "meanflow_loss",
    "one_step_sample",
    "profile",
    "sample_time_triangle",
]
