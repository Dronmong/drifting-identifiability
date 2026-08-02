"""Stage B2.5: prospective B0/B1/B2/B1+B2 development factorial.

The public training names are loaded lazily.  In particular, invoking the
data-provenance builder must not import Torch merely because it lives inside
this package.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "B25_ARMS",
    "B25_PHASE",
    "B25_UNITS",
    "B25Config",
    "B25TrainResult",
    "b25_config",
    "train_b25_arm",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import core

    return getattr(core, name)
