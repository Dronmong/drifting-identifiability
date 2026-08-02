"""Stage B3: matched reporting for the encoder-free one-step drifting proxy."""

from __future__ import annotations

from typing import Any

__all__ = (
    "B3_ARMS",
    "B3_UNITS",
    "B3ArmSpec",
    "B3Config",
    "b3_config",
    "train_b3_arm",
)


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(name)
