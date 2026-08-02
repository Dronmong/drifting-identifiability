"""Re-evaluate frozen B0/B1/B2 checkpoints on B3's exact instruments."""

from __future__ import annotations

import torch

from ..f3b import sample_model
from ..f3b_freeze import profile_from_payload
from ..stage_b2.artifacts import load_checkpoint_model
from .core import B3_UNITS, b3_seed

BRIDGE_UNITS = (300, 301, 302)


def bridge_unit_for_b3(unit: int) -> int:
    if unit not in B3_UNITS:
        raise ValueError(f"unknown B3 unit {unit}")
    return BRIDGE_UNITS[B3_UNITS.index(unit)]


def load_reference_models(references: dict[str, dict], unit: int, device) -> dict:
    if set(references) != {"B0", "B1", "B2"}:
        raise ValueError("B3 needs the exact B0/B1/B2 reference set")
    bridge_unit = bridge_unit_for_b3(unit)
    result = {}
    for name, payload in references.items():
        result[name] = load_checkpoint_model(
            payload["checkpoints"][str(bridge_unit)],
            payload["profile"],
            bridge_unit,
            device,
        )
    return result


def sample_reference_models(
    models: dict[str, torch.nn.Module],
    frozen_profile: dict,
    unit: int,
    count: int,
    device,
    *,
    role: str,
    event: int | None = None,
) -> dict[str, torch.Tensor]:
    """All bridge models see the same image-space prior draw."""
    selected = profile_from_payload(frozen_profile)
    seed = b3_seed(unit, role, event)
    return {
        name: sample_model(
            model,
            count,
            selected.model,
            selected.evaluation.nfe_ladder[0],
            seed,
            device,
        )
        for name, model in models.items()
    }
