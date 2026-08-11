"""Small end-to-end execution test for the actual ASFD correction graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...stage_cap.config import CAPModelConfig
from ...stage_cap.model import CAPPixelTransformer
from .. import correction as correction_module
from ..artifacts import file_sha256
from ..config import LEVEL_NAMES, smoke_config
from ..correction import ASFDCorrection


def _tiny_model(seed: int) -> CAPPixelTransformer:
    return CAPPixelTransformer(
        CAPModelConfig(
            width=32,
            depth=12,
            heads=4,
            mlp_ratio=2.0,
            time_embedding_dim=32,
            condition_dim=32,
            refiner_width=8,
            refiner_depth=1,
        ),
        seed=seed,
    )


def test_actual_correction_forward_backward_and_state_roundtrip(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise B1, raw and self-feature losses through one generated batch.

    Unit tests of the individual field/gradient helpers cannot detect a broken
    tensor layout or a detached frozen-teacher input in their composition.
    This mechanics test uses synthetic target data but the production classes
    and autograd graph end to end.
    """

    torch.manual_seed(41)
    pool = torch.rand(256, 3, 32, 32) * 2 - 1
    teacher = _tiny_model(3)
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"synthetic teacher identity")
    checkpoint_sha = file_sha256(checkpoint)
    bank_path = tmp_path / "feature_bank.json"
    bank_path.write_text("{}\n", encoding="utf-8")
    qualification_sha = "a" * 64

    class FakeBank:
        def __init__(self, _path: Path) -> None:
            self.payload = {
                "teacher_checkpoint": {"sha256": checkpoint_sha},
                "qualification": {"sha256": qualification_sha},
                "dataset_binding": {"synthetic": True},
                "roles": {
                    role: {"rows": 256}
                    for role in (
                        "train_positive",
                        "train_probe",
                        "fresh_positive",
                        "fresh_probe",
                    )
                },
            }

        def sample(
            self, role: str, indices: np.ndarray, *, device: torch.device
        ) -> dict[str, torch.Tensor]:
            rows = torch.as_tensor(indices, dtype=torch.float32).reshape(-1, 1, 1)
            location = torch.arange(66, dtype=torch.float32).reshape(1, 66, 1)
            channel = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32)
            role_shift = 0.17 * tuple(self.payload["roles"]).index(role)
            return {
                level: torch.sin(
                    0.013 * rows
                    + 0.031 * location
                    + 0.047 * channel
                    + role_shift
                    + level_index
                ).to(device)
                for level_index, level in enumerate(LEVEL_NAMES)
            }

        def raw_batch(
            self,
            _role: str,
            indices: np.ndarray,
            source: torch.Tensor,
            *,
            device: torch.device,
        ) -> torch.Tensor:
            selected = torch.as_tensor(indices % len(source), dtype=torch.long)
            return source[selected].clone().to(device)

    monkeypatch.setattr(correction_module, "FeatureBank", FakeBank)
    monkeypatch.setattr(correction_module, "cifar10_train_pool", lambda _root: pool)
    monkeypatch.setattr(
        correction_module,
        "cifar10_train_labels",
        lambda _root: torch.arange(len(pool), dtype=torch.int64) % 10,
    )
    monkeypatch.setattr(
        correction_module, "verify_dataset_binding", lambda *_args: {"synthetic": True}
    )
    monkeypatch.setattr(
        correction_module,
        "_load_model",
        lambda _path, _device: (
            {"kind": "ema", "step": 750_000},
            teacher,
            teacher.config,
        ),
    )

    normalization = {
        level: {
            "channel_scale": None,
            "level_scale": 1.0,
            "pc1_share": 0.1,
            "per_channel_applied": False,
        }
        for level in LEVEL_NAMES
    }
    raw_bandwidths = {
        str(radius): {"tau": 30.0} for radius in smoke_config().field_config.radii
    }
    feature_bandwidths = {
        level: {
            str(radius): {"tau": 10.0} for radius in smoke_config().field_config.radii
        }
        for level in LEVEL_NAMES
    }
    qualification = {
        "decision": "GO",
        "artifact_sha256": qualification_sha,
        "teacher_checkpoint": {"sha256": checkpoint_sha},
        "selected": {
            "t_f": 0.1,
            "normalization": normalization,
            "bandwidths": {
                "raw": raw_bandwidths,
                "feature": feature_bandwidths,
            },
        },
    }
    correction = ASFDCorrection(
        teacher_checkpoint=checkpoint,
        bank_metadata=bank_path,
        qualification=qualification,
        coefficients={"b1": 1e-3, "raw": 1e-3, "self": 1e-3},
        spectral_scale=1.0,
        device=torch.device("cpu"),
        data_root=None,
        config=smoke_config(),
    )
    model = _tiny_model(7)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter) * 1e-3
    record = correction.apply(750_010, model)
    assert record is not None
    assert record["event"] == 1
    assert record["generated_forwards"] == 1
    assert record["generated_input_gradient_norm"] > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    saved = correction.state_dict()
    correction.load_state_dict(saved)
    assert correction.event_count == 1
    assert correction.generated_forwards == 1
    assert correction.state_dict()["role_rng"] == saved["role_rng"]
