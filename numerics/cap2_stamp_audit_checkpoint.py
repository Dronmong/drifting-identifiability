"""Add the identity fields a candidate-audit checkpoint needs, without editing it.

``numerical_admission`` requires ``stage``, ``kind`` and ``parameter_count`` to
validate a checkpoint's identity. The candidate diagnostic did not write them
until they were needed, so checkpoints produced before that change are refused
on ``protocol_checks.checkpoint_identity`` despite passing every numerical test.

This writes a **new** checkpoint carrying those fields. It never modifies its
input, and it adds nothing that is not already true of the file:

* ``stage`` is ``cap-emf-2-candidate-audit`` -- the non-promoting audit path
  that actually produced it, never the screen's label;
* ``kind`` is taken from the caller and must be ``raw`` or ``ema``;
* ``parameter_count`` is recomputed from the profile's own model config and
  cross-checked against the stored ``state_dict``, so a mismatched pairing is
  refused rather than stamped.

The source SHA-256 is recorded in the output so the provenance chain stays
intact, and the weights are compared tensor by tensor after the round trip.
Regenerating the checkpoint by retraining would be equally correct and costs
several GPU-hours; this is the same artifact with an accurate label.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from numerics.encoder_independent_drifting.stage_cap.config import CAPModelConfig
from numerics.encoder_independent_drifting.stage_cap.model import CAPPixelTransformer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=("raw", "ema"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for required in ("state_dict", "profile", "step"):
        if required not in payload:
            raise SystemExit(f"checkpoint lacks {required!r}; not an audit checkpoint")
    if "stage" in payload:
        raise SystemExit(
            f"checkpoint already declares stage {payload['stage']!r}; refusing to "
            "relabel an artifact that already has an identity"
        )

    model_config = CAPModelConfig(**payload["profile"]["model"])
    model = CAPPixelTransformer(model_config, seed=1)
    # Refuse to stamp a state dict that does not belong to this profile.
    model.load_state_dict(payload["state_dict"])

    source_digest = sha256(args.checkpoint)
    payload["stage"] = "cap-emf-2-candidate-audit"
    payload["kind"] = args.kind
    payload["parameter_count"] = model.parameter_count()
    payload["stamped_from"] = {
        "path": str(args.checkpoint),
        "sha256": source_digest,
        "note": (
            "identity fields added to a checkpoint written before the diagnostic "
            "recorded them; weights, profile and step are unchanged"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    # Prove the weights survived the round trip before reporting success.
    restored = torch.load(args.out, map_location="cpu", weights_only=True)
    original = payload["state_dict"]
    if set(restored["state_dict"]) != set(original):
        raise SystemExit("stamped checkpoint lost or gained parameters")
    for key, value in original.items():
        if not torch.equal(restored["state_dict"][key], value):
            raise SystemExit(f"stamped checkpoint altered parameter {key!r}")

    print(f"source sha256    : {source_digest}")
    print(f"parameter_count  : {payload['parameter_count']}")
    print(f"step / kind      : {payload['step']} / {args.kind}")
    print(f"weights verified : identical across {len(original)} tensors")
    print(f"wrote            : {args.out}")


if __name__ == "__main__":
    main()
