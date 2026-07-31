"""Pre-serialize exact and KLL atlases before confirmatory model training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from neural_pooled_rank import apache_kll_target_atlas, exact_target_atlas
from run_conditioned_transport_development import (
    PROFILES,
    conditioned_directions,
    sha256_file,
)
from run_neural_pooled_rank_development import normalized_target_data


HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "neural_conditioned_confirmatory_registry_v3.json"
REGISTRY_HASH = REGISTRY.with_suffix(REGISTRY.suffix + ".sha256")
PROTOCOL = HERE / "NeuralConditionedTransportConfirmatoryProtocol.md"
TABLES = HERE / "neural_conditioned_confirmatory_atlases_v3.npz"
STATES = HERE / "neural_conditioned_confirmatory_kll_v3.bin"
STATE_INDEX = HERE / "neural_conditioned_confirmatory_kll_index_v3.json"
FREEZE = HERE / "neural_conditioned_confirmatory_freeze_v3.json"
SOURCES = (
    PROTOCOL,
    HERE / "generate_neural_conditioned_registry.py",
    HERE / "freeze_neural_conditioned_atlases.py",
    HERE / "run_neural_conditioned_confirmatory.py",
    HERE / "NeuralConditionedTransportFailures.md",
    HERE / "neural_pooled_rank.py",
    HERE / "neural_pooled_rank_tests.py",
    HERE / "run_conditioned_transport_development.py",
    HERE / "run_neural_pooled_rank_development.py",
)


def load_registry() -> tuple[dict, str]:
    payload = REGISTRY.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = REGISTRY_HASH.read_text(encoding="utf-8").split()[0]
    registry = json.loads(payload)
    if digest != expected:
        raise RuntimeError("registry sidecar mismatch")
    if (
        registry.get("schema") != "neural-conditioned-confirmatory-v1"
        or registry.get("target_count") != 64
        or registry.get("rejection_count") != 0
    ):
        raise RuntimeError("unexpected confirmatory registry")
    return registry, digest


def main() -> None:
    outputs = (TABLES, STATES, STATE_INDEX, FREEZE)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen files: {existing}")
    missing = [path for path in SOURCES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"freeze sources are missing: {missing}")
    registry, registry_digest = load_registry()
    profile = PROFILES["consumed"]
    arrays: dict[str, np.ndarray] = {}
    index: list[dict] = []
    target_metadata: list[dict] = []
    position = 0
    with STATES.open("wb") as stream:
        for target in registry["targets"]:
            target_index = int(target["index"])
            prefix = f"t{target_index:03d}"
            data = normalized_target_data(target, 0, profile)
            directions, diagnostics = conditioned_directions(target)
            exact = exact_target_atlas(data.pool, directions, profile.atlas_knots)
            kll = apache_kll_target_atlas(
                data.pool, directions, profile.atlas_knots, k=128
            )
            arrays[f"{prefix}_directions"] = directions
            arrays[f"{prefix}_exact_grid"] = exact.grid
            arrays[f"{prefix}_exact_quantiles"] = exact.quantiles
            arrays[f"{prefix}_kll_grid"] = kll.grid
            arrays[f"{prefix}_kll_quantiles"] = kll.quantiles
            payloads = kll.sketch_payloads or ()
            for direction, payload in enumerate(payloads):
                stream.write(payload)
                index.append(
                    {
                        "target_index": target_index,
                        "target": target["name"],
                        "direction": direction,
                        "offset": position,
                        "length": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                position += len(payload)
            target_metadata.append(
                {
                    "target_index": target_index,
                    "target": target["name"],
                    **diagnostics,
                    "kll_normalized_rank_error": kll.normalized_rank_error,
                    "kll_state_count": len(payloads),
                    "kll_serialized_bytes": sum(map(len, payloads)),
                }
            )
            print(
                f"froze {target['name']}: L={len(directions)}, "
                f"kappa={diagnostics['quadratic_condition_number']:.2f}",
                flush=True,
            )
    np.savez_compressed(TABLES, **arrays)
    STATE_INDEX.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "schema": "neural-conditioned-freeze-v1",
        "registry_sha256": registry_digest,
        "target_count": len(registry["targets"]),
        "atlas_knots": profile.atlas_knots,
        "kll_k": 128,
        "target_metadata": target_metadata,
        "artifact_sha256": {
            TABLES.name: sha256_file(TABLES),
            STATES.name: sha256_file(STATES),
            STATE_INDEX.name: sha256_file(STATE_INDEX),
        },
        "source_sha256": {
            str(path.relative_to(HERE.parent)): sha256_file(path) for path in SOURCES
        },
    }
    FREEZE.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {FREEZE}")
    print(f"sha256 {sha256_file(FREEZE)}")


if __name__ == "__main__":
    main()
