"""Paired learning-rate screen for the Phase-1 pooled-rank smoke targets.

This is a development-only repair experiment.  It reuses the four already
consumed smoke targets and changes only Adam's learning rate.  It must not be
reported as confirmation or used to estimate generalization performance.

Run from the repository root with::

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      --with datasketches==5.2.0 \
      python numerics/run_neural_pooled_rank_optimization_screen.py
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

from neural_pooled_rank import apache_kll_target_atlas, exact_target_atlas
from run_neural_pooled_rank_development import (
    ADAM_LR,
    Generator,
    PROFILES,
    REGISTRY,
    REGISTRY_HASH,
    ROOT,
    evaluate_output,
    git,
    load_registry,
    mixed_seed,
    normalized_target_data,
    phase_invariants,
    safe_key,
    sha256_file,
    train_neural_arm,
)


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "neural_pooled_rank_optimization_runs"
SCREEN = (
    ("paper-neural", 0.25),
    ("paper-neural", 0.5),
    ("paper-neural", 1.0),
    ("paper-neural", 2.0),
    ("paper-neural", 4.0),
    ("paper-neural", 8.0),
    ("paper-neural", 16.0),
    ("paper-neural", 32.0),
    ("minibatch-sw", 0.25),
    ("minibatch-sw", 0.5),
    ("minibatch-sw", 1.0),
    ("minibatch-sw", 2.0),
    ("minibatch-sw", 4.0),
    ("minibatch-sw", 8.0),
    ("minibatch-sw", 16.0),
    ("minibatch-sw", 32.0),
    ("exact-atlas-rsr", 8.0),
    ("exact-atlas-rsr", 16.0),
    ("exact-atlas-rsr", 32.0),
    ("exact-atlas-rsr", 64.0),
    ("kll-atlas-rsr", 8.0),
    ("kll-atlas-rsr", 16.0),
    ("kll-atlas-rsr", 32.0),
    ("kll-atlas-rsr", 64.0),
    ("kll-small-rsr", 4.0),
    ("kll-small-rsr", 8.0),
    ("kll-small-rsr", 16.0),
    ("kll-small-rsr", 32.0),
    ("kll-paper-hybrid", 8.0),
    ("kll-paper-hybrid", 16.0),
    ("kll-paper-hybrid", 32.0),
    ("kll-paper-hybrid", 64.0),
)


class ScreenArtifact:
    def __init__(self, registry_digest: str, expected_rows: int) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = RUN_ROOT / f"{stamp}-lr-screen"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.lines: list[str] = []
        self.kll_index: list[dict[str, Any]] = []
        self.kll_stream = (self.directory / "kll_states.bin").open("wb")
        sources = [
            HERE / "run_neural_pooled_rank_optimization_screen.py",
            HERE / "run_neural_pooled_rank_development.py",
            HERE / "neural_pooled_rank.py",
            HERE / "neural_pooled_rank_tests.py",
            HERE / "NeuralPooledRankPhase1Protocol.md",
            HERE / "NeuralPooledRankPhase1SmokeResults.md",
            HERE / "KLLPSQTNeuralAmortizationResearch.md",
        ]
        status = git("status", "--porcelain")
        self.manifest = {
            "experiment": "neural-pooled-rank-development-lr-screen-v2",
            "status": "development-only; smoke targets already consumed",
            "screen": [
                {"arm": arm, "lr_multiplier": multiplier} for arm, multiplier in SCREEN
            ],
            "profile": asdict(PROFILES["smoke"]),
            "registry_sha256": registry_digest,
            "expected_rows": expected_rows,
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(status),
            "git_status": status.splitlines() if status else [],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "datasketches": importlib.metadata.version("datasketches"),
            "cuda_available": torch.cuda.is_available(),
            "source_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in sources
            },
            "command": sys.argv,
        }
        shutil.copy2(REGISTRY, self.directory / REGISTRY.name)
        shutil.copy2(REGISTRY_HASH, self.directory / REGISTRY_HASH.name)
        for source in sources:
            destination = self.directory / "source_snapshots" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self.lines.append(message)

    def record_kll(self, target: str, payloads: tuple[bytes, ...]) -> None:
        for direction, payload in enumerate(payloads):
            offset = self.kll_stream.tell()
            self.kll_stream.write(payload)
            self.kll_index.append(
                {
                    "target": target,
                    "direction": direction,
                    "offset": offset,
                    "length": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

    def finish(
        self,
        rows: list[dict[str, Any]],
        outputs: dict[str, np.ndarray],
        references: dict[str, np.ndarray],
        summary: dict[str, Any],
    ) -> None:
        self.kll_stream.close()
        if len(rows) != self.manifest["expected_rows"]:
            raise RuntimeError("row count does not match registered screen")
        with (self.directory / "rows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(self.directory / "outputs.npz", **outputs)
        np.savez_compressed(self.directory / "references.npz", **references)
        (self.directory / "kll_state_index.json").write_text(
            json.dumps(self.kll_index, indent=2) + "\n", encoding="utf-8"
        )
        (self.directory / "results.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.manifest.update(
            {
                "actual_rows": len(rows),
                "kll_state_count": len(self.kll_index),
                "wall_seconds": time.perf_counter() - self.started,
            }
        )
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.directory / "summary.md").write_text(
            "```text\n" + "\n".join(self.lines) + "\n```\n", encoding="utf-8"
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_setting: dict[str, Any] = {}
    for arm, multiplier in SCREEN:
        selected = [
            row
            for row in rows
            if row["arm"] == arm and row["lr_multiplier"] == multiplier
        ]
        key = f"{arm}@{multiplier:g}x"
        by_setting[key] = {
            "rows": len(selected),
            "median_ed2": float(np.median([row["ed2"] for row in selected])),
            "median_heldout_sw1": float(
                np.median([row["heldout_sw1"] for row in selected])
            ),
            "median_training_quantile_rmse": float(
                np.median([row["training_quantile_rmse"] for row in selected])
            ),
            "divergences": int(sum(row["diverged"] for row in selected)),
            "median_wall_seconds": float(
                np.median([row["wall_seconds"] for row in selected])
            ),
        }
    return {"by_setting": by_setting}


def main() -> None:
    profile = PROFILES["smoke"]
    profile.validate()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    registry, registry_digest = load_registry()
    targets = [target for target in registry["targets"] if target["smoke"]]
    run = ScreenArtifact(registry_digest, len(targets) * len(SCREEN))
    run.log("Neural pooled-rank optimization repair screen")
    run.log("  DEVELOPMENT ONLY: reuses four consumed smoke targets")
    run.log(f"  registry {registry_digest}")
    run.log(f"  {len(targets)} targets x {len(SCREEN)} paired settings")
    phase_invariants(registry, run.log)

    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for target in targets:
        target_started = time.perf_counter()
        dimension = int(target["dimension"])
        training_directions = np.asarray(target["training_directions"], dtype=float)
        heldout_directions = np.asarray(target["heldout_directions"], dtype=float)
        data = normalized_target_data(target, 0, profile)
        exact_atlas = exact_target_atlas(
            data.pool, training_directions, profile.atlas_knots
        )
        kll_atlas = apache_kll_target_atlas(
            data.pool, training_directions, profile.atlas_knots, k=128
        )
        if kll_atlas.sketch_payloads is None:
            raise AssertionError("Apache KLL atlas did not expose serialized states")
        run.record_kll(str(target["name"]), kll_atlas.sketch_payloads)
        references[safe_key(str(target["name"]))] = data.reference.astype(np.float32)

        model_seed = mixed_seed(int(target["seeds"]["model"]), 0, 11)
        base_model = Generator(dimension, "concentrated", model_seed).to(
            dtype=torch.float32
        )
        latent_rng = np.random.default_rng(
            mixed_seed(int(target["seeds"]["training_latent"]), 0, 20)
        )
        latent_bank = torch.tensor(
            latent_rng.normal(
                size=(profile.generator_budget, base_model.latent_dimension)
            ),
            dtype=torch.float32,
        )
        evaluation_latent = torch.tensor(
            np.random.default_rng(
                mixed_seed(int(target["seeds"]["evaluation_latent"]), 0, 21)
            ).normal(size=(profile.evaluation_samples, base_model.latent_dimension)),
            dtype=torch.float32,
        )

        for arm, multiplier in SCREEN:
            model, ledger = train_neural_arm(
                arm,
                base_model,
                latent_bank,
                data.pool,
                training_directions,
                exact_atlas,
                kll_atlas,
                profile,
                learning_rate=ADAM_LR * multiplier,
            )
            with torch.no_grad():
                final = model(evaluation_latent).numpy()
            metrics = evaluate_output(
                final, data, training_directions, heldout_directions
            )
            row = {
                "arm": arm,
                "lr_multiplier": multiplier,
                "learning_rate": ADAM_LR * multiplier,
                "target": target["name"],
                "family": target["family"],
                "dimension": dimension,
                "replication": 0,
                "initialization": "concentrated",
                **metrics,
                **asdict(ledger),
                "model_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
            if ledger.generator_example_evals_training != profile.generator_budget:
                raise AssertionError("screen arm missed the matched generator budget")
            rows.append(row)
            key = safe_key(f"{arm}__{multiplier:g}x__{target['name']}")
            outputs[key] = final.astype(np.float32)
        run.log(
            f"  completed {target['name']} in "
            f"{time.perf_counter() - target_started:.2f}s"
        )

    summary = summarize(rows)
    for setting, values in summary["by_setting"].items():
        run.log(
            f"  {setting:28s} ED2={values['median_ed2']:.6f} "
            f"heldoutSW1={values['median_heldout_sw1']:.6f} "
            f"div={values['divergences']}"
        )
    run.finish(rows, outputs, references, summary)
    print(f"ARTIFACT={run.directory}")


if __name__ == "__main__":
    main()
