"""Post-hoc development runner for conditioned transport-then-amortize.

This runner deliberately reuses the consumed neural pooled-rank registry.  Its
outputs are development evidence, not confirmation.  Run from the repository
root with the environment documented in ``numerics/README.md``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from neural_pooled_rank import (
    QuantileAtlas,
    apache_kll_target_atlas,
    balanced_orthogonal_block_schedule,
    evenly_spaced_step_indices,
    exact_target_atlas,
    extend_to_conditioned_quadratic_frame,
    projection_tree_representatives,
    quadratic_frame_diagnostics,
    transport_then_amortize_step,
    weighted_normalized_paper_field,
    weighted_paper_field_audit,
)
from run_neural_pooled_rank_development import (
    Generator,
    PAPER_GAIN,
    PAPER_TAU,
    REGISTRY,
    atlas_bytes,
    heldout_sw1,
    evaluate_output,
    mixed_seed,
    normalized_target_data,
    safe_key,
    torch_paper_field,
)
from lowdim_drift import energy_distance2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN_ROOT = HERE / "conditioned_transport_runs"
PARTICLE_POPULATION = 512
STUDENT_MICROBATCH = 64
PARTICLE_STEP = 0.5
LOCAL_WEIGHT = 0.25
LOCAL_SCALE_CAP = 256.0
LEARNING_RATE = 0.016
MAXIMUM_QUADRATIC_CONDITION = 25.0
LEGACY_ARMS = (
    "cta-exact-global",
    "cta-kll-global",
    "cta-exact-hybrid",
    "cta-kll-hybrid",
    "cta-kll-guarded",
)
PAPER_ARM = "paper-neural-optimized"
ARMS = (*LEGACY_ARMS, PAPER_ARM)
PAPER_BATCH = 64


@dataclass(frozen=True)
class Profile:
    name: str
    target_pool: int
    generator_budget: int
    evaluation_samples: int
    atlas_knots: int
    smoke_only: bool

    @property
    def macro_steps(self) -> int:
        return self.generator_budget // (2 * PARTICLE_POPULATION)

    def validate(self) -> None:
        if self.generator_budget % (2 * PARTICLE_POPULATION):
            raise ValueError("generator budget must divide into transport macros")
        if self.target_pool < self.macro_steps * PARTICLE_POPULATION:
            raise ValueError("target pool cannot supply the local-field batches")
        if self.target_pool < self.generator_budget:
            raise ValueError("target pool cannot supply the paper comparator batches")
        if self.generator_budget % PAPER_BATCH:
            raise ValueError("generator budget must divide into paper batches")


PROFILES = {
    "smoke": Profile("smoke", 2048, 2048, 512, 64, True),
    "consumed": Profile("consumed", 20480, 20480, 2048, 128, False),
}


def load_registry_path(path: Path) -> tuple[dict[str, Any], str]:
    registry_path = path.resolve()
    sidecar = registry_path.with_suffix(registry_path.suffix + ".sha256")
    if not registry_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("registry and its .sha256 sidecar are required")
    payload = registry_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if actual != expected:
        raise RuntimeError("registry digest does not match its sidecar")
    registry = json.loads(payload)
    targets = registry.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("registry must contain a nonempty target list")
    names = [target.get("name") for target in targets]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("registry target names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("registry target names must be unique")
    return registry, actual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Artifact:
    def __init__(
        self,
        profile: Profile,
        registry_path: Path,
        registry_hash: str,
        expected_rows: int,
        *,
        shard_index: int,
        shard_count: int,
    ) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = (
            "" if shard_count == 1 else f"-shard{shard_index:02d}of{shard_count:02d}"
        )
        self.directory = RUN_ROOT / f"{stamp}-{profile.name}{suffix}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.expected_rows = expected_rows
        self.lines: list[str] = []
        self.kll_index: list[dict[str, Any]] = []
        self.kll_stream = (self.directory / "kll_states.bin").open("wb")
        sources = (
            HERE / "run_conditioned_transport_development.py",
            HERE / "neural_pooled_rank.py",
            HERE / "neural_pooled_rank_tests.py",
            HERE / "run_neural_pooled_rank_development.py",
            HERE / "lowdim_drift.py",
            HERE / "persistent_quantile_transport.py",
            HERE / "projected_quantile_accumulators.py",
            HERE / "standard_projected_kll.py",
            HERE / "ConditionedTransportAmortizationResearch.md",
            HERE / "ProjectionKernelCostOptimizationPlan.md",
        )
        for source in sources:
            destination = self.directory / "source_snapshots" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        registry_path = registry_path.resolve()
        registry_sidecar = registry_path.with_suffix(registry_path.suffix + ".sha256")
        shutil.copy2(registry_path, self.directory / registry_path.name)
        shutil.copy2(registry_sidecar, self.directory / registry_sidecar.name)
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        self.manifest = {
            "schema": "conditioned-transport-development-v2",
            "profile": asdict(profile),
            "registry_sha256": registry_hash,
            "registry_filename": registry_path.name,
            "expected_rows": expected_rows,
            "registered_arms": list(ARMS),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "command": sys.argv,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "git_commit": git_commit or None,
            "git_dirty": bool(git_status),
            "timing_scope": {
                "online_training": "train loop only; excludes atlas and evaluation",
                "setup_plus_training": (
                    "conditioned-direction construction plus selected atlas build "
                    "plus online training; excludes target generation and evaluation"
                ),
            },
            "cost_constants": {
                "particle_population": PARTICLE_POPULATION,
                "student_microbatch": STUDENT_MICROBATCH,
                "paper_batch": PAPER_BATCH,
            },
            "source_sha256": {
                str(source.relative_to(ROOT)): sha256_file(source) for source in sources
            },
        }

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.lines.append(message)

    def record_kll(self, target: str, atlas: QuantileAtlas) -> None:
        if atlas.sketch_payloads is None:
            raise ValueError("KLL atlas has no serialized state")
        for direction, payload in enumerate(atlas.sketch_payloads):
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
    ) -> None:
        self.kll_stream.close()
        if len(rows) != self.expected_rows:
            raise RuntimeError("artifact row count differs from registered design")
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
        summary = summarize(rows)
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
        (self.directory / "summary.md").write_text(
            "```text\n" + "\n".join(self.lines) + "\n```\n", encoding="utf-8"
        )
        hashed_artifacts = (
            "rows.csv",
            "outputs.npz",
            "references.npz",
            "results.json",
            "kll_state_index.json",
            "kll_states.bin",
            "summary.md",
        )
        self.manifest["artifact_sha256"] = {
            name: sha256_file(self.directory / name) for name in hashed_artifacts
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit = audit_artifact(self.directory, deep_metrics=True)
        (self.directory / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.manifest["audit_sha256"] = sha256_file(self.directory / "audit.json")
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def audit_artifact(
    directory: Path, *, deep_metrics: bool = False
) -> dict[str, Any]:
    """Fail closed on coverage, integrity, ledgers, and saved-array structure."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    registered_arms = tuple(manifest.get("registered_arms", LEGACY_ARMS))
    if len(set(registered_arms)) != len(registered_arms):
        raise AssertionError("manifest contains duplicate registered arms")
    for name, digest in manifest.get("artifact_sha256", {}).items():
        if sha256_file(directory / name) != digest:
            raise AssertionError(f"artifact payload hash mismatch: {name}")
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(manifest["expected_rows"]):
        raise AssertionError("audited row count differs from manifest")
    cells = {(row["target"], row["arm"]) for row in rows}
    if len(cells) != len(rows):
        raise AssertionError("duplicate target/arm cell in artifact")
    target_names = {row["target"] for row in rows}
    target_count = len(target_names)
    if len(rows) != target_count * len(registered_arms):
        raise AssertionError("artifact does not contain every registered arm")
    for target in target_names:
        actual_arms = {row["arm"] for row in rows if row["target"] == target}
        if actual_arms != set(registered_arms):
            raise AssertionError(f"target has wrong arm set: {target}")
    budget = int(manifest["profile"]["generator_budget"])
    particle_population = int(
        manifest.get("cost_constants", {}).get(
            "particle_population", PARTICLE_POPULATION
        )
    )
    paper_batch = int(
        manifest.get("cost_constants", {}).get("paper_batch", PAPER_BATCH)
    )
    for row in rows:
        if int(row["generator_example_evals_training"]) != budget:
            raise AssertionError("artifact row missed generator budget")
        for metric in ("ed2", "heldout_sw1", "training_quantile_rmse"):
            if not math.isfinite(float(row[metric])):
                raise AssertionError("artifact contains a non-finite endpoint")
        if manifest["schema"] == "conditioned-transport-development-v2":
            if row["arm"] == PAPER_ARM:
                expected_pairs = (
                    int(row["completed_updates"]) * paper_batch * (2 * paper_batch)
                )
            elif row["arm"] in (
                "cta-exact-hybrid",
                "cta-kll-hybrid",
                "cta-kll-guarded",
            ):
                expected_pairs = (
                    int(row["local_field_calls"])
                    * particle_population
                    * (2 * int(row["local_representative_count"]))
                )
            else:
                expected_pairs = 0
            if int(row["paper_kernel_pairs"]) != expected_pairs:
                raise AssertionError("kernel-pair ledger is inconsistent")
            if int(row["completed_updates"]) <= 0:
                raise AssertionError("artifact row has no optimizer updates")

    with np.load(directory / "outputs.npz") as outputs:
        output_keys = set(outputs.files)
        expected_output_keys = {
            safe_key(f"{row['arm']}__{row['target']}") for row in rows
        }
        if output_keys != expected_output_keys:
            raise AssertionError("saved output keys do not match target/arm rows")
        if any(not np.all(np.isfinite(outputs[key])) for key in output_keys):
            raise AssertionError("saved output contains non-finite values")
        with np.load(directory / "references.npz") as references:
            expected_reference_keys = {safe_key(target) for target in target_names}
            if set(references.files) != expected_reference_keys:
                raise AssertionError("saved reference keys do not match targets")
            evaluation_samples = int(manifest["profile"]["evaluation_samples"])
            for row in rows:
                output = outputs[safe_key(f"{row['arm']}__{row['target']}")]
                reference = references[safe_key(row["target"])]
                dimension = int(row["dimension"])
                expected_shape = (evaluation_samples, dimension)
                if output.shape != expected_shape or reference.shape != expected_shape:
                    raise AssertionError("saved output/reference has wrong shape")

    index = json.loads((directory / "kll_state_index.json").read_text(encoding="utf-8"))
    state = (directory / "kll_states.bin").read_bytes()
    position = 0
    seen_states: set[tuple[str, int]] = set()
    for entry in index:
        offset = int(entry["offset"])
        length = int(entry["length"])
        if offset != position:
            raise AssertionError("KLL state offsets are not contiguous")
        payload = state[offset : offset + length]
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise AssertionError("KLL state payload hash mismatch")
        key = (str(entry["target"]), int(entry["direction"]))
        if key in seen_states:
            raise AssertionError("duplicate KLL state key")
        seen_states.add(key)
        position += length
    if position != len(state):
        raise AssertionError("KLL state index does not consume the binary file")
    expected_states = sum(
        int(row["conditioned_direction_count"])
        for row in rows
        if row["arm"] == "cta-exact-global"
    )
    if len(index) != expected_states:
        raise AssertionError("KLL state count differs from direction ledger")

    for relative, digest in manifest["source_sha256"].items():
        snapshot = directory / "source_snapshots" / Path(relative).name
        if sha256_file(snapshot) != digest:
            raise AssertionError("source snapshot hash mismatch")
    registry_copy = directory / manifest.get("registry_filename", REGISTRY.name)
    if sha256_file(registry_copy) != manifest["registry_sha256"]:
        raise AssertionError("copied registry hash mismatch")
    registry = json.loads(registry_copy.read_text(encoding="utf-8"))
    registry_targets = {target["name"]: target for target in registry["targets"]}
    if not target_names.issubset(registry_targets):
        raise AssertionError("artifact contains a target absent from its registry")

    maximum_ed2_error = 0.0
    maximum_sw1_error = 0.0
    if deep_metrics:
        with np.load(directory / "outputs.npz") as outputs, np.load(
            directory / "references.npz"
        ) as references:
            for row in rows:
                target = row["target"]
                output = outputs[safe_key(f"{row['arm']}__{target}")]
                reference = references[safe_key(target)]
                heldout = np.asarray(
                    registry_targets[target]["heldout_directions"], dtype=float
                )
                recomputed_ed2 = max(
                    0.0, float(energy_distance2(output, reference))
                )
                recomputed_sw1 = heldout_sw1(output, reference, heldout)
                maximum_ed2_error = max(
                    maximum_ed2_error, abs(recomputed_ed2 - float(row["ed2"]))
                )
                maximum_sw1_error = max(
                    maximum_sw1_error,
                    abs(recomputed_sw1 - float(row["heldout_sw1"])),
                )
        # References are deliberately saved as float32, while endpoint rows
        # were evaluated against the original float64 reference.  These bounds
        # cover only that serialization roundoff.
        if maximum_ed2_error > 3e-6 or maximum_sw1_error > 3e-8:
            raise AssertionError(
                "saved outputs do not reproduce primary endpoint metrics"
            )
    return {
        "status": "pass",
        "schema": manifest["schema"],
        "rows": len(rows),
        "targets": target_count,
        "unique_cells": len(cells),
        "outputs": len(output_keys),
        "references": target_count,
        "kll_states": len(index),
        "kll_bytes": len(state),
        "source_snapshots": len(manifest["source_sha256"]),
        "generator_budget_per_row": budget,
        "artifact_payload_hashes": len(manifest.get("artifact_sha256", {})),
        "deep_metrics": deep_metrics,
        "maximum_recomputed_ed2_error": maximum_ed2_error,
        "maximum_recomputed_heldout_sw1_error": maximum_sw1_error,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_arm": {}}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        result["by_arm"][arm] = {
            "count": len(selected),
            "median_ed2": float(np.median([row["ed2"] for row in selected])),
            "median_heldout_sw1": float(
                np.median([row["heldout_sw1"] for row in selected])
            ),
            "mean_selected_local_weight": float(
                np.mean([row["mean_selected_local_weight"] for row in selected])
            ),
        }
    return result


def conditioned_directions(target: dict) -> tuple[np.ndarray, dict[str, float | int]]:
    registered = np.asarray(target["training_directions"], dtype=float)
    seed = mixed_seed(int(target["seeds"]["reserved"]), 0, 99)
    directions, diagnostics = extend_to_conditioned_quadratic_frame(
        registered,
        np.random.default_rng(seed),
        maximum_condition_number=MAXIMUM_QUADRATIC_CONDITION,
    )
    return directions, {
        "registered_direction_count": len(registered),
        "conditioned_direction_count": len(directions),
        "quadratic_parameter_dimension": diagnostics.parameter_dimension,
        "quadratic_rank": diagnostics.rank,
        "quadratic_smallest_singular_value": diagnostics.smallest_singular_value,
        "quadratic_condition_number": diagnostics.condition_number,
    }


def clone_generator(base: Generator, dimension: int) -> Generator:
    model = Generator(dimension, "concentrated", 0).to(dtype=torch.float32)
    model.load_state_dict(base.state_dict())
    return model


def train_optimized_paper_arm(
    base: Generator,
    latent_bank: torch.Tensor,
    target_pool: np.ndarray,
    *,
    direction_count: int,
) -> tuple[Generator, dict[str, float | int]]:
    """Run the paper-neural comparator through the current exact E0 field."""
    if len(latent_bank) * 1 < len(target_pool):
        raise ValueError("paper comparator requires one latent per target observation")
    updates, remainder = divmod(len(target_pool), PAPER_BATCH)
    if remainder or len(latent_bank) < updates * PAPER_BATCH:
        raise ValueError("paper comparator budget must divide into complete batches")
    model = clone_generator(base, target_pool.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    started = time.perf_counter()
    for step in range(updates):
        start = step * PAPER_BATCH
        stop = start + PAPER_BATCH
        positive = torch.tensor(target_pool[start:stop], dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        generated = model(latent_bank[start:stop])
        field = torch_paper_field(generated.detach(), positive)
        loss = -PAPER_GAIN * torch.mean(torch.sum(generated * field, dim=1))
        loss.backward()
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise FloatingPointError("optimized paper comparator diverged")
    wall = time.perf_counter() - started
    pairs = updates * PAPER_BATCH * (2 * PAPER_BATCH)
    ledger: dict[str, float | int] = {
        "completed_updates": updates,
        "unique_latent_samples": updates * PAPER_BATCH,
        "generator_example_evals_training": updates * PAPER_BATCH,
        "generator_forward_calls_training": updates,
        "unique_target_observations": len(target_pool),
        "target_example_accesses": updates * PAPER_BATCH,
        "projection_scalar_products": 0,
        "atlas_projection_scalar_products": 0,
        "training_projection_scalar_products": 0,
        "sort_work": 0.0,
        "atlas_sort_work": 0.0,
        "training_sort_work": 0.0,
        "full_direction_count": direction_count,
        "active_direction_count_min": 0,
        "active_direction_count_max": 0,
        "mean_active_direction_count": 0.0,
        "direction_exposure_min": 0,
        "direction_exposure_max": 0,
        "direction_schedule_seed": 0,
        "orthogonal_direction_block_count": 0,
        "active_direction_block_count": 0,
        "local_field_calls": 0,
        "registered_local_field_call_count": 0,
        "paper_kernel_pairs": pairs,
        "dense_equivalent_paper_kernel_pairs": pairs,
        "local_representative_count": 0,
        "representative_tree_levels": 0,
        "representative_partition_projection_scalar_products": 0,
        "representative_partition_sort_work": 0.0,
        "representative_center_coordinate_accumulations": 0,
        "representative_audit_calls": 0,
        "representative_audit_dense_kernel_pairs": 0,
        "mean_representative_field_relative_l2_error": 0.0,
        "minimum_representative_field_cosine": 1.0,
        "mean_representative_row_mass_relative_l2_error": 0.0,
        "mean_representative_column_mass_relative_l2_error": 0.0,
        "maximum_representative_row_mass_relative_error": 0.0,
        "maximum_representative_column_mass_relative_error": 0.0,
        "mean_positive_representative_rms_radius": 0.0,
        "maximum_positive_representative_radius": 0.0,
        "mean_negative_representative_rms_radius": 0.0,
        "maximum_negative_representative_radius": 0.0,
        "representative_leaf_population_min": 0,
        "representative_leaf_population_max": 0,
        "atlas_bytes": 0,
        "kll_serialized_bytes": 0,
        "wall_seconds": wall,
        "mean_selected_local_weight": 0.0,
        "minimum_selected_local_weight": 0.0,
        "mean_selected_local_weight_when_active": 0.0,
        "mean_local_scale": 0.0,
        "mean_teacher_displacement_rms": 0.0,
    }
    return model, ledger


def registered_weighted_paper_field(
    query: torch.Tensor,
    positive_representatives,
    negative_representatives,
) -> torch.Tensor:
    """Single registered-bandwidth entry point for every K2 training call."""
    return weighted_normalized_paper_field(
        query,
        positive_representatives,
        negative_representatives,
        tau=PAPER_TAU,
    )


def train_arm(
    arm: str,
    base: Generator,
    latent_bank: torch.Tensor,
    target_batches: np.ndarray,
    exact_atlas: QuantileAtlas,
    kll_atlas: QuantileAtlas,
    permutations: tuple[torch.Tensor, ...],
    *,
    active_direction_count: int | None = None,
    direction_schedule_seed: int = 0,
    local_field_call_count: int | None = None,
    local_representative_count: int | None = None,
    representative_audit_calls: int = 0,
) -> tuple[Generator, dict[str, float | int]]:
    model = clone_generator(base, exact_atlas.dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    atlas = exact_atlas if "exact" in arm else kll_atlas
    use_local = "hybrid" in arm or "guarded" in arm
    protect = "guarded" in arm
    macro_steps = len(target_batches)
    configured_local_calls = (
        macro_steps if local_field_call_count is None else int(local_field_call_count)
    )
    if not 0 <= configured_local_calls <= macro_steps:
        raise ValueError("local-field calls must lie between zero and macro steps")
    representative_count = (
        PARTICLE_POPULATION
        if local_representative_count is None
        else int(local_representative_count)
    )
    if (
        representative_count < 1
        or representative_count > PARTICLE_POPULATION
        or representative_count & (representative_count - 1)
    ):
        raise ValueError(
            "local representative count must be a power of two within the population"
        )
    if representative_audit_calls < 0:
        raise ValueError("representative audit calls must be nonnegative")
    use_representatives = local_representative_count is not None
    requested_local_calls = configured_local_calls if use_local else 0
    local_steps = (
        set(map(int, evenly_spaced_step_indices(macro_steps, requested_local_calls)))
        if use_local
        else set()
    )
    active_count = (
        atlas.direction_count
        if active_direction_count is None
        else int(active_direction_count)
    )
    direction_schedule = balanced_orthogonal_block_schedule(
        atlas.directions,
        active_count,
        macro_steps,
        np.random.default_rng(direction_schedule_seed),
    )
    step_atlases = (
        tuple(atlas for _ in range(macro_steps))
        if active_count == atlas.direction_count
        else tuple(atlas.select_directions(indices) for indices in direction_schedule)
    )
    selected_weights: list[float] = []
    local_scales: list[float] = []
    teacher_rms: list[float] = []
    positive_representative_rms: list[float] = []
    positive_representative_max: list[float] = []
    negative_representative_rms: list[float] = []
    negative_representative_max: list[float] = []
    representative_field_relative_l2: list[float] = []
    representative_field_cosine: list[float] = []
    representative_row_mass_relative_l2: list[float] = []
    representative_column_mass_relative_l2: list[float] = []
    representative_row_mass_max_relative: list[float] = []
    representative_column_mass_max_relative: list[float] = []
    remaining_representative_audits = int(representative_audit_calls)
    started = time.perf_counter()
    for macro in range(macro_steps):
        start = macro * PARTICLE_POPULATION
        stop = (macro + 1) * PARTICLE_POPULATION
        builder = None
        local_active = macro in local_steps
        if local_active:
            positive = torch.tensor(target_batches[macro], dtype=torch.float32)
            if use_representatives:
                split_directions = step_atlases[macro].torch_directions(positive)
                positive_representatives = projection_tree_representatives(
                    positive, split_directions, representative_count
                )
                positive_representative_rms.append(positive_representatives.rms_radius)
                positive_representative_max.append(positive_representatives.max_radius)

                def builder(
                    run_features: torch.Tensor,
                    positive: torch.Tensor = positive,
                    positive_representatives=positive_representatives,
                    split_directions: torch.Tensor = split_directions,
                ) -> torch.Tensor:
                    nonlocal remaining_representative_audits
                    negative_representatives = projection_tree_representatives(
                        run_features, split_directions, representative_count
                    )
                    negative_representative_rms.append(
                        negative_representatives.rms_radius
                    )
                    negative_representative_max.append(
                        negative_representatives.max_radius
                    )
                    approximate = registered_weighted_paper_field(
                        run_features,
                        positive_representatives,
                        negative_representatives,
                    )
                    if remaining_representative_audits > 0:
                        audit = weighted_paper_field_audit(
                            run_features,
                            positive,
                            positive_representatives,
                            negative_representatives,
                            tau=PAPER_TAU,
                            approximate_field=approximate,
                        )
                        representative_field_relative_l2.append(
                            audit["field_relative_l2_error"]
                        )
                        representative_field_cosine.append(
                            audit["field_cosine"]
                        )
                        representative_row_mass_relative_l2.append(
                            audit["row_mass_relative_l2_error"]
                        )
                        representative_column_mass_relative_l2.append(
                            audit["column_mass_relative_l2_error"]
                        )
                        representative_row_mass_max_relative.append(
                            audit["row_mass_max_relative_error"]
                        )
                        representative_column_mass_max_relative.append(
                            audit["column_mass_max_relative_error"]
                        )
                        remaining_representative_audits -= 1
                    return approximate

            else:

                def builder(
                    run_features: torch.Tensor, positive: torch.Tensor = positive
                ) -> torch.Tensor:
                    return torch_paper_field(run_features, positive)

        result = transport_then_amortize_step(
            model,
            optimizer,
            latent_bank[start:stop],
            step_atlases[macro],
            microbatch=STUDENT_MICROBATCH,
            particle_step=PARTICLE_STEP,
            local_field_builder=builder,
            local_weight=LOCAL_WEIGHT if local_active else 0.0,
            local_scale_cap=LOCAL_SCALE_CAP,
            permutation=permutations[macro],
            protect_tails=protect,
        )
        selected_weights.append(result.selected_local_weight)
        local_scales.append(result.local_scale)
        teacher_rms.append(result.teacher_displacement_rms)
    direction_count = atlas.direction_count
    scheduled_direction_total = sum(len(indices) for indices in direction_schedule)
    direction_exposure = np.zeros(direction_count, dtype=np.int64)
    for indices in direction_schedule:
        direction_exposure[indices] += 1
    chunks = math.ceil(PARTICLE_POPULATION / STUDENT_MICROBATCH)
    atlas_projection_products = atlas.target_count * direction_count
    # ``psqt_feature_correction`` reuses the detached Run projection used for
    # rank assignment; the teacher therefore needs one, not two, projection
    # passes per macro-step.
    training_projection_products = PARTICLE_POPULATION * scheduled_direction_total
    training_sort_work = (
        scheduled_direction_total
        * PARTICLE_POPULATION
        * math.log2(PARTICLE_POPULATION)
    )
    atlas_sort_work = 0.0
    if atlas.source.startswith("exact"):
        atlas_sort_work = (
            atlas.target_count * direction_count * math.log2(atlas.target_count)
        )
    if protect:
        # One guard assignment plus current and four candidate loss projections.
        training_projection_products += (
            6 * PARTICLE_POPULATION * scheduled_direction_total
        )
        training_sort_work += (
            scheduled_direction_total
            * PARTICLE_POPULATION
            * math.log2(PARTICLE_POPULATION)
        )
    representative_levels = (
        representative_count.bit_length() - 1 if use_representatives else 0
    )
    representative_partition_projection_products = (
        2 * len(local_steps) * PARTICLE_POPULATION * representative_levels
        if use_local and use_representatives
        else 0
    )
    representative_partition_sort_work = (
        2
        * len(local_steps)
        * PARTICLE_POPULATION
        * sum(
            math.log2(PARTICLE_POPULATION) - level
            for level in range(representative_levels)
        )
        if use_local and use_representatives
        else 0.0
    )
    training_projection_products += representative_partition_projection_products
    training_sort_work += representative_partition_sort_work
    projection_products = atlas_projection_products + training_projection_products
    sort_work = atlas_sort_work + training_sort_work
    paper_pairs = (
        len(local_steps) * PARTICLE_POPULATION * (2 * representative_count)
        if use_local
        else 0
    )
    representative_audit_pairs = (
        len(representative_field_relative_l2)
        * PARTICLE_POPULATION
        * (2 * PARTICLE_POPULATION)
    )
    ledger: dict[str, float | int] = {
        "completed_updates": macro_steps * chunks,
        "unique_latent_samples": macro_steps * PARTICLE_POPULATION,
        "generator_example_evals_training": 2 * macro_steps * PARTICLE_POPULATION,
        "generator_forward_calls_training": 2 * macro_steps * chunks,
        "unique_target_observations": atlas.target_count,
        "target_example_accesses": atlas.target_count
        + (len(local_steps) * PARTICLE_POPULATION if use_local else 0),
        "projection_scalar_products": projection_products,
        "atlas_projection_scalar_products": atlas_projection_products,
        "training_projection_scalar_products": training_projection_products,
        "sort_work": sort_work,
        "atlas_sort_work": atlas_sort_work,
        "training_sort_work": training_sort_work,
        "full_direction_count": direction_count,
        "active_direction_count_min": min(map(len, direction_schedule)),
        "active_direction_count_max": max(map(len, direction_schedule)),
        "mean_active_direction_count": scheduled_direction_total / macro_steps,
        "direction_exposure_min": int(direction_exposure.min()),
        "direction_exposure_max": int(direction_exposure.max()),
        "direction_schedule_seed": int(direction_schedule_seed),
        "orthogonal_direction_block_count": direction_count // atlas.dimension,
        "active_direction_block_count": active_count // atlas.dimension,
        "local_field_calls": len(local_steps),
        "registered_local_field_call_count": requested_local_calls,
        "paper_kernel_pairs": paper_pairs,
        "dense_equivalent_paper_kernel_pairs": (
            len(local_steps) * PARTICLE_POPULATION * (2 * PARTICLE_POPULATION)
            if use_local
            else 0
        ),
        "local_representative_count": (
            representative_count if use_representatives else PARTICLE_POPULATION
        ),
        "representative_tree_levels": representative_levels,
        "representative_partition_projection_scalar_products": (
            representative_partition_projection_products
        ),
        "representative_partition_sort_work": representative_partition_sort_work,
        "representative_center_coordinate_accumulations": (
            2 * len(local_steps) * PARTICLE_POPULATION * atlas.dimension
            if use_local and use_representatives
            else 0
        ),
        "representative_audit_calls": len(representative_field_relative_l2),
        "representative_audit_dense_kernel_pairs": representative_audit_pairs,
        "mean_representative_field_relative_l2_error": (
            float(np.mean(representative_field_relative_l2))
            if representative_field_relative_l2
            else 0.0
        ),
        "minimum_representative_field_cosine": (
            float(np.min(representative_field_cosine))
            if representative_field_cosine
            else 1.0
        ),
        "mean_representative_row_mass_relative_l2_error": (
            float(np.mean(representative_row_mass_relative_l2))
            if representative_row_mass_relative_l2
            else 0.0
        ),
        "mean_representative_column_mass_relative_l2_error": (
            float(np.mean(representative_column_mass_relative_l2))
            if representative_column_mass_relative_l2
            else 0.0
        ),
        "maximum_representative_row_mass_relative_error": (
            float(np.max(representative_row_mass_max_relative))
            if representative_row_mass_max_relative
            else 0.0
        ),
        "maximum_representative_column_mass_relative_error": (
            float(np.max(representative_column_mass_max_relative))
            if representative_column_mass_max_relative
            else 0.0
        ),
        "mean_positive_representative_rms_radius": (
            float(np.mean(positive_representative_rms))
            if positive_representative_rms
            else 0.0
        ),
        "maximum_positive_representative_radius": (
            float(np.max(positive_representative_max))
            if positive_representative_max
            else 0.0
        ),
        "mean_negative_representative_rms_radius": (
            float(np.mean(negative_representative_rms))
            if negative_representative_rms
            else 0.0
        ),
        "maximum_negative_representative_radius": (
            float(np.max(negative_representative_max))
            if negative_representative_max
            else 0.0
        ),
        "representative_leaf_population_min": (
            PARTICLE_POPULATION // representative_count if use_local else 0
        ),
        "representative_leaf_population_max": (
            PARTICLE_POPULATION // representative_count if use_local else 0
        ),
        "atlas_bytes": atlas_bytes(atlas),
        "kll_serialized_bytes": sum(
            len(payload) for payload in (atlas.sketch_payloads or ())
        ),
        "wall_seconds": time.perf_counter() - started,
        "mean_selected_local_weight": float(np.mean(selected_weights)),
        "minimum_selected_local_weight": float(np.min(selected_weights)),
        "mean_selected_local_weight_when_active": (
            float(np.mean([selected_weights[index] for index in sorted(local_steps)]))
            if local_steps
            else 0.0
        ),
        "mean_local_scale": float(np.mean(local_scales)),
        "mean_teacher_displacement_rms": float(np.mean(teacher_rms)),
    }
    return model, ledger


def runner_optimization_invariants() -> dict[str, float]:
    """Exercise the actual dense/K2 runner path at the registered bandwidth."""
    rng = np.random.default_rng(2026072209)
    dimension = 2
    directions = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0**-0.5, 2.0**-0.5],
            [-2.0**-0.5, 2.0**-0.5],
        ]
    )
    target_batches = rng.normal(
        size=(1, PARTICLE_POPULATION, dimension)
    ).astype(np.float32)
    atlas = exact_target_atlas(target_batches[0], directions, knot_count=64)
    positive = torch.tensor(target_batches[0], dtype=torch.float32)
    query = torch.tensor(
        rng.normal(size=(PARTICLE_POPULATION, dimension)), dtype=torch.float32
    )
    split_directions = atlas.torch_directions(query)
    positive_representatives = projection_tree_representatives(
        positive, split_directions, PARTICLE_POPULATION
    )
    negative_representatives = projection_tree_representatives(
        query, split_directions, PARTICLE_POPULATION
    )
    dense_field = torch_paper_field(query, positive)
    weighted_field = registered_weighted_paper_field(
        query, positive_representatives, negative_representatives
    )
    field_error = float(torch.max(torch.abs(dense_field - weighted_field)))
    if field_error > 3e-6:
        raise AssertionError(
            f"runner M=B field changed at PAPER_TAU: {field_error}"
        )

    base = Generator(dimension, "concentrated", 1701).to(dtype=torch.float32)
    latent_bank = torch.tensor(
        rng.normal(size=(PARTICLE_POPULATION, base.latent_dimension)),
        dtype=torch.float32,
    )
    permutations = (
        torch.tensor(rng.permutation(PARTICLE_POPULATION), dtype=torch.long),
    )
    dense_model, _ = train_arm(
        "cta-exact-hybrid",
        base,
        latent_bank,
        target_batches,
        atlas,
        atlas,
        permutations,
    )
    weighted_model, _ = train_arm(
        "cta-exact-hybrid",
        base,
        latent_bank,
        target_batches,
        atlas,
        atlas,
        permutations,
        local_representative_count=PARTICLE_POPULATION,
    )
    parameter_error = max(
        float(torch.max(torch.abs(left - right)))
        for left, right in zip(
            dense_model.state_dict().values(),
            weighted_model.state_dict().values(),
            strict=True,
        )
    )
    if parameter_error > 1e-5:
        raise AssertionError(
            f"runner M=B training path changed parameters: {parameter_error}"
        )
    return {
        "runner_full_representative_field_max_abs_error": field_error,
        "runner_full_representative_parameter_max_abs_error": parameter_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY,
        help="frozen registry JSON; its adjacent .sha256 file is required",
    )
    parser.add_argument(
        "--active-directions",
        type=int,
        default=0,
        help="complete orthogonal-block directions per macro; zero uses all",
    )
    parser.add_argument(
        "--local-field-calls",
        type=int,
        default=-1,
        help="midpoint-spaced local calls; -1 uses every macro-step",
    )
    parser.add_argument(
        "--local-representatives",
        type=int,
        default=0,
        help="power-of-two representatives per local support; zero uses dense",
    )
    parser.add_argument(
        "--representative-audit-calls",
        type=int,
        default=0,
        help="dense field comparisons per local arm (diagnostic overhead)",
    )
    parser.add_argument(
        "--arm-order-seed",
        type=int,
        default=2026072201,
        help="deterministic per-target arm-order randomization seed",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard selection")
    if args.active_directions < 0:
        parser.error("active directions must be nonnegative")
    profile = PROFILES[args.profile]
    profile.validate()
    if args.local_field_calls < -1 or args.local_field_calls > profile.macro_steps:
        parser.error("local-field calls must be -1 or between zero and macro-steps")
    if (
        args.local_representatives < 0
        or args.local_representatives > PARTICLE_POPULATION
        or (
            args.local_representatives > 0
            and args.local_representatives & (args.local_representatives - 1)
        )
    ):
        parser.error(
            "local representatives must be zero or a power of two up to 512"
        )
    if args.representative_audit_calls < 0:
        parser.error("representative audit calls must be nonnegative")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    registry, registry_hash = load_registry_path(args.registry)
    from neural_pooled_rank_tests import (
        direction_subset_and_schedule_test,
        exact_cost_refactor_test,
        quadratic_frame_diagnostic_test,
        transport_gradient_identity_test,
        transport_step_and_guard_test,
        weighted_representative_field_test,
    )

    for invariant in (
        direction_subset_and_schedule_test,
        exact_cost_refactor_test,
        weighted_representative_field_test,
        quadratic_frame_diagnostic_test,
        transport_gradient_identity_test,
        transport_step_and_guard_test,
    ):
        invariant()
    runner_optimization_invariants()
    eligible = [
        target
        for target in registry["targets"]
        if not profile.smoke_only or target["smoke"]
    ]
    targets = [
        target
        for index, target in enumerate(eligible)
        if index % args.shard_count == args.shard_index
    ]
    if not targets:
        parser.error("selected shard contains no targets")
    artifact = Artifact(
        profile,
        args.registry,
        registry_hash,
        len(targets) * len(ARMS),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    artifact.log(
        f"Conditioned transport audit runner: {len(targets)} targets x {len(ARMS)} arms"
    )
    artifact.log(
        "  active directions per macro: "
        + (
            "full registered atlas"
            if args.active_directions == 0
            else str(args.active_directions)
        )
    )
    artifact.log(
        "  local field calls: "
        + (
            "every macro-step"
            if args.local_field_calls == -1
            else str(args.local_field_calls)
        )
    )
    artifact.log(
        "  local supports: "
        + (
            "dense 512-point batches"
            if args.local_representatives == 0
            else f"{args.local_representatives} weighted projection-tree representatives"
        )
    )
    if args.representative_audit_calls:
        artifact.log(
            "  representative field audit calls per local arm: "
            f"{args.representative_audit_calls} (excluded from model kernel ledger)"
        )
    artifact.log(
        "  [PASS] E0, P1, K2, quadratic-frame, and transport regression suite"
    )
    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for target_index, target in enumerate(targets):
        started = time.perf_counter()
        dimension = int(target["dimension"])
        direction_started = time.perf_counter()
        directions, direction_record = conditioned_directions(target)
        direction_build_seconds = time.perf_counter() - direction_started
        diagnostics = quadratic_frame_diagnostics(directions)
        if not diagnostics.is_full_rank or diagnostics.condition_number > 25.0:
            raise AssertionError("conditioned direction certificate was lost")
        heldout = np.asarray(target["heldout_directions"], dtype=float)
        data = normalized_target_data(target, 0, profile)
        exact_started = time.perf_counter()
        exact_atlas = exact_target_atlas(data.pool, directions, profile.atlas_knots)
        exact_build_seconds = time.perf_counter() - exact_started
        kll_started = time.perf_counter()
        kll_atlas = apache_kll_target_atlas(
            data.pool, directions, profile.atlas_knots, k=128
        )
        kll_build_seconds = time.perf_counter() - kll_started
        artifact.record_kll(target["name"], kll_atlas)
        references[safe_key(target["name"])] = data.reference.astype(np.float32)

        model_seed = mixed_seed(int(target["seeds"]["model"]), 0, 11)
        base = Generator(dimension, "concentrated", model_seed).to(dtype=torch.float32)
        latent_seed = mixed_seed(int(target["seeds"]["training_latent"]), 0, 20)
        all_latent_bank = torch.tensor(
            np.random.default_rng(latent_seed).normal(
                size=(
                    profile.generator_budget,
                    base.latent_dimension,
                )
            ),
            dtype=torch.float32,
        )
        latent_bank = all_latent_bank[
            : profile.macro_steps * PARTICLE_POPULATION
        ]
        order_rng = np.random.default_rng(
            mixed_seed(int(target["seeds"]["minibatch_order"]), 0, 31)
        )
        target_order = order_rng.permutation(len(data.pool))[
            : profile.macro_steps * PARTICLE_POPULATION
        ]
        target_batches = data.pool[target_order].reshape(
            profile.macro_steps, PARTICLE_POPULATION, dimension
        )
        permutations = tuple(
            torch.tensor(order_rng.permutation(PARTICLE_POPULATION), dtype=torch.long)
            for _ in range(profile.macro_steps)
        )
        evaluation_seed = mixed_seed(int(target["seeds"]["evaluation_latent"]), 0, 21)
        evaluation_latent = torch.tensor(
            np.random.default_rng(evaluation_seed).normal(
                size=(profile.evaluation_samples, base.latent_dimension)
            ),
            dtype=torch.float32,
        )
        arm_rng = np.random.default_rng(
            mixed_seed(args.arm_order_seed, target_index, 401)
        )
        execution_order = [ARMS[index] for index in arm_rng.permutation(len(ARMS))]
        for execution_position, arm in enumerate(execution_order):
            if arm == PAPER_ARM:
                model, ledger = train_optimized_paper_arm(
                    base,
                    all_latent_bank,
                    data.pool,
                    direction_count=len(directions),
                )
                atlas_build_seconds = 0.0
                arm_direction_build_seconds = 0.0
            else:
                model, ledger = train_arm(
                    arm,
                    base,
                    latent_bank,
                    target_batches,
                    exact_atlas,
                    kll_atlas,
                    permutations,
                    active_direction_count=(
                        None
                        if args.active_directions == 0
                        else args.active_directions
                    ),
                    direction_schedule_seed=mixed_seed(
                        int(target["seeds"]["reserved"]), 0, 117
                    ),
                    local_field_call_count=(
                        None
                        if args.local_field_calls == -1
                        else args.local_field_calls
                    ),
                    local_representative_count=(
                        None
                        if args.local_representatives == 0
                        else args.local_representatives
                    ),
                    representative_audit_calls=args.representative_audit_calls,
                )
                atlas_build_seconds = (
                    exact_build_seconds if "exact" in arm else kll_build_seconds
                )
                arm_direction_build_seconds = direction_build_seconds
            with torch.no_grad():
                final = model(evaluation_latent).numpy()
            metrics = evaluate_output(final, data, directions, heldout)
            row = {
                "arm": arm,
                "target": target["name"],
                "family": target["family"],
                "dimension": dimension,
                "replication": 0,
                "initialization": "concentrated",
                "learning_rate": LEARNING_RATE,
                "particle_step": PARTICLE_STEP,
                "registered_local_weight": LOCAL_WEIGHT,
                "registered_active_direction_count": args.active_directions,
                "registered_local_field_calls": args.local_field_calls,
                "registered_local_representative_count": (
                    args.local_representatives
                ),
                "registered_representative_audit_calls": (
                    args.representative_audit_calls
                ),
                "execution_position": execution_position,
                "arm_order_seed": args.arm_order_seed,
                "paper_tau": PAPER_TAU,
                "paper_gain": PAPER_GAIN,
                "online_training_wall_seconds": ledger["wall_seconds"],
                "conditioned_direction_build_seconds": (
                    arm_direction_build_seconds
                ),
                "atlas_build_wall_seconds": atlas_build_seconds,
                "setup_plus_training_wall_seconds": (
                    ledger["wall_seconds"]
                    + arm_direction_build_seconds
                    + atlas_build_seconds
                ),
                **direction_record,
                **metrics,
                **ledger,
            }
            if row["generator_example_evals_training"] != profile.generator_budget:
                raise AssertionError("arm missed the generator-example budget")
            rows.append(row)
            outputs[safe_key(f"{arm}__{target['name']}")] = final.astype(np.float32)
        artifact.log(
            f"  {target['name']}: L={len(directions)}, "
            f"kappa={diagnostics.condition_number:.2f}, "
            f"{time.perf_counter() - started:.2f}s"
        )
    summary = summarize(rows)
    for arm in ARMS:
        values = summary["by_arm"][arm]
        artifact.log(
            f"  {arm:24s} ED2={values['median_ed2']:.6f} "
            f"SW1={values['median_heldout_sw1']:.6f} "
            f"local={values['mean_selected_local_weight']:.3f}"
        )
    artifact.finish(rows, outputs, references)
    print(f"artifact: {artifact.directory}", flush=True)


if __name__ == "__main__":
    main()
