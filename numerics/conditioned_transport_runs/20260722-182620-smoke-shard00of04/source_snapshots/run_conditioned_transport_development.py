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
    quadratic_frame_diagnostics,
    transport_then_amortize_step,
)
from run_neural_pooled_rank_development import (
    Generator,
    REGISTRY,
    REGISTRY_HASH,
    atlas_bytes,
    evaluate_output,
    load_registry,
    mixed_seed,
    normalized_target_data,
    safe_key,
    torch_paper_field,
)


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
ARMS = (
    "cta-exact-global",
    "cta-kll-global",
    "cta-exact-hybrid",
    "cta-kll-hybrid",
    "cta-kll-guarded",
)


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


PROFILES = {
    "smoke": Profile("smoke", 2048, 2048, 512, 64, True),
    "consumed": Profile("consumed", 20480, 20480, 2048, 128, False),
}


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
            HERE / "ConditionedTransportAmortizationResearch.md",
            HERE / "ProjectionKernelCostOptimizationPlan.md",
        )
        for source in sources:
            destination = self.directory / "source_snapshots" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(REGISTRY, self.directory / REGISTRY.name)
        shutil.copy2(REGISTRY_HASH, self.directory / REGISTRY_HASH.name)
        self.manifest = {
            "schema": "conditioned-transport-development-v1",
            "profile": asdict(profile),
            "registry_sha256": registry_hash,
            "expected_rows": expected_rows,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "command": sys.argv,
            "numpy": np.__version__,
            "torch": torch.__version__,
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
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.directory / "summary.md").write_text(
            "```text\n" + "\n".join(self.lines) + "\n```\n", encoding="utf-8"
        )
        audit = audit_artifact(self.directory)
        (self.directory / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.manifest["audit_sha256"] = sha256_file(self.directory / "audit.json")
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def audit_artifact(directory: Path) -> dict[str, Any]:
    """Fail closed on coverage, hashes, budgets, and saved-array structure."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(manifest["expected_rows"]):
        raise AssertionError("audited row count differs from manifest")
    cells = {(row["target"], row["arm"]) for row in rows}
    if len(cells) != len(rows):
        raise AssertionError("duplicate target/arm cell in artifact")
    target_count = len({row["target"] for row in rows})
    if len(rows) != target_count * len(ARMS):
        raise AssertionError("artifact does not contain every registered arm")
    budget = int(manifest["profile"]["generator_budget"])
    for row in rows:
        if int(row["generator_example_evals_training"]) != budget:
            raise AssertionError("artifact row missed generator budget")
        for metric in ("ed2", "heldout_sw1", "training_quantile_rmse"):
            if not math.isfinite(float(row[metric])):
                raise AssertionError("artifact contains a non-finite endpoint")

    with np.load(directory / "outputs.npz") as outputs:
        output_keys = set(outputs.files)
        if len(output_keys) != len(rows):
            raise AssertionError("saved outputs do not cover every row")
        if any(not np.all(np.isfinite(outputs[key])) for key in output_keys):
            raise AssertionError("saved output contains non-finite values")
    with np.load(directory / "references.npz") as references:
        if len(references.files) != target_count:
            raise AssertionError("saved references do not cover every target")

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
        int(row["conditioned_direction_count"]) for row in rows if row["arm"] == ARMS[0]
    )
    if len(index) != expected_states:
        raise AssertionError("KLL state count differs from direction ledger")

    for relative, digest in manifest["source_sha256"].items():
        snapshot = directory / "source_snapshots" / Path(relative).name
        if sha256_file(snapshot) != digest:
            raise AssertionError("source snapshot hash mismatch")
    registry_copy = directory / REGISTRY.name
    if sha256_file(registry_copy) != manifest["registry_sha256"]:
        raise AssertionError("copied registry hash mismatch")
    return {
        "status": "pass",
        "rows": len(rows),
        "targets": target_count,
        "unique_cells": len(cells),
        "outputs": len(output_keys),
        "references": target_count,
        "kll_states": len(index),
        "kll_bytes": len(state),
        "source_snapshots": len(manifest["source_sha256"]),
        "generator_budget_per_row": budget,
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
) -> tuple[Generator, dict[str, float | int]]:
    model = clone_generator(base, exact_atlas.dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    atlas = exact_atlas if "exact" in arm else kll_atlas
    use_local = "hybrid" in arm or "guarded" in arm
    protect = "guarded" in arm
    macro_steps = len(target_batches)
    requested_local_calls = (
        macro_steps if local_field_call_count is None else int(local_field_call_count)
    )
    if not 0 <= requested_local_calls <= macro_steps:
        raise ValueError("local-field calls must lie between zero and macro steps")
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
    started = time.perf_counter()
    for macro in range(macro_steps):
        start = macro * PARTICLE_POPULATION
        stop = (macro + 1) * PARTICLE_POPULATION
        builder = None
        local_active = macro in local_steps
        if local_active:
            positive = torch.tensor(target_batches[macro], dtype=torch.float32)

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
    projection_products = atlas_projection_products + training_projection_products
    sort_work = atlas_sort_work + training_sort_work
    paper_pairs = (
        len(local_steps) * PARTICLE_POPULATION * (2 * PARTICLE_POPULATION)
        if use_local
        else 0
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
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
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    registry, registry_hash = load_registry()
    from neural_pooled_rank_tests import (
        direction_subset_and_schedule_test,
        quadratic_frame_diagnostic_test,
        transport_gradient_identity_test,
        transport_step_and_guard_test,
    )

    for invariant in (
        direction_subset_and_schedule_test,
        quadratic_frame_diagnostic_test,
        transport_gradient_identity_test,
        transport_step_and_guard_test,
    ):
        invariant()
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
        registry_hash,
        len(targets) * len(ARMS),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    artifact.log(
        f"Conditioned transport development: {len(targets)} targets x {len(ARMS)} arms"
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
    artifact.log("  [PASS] quadratic-frame and transport regression suite")
    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for target in targets:
        started = time.perf_counter()
        dimension = int(target["dimension"])
        directions, direction_record = conditioned_directions(target)
        diagnostics = quadratic_frame_diagnostics(directions)
        if not diagnostics.is_full_rank or diagnostics.condition_number > 25.0:
            raise AssertionError("conditioned direction certificate was lost")
        heldout = np.asarray(target["heldout_directions"], dtype=float)
        data = normalized_target_data(target, 0, profile)
        exact_atlas = exact_target_atlas(data.pool, directions, profile.atlas_knots)
        kll_atlas = apache_kll_target_atlas(
            data.pool, directions, profile.atlas_knots, k=128
        )
        artifact.record_kll(target["name"], kll_atlas)
        references[safe_key(target["name"])] = data.reference.astype(np.float32)

        model_seed = mixed_seed(int(target["seeds"]["model"]), 0, 11)
        base = Generator(dimension, "concentrated", model_seed).to(dtype=torch.float32)
        latent_seed = mixed_seed(int(target["seeds"]["training_latent"]), 0, 20)
        latent_bank = torch.tensor(
            np.random.default_rng(latent_seed).normal(
                size=(
                    profile.macro_steps * PARTICLE_POPULATION,
                    base.latent_dimension,
                )
            ),
            dtype=torch.float32,
        )
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
        for arm in ARMS:
            model, ledger = train_arm(
                arm,
                base,
                latent_bank,
                target_batches,
                exact_atlas,
                kll_atlas,
                permutations,
                active_direction_count=(
                    None if args.active_directions == 0 else args.active_directions
                ),
                direction_schedule_seed=mixed_seed(
                    int(target["seeds"]["reserved"]), 0, 117
                ),
                local_field_call_count=(
                    None if args.local_field_calls == -1 else args.local_field_calls
                ),
            )
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
