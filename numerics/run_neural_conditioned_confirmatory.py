"""Run the frozen multidimensional neural conditioned-transport confirmation."""

from __future__ import annotations

import csv
from dataclasses import asdict
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

from neural_pooled_rank import QuantileAtlas, exact_target_atlas
from run_conditioned_transport_development import (
    LEARNING_RATE,
    PARTICLE_POPULATION,
    PROFILES as CTA_PROFILES,
    train_arm as train_cta_arm,
)
from run_neural_pooled_rank_development import (
    ADAM_LR,
    Generator,
    PROFILES as BASELINE_PROFILES,
    evaluate_output,
    mixed_seed,
    normalized_target_data,
    safe_key,
    sha256_file,
    train_neural_arm,
)


HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "neural_conditioned_confirmatory_registry_v3.json"
REGISTRY_HASH = REGISTRY.with_suffix(REGISTRY.suffix + ".sha256")
FREEZE = HERE / "neural_conditioned_confirmatory_freeze_v3.json"
TABLES = HERE / "neural_conditioned_confirmatory_atlases_v3.npz"
STATES = HERE / "neural_conditioned_confirmatory_kll_v3.bin"
STATE_INDEX = HERE / "neural_conditioned_confirmatory_kll_index_v3.json"
RUN_ROOT = HERE / "neural_conditioned_confirmatory_runs"
INITIALIZATIONS = ("concentrated", "broad")
ARMS = (
    "paper-neural",
    "minibatch-sw",
    "kll-small-rsr",
    "cta-exact-hybrid",
    "cta-kll-hybrid",
)
METRICS = ("ed2", "heldout_sw1")
BOOTSTRAP_DRAWS = 5000


def load_and_verify() -> tuple[dict, dict, list[dict], bytes]:
    registry_payload = REGISTRY.read_bytes()
    registry_digest = hashlib.sha256(registry_payload).hexdigest()
    if registry_digest != REGISTRY_HASH.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("confirmatory registry sidecar mismatch")
    registry = json.loads(registry_payload)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (
        registry.get("schema") != "neural-conditioned-confirmatory-v1"
        or freeze.get("schema") != "neural-conditioned-freeze-v1"
        or freeze.get("registry_sha256") != registry_digest
    ):
        raise RuntimeError("registry/freeze schema mismatch")
    artifacts = {
        TABLES.name: TABLES,
        STATES.name: STATES,
        STATE_INDEX.name: STATE_INDEX,
    }
    for name, path in artifacts.items():
        if sha256_file(path) != freeze["artifact_sha256"][name]:
            raise RuntimeError(f"frozen artifact hash mismatch: {name}")
    for relative, digest in freeze["source_sha256"].items():
        path = HERE.parent / relative
        if sha256_file(path) != digest:
            raise RuntimeError(f"frozen source hash mismatch: {relative}")
    state_index = json.loads(STATE_INDEX.read_text(encoding="utf-8"))
    state = STATES.read_bytes()
    position = 0
    for entry in state_index:
        if int(entry["offset"]) != position:
            raise RuntimeError("KLL state offsets are not contiguous")
        length = int(entry["length"])
        payload = state[position : position + length]
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise RuntimeError("KLL payload hash mismatch")
        position += length
    if position != len(state):
        raise RuntimeError("KLL state index does not consume its binary file")
    return registry, freeze, state_index, state


def frozen_atlases(
    target: dict,
    tables: Any,
    freeze: dict,
    state_index: list[dict],
    state: bytes,
) -> tuple[QuantileAtlas, QuantileAtlas, dict]:
    target_index = int(target["index"])
    prefix = f"t{target_index:03d}"
    directions = np.asarray(tables[f"{prefix}_directions"], dtype=float)
    metadata = freeze["target_metadata"][target_index]
    entries = sorted(
        (entry for entry in state_index if int(entry["target_index"]) == target_index),
        key=lambda entry: int(entry["direction"]),
    )
    payloads = tuple(
        state[int(entry["offset"]) : int(entry["offset"]) + int(entry["length"])]
        for entry in entries
    )
    exact = QuantileAtlas(
        directions=directions,
        grid=tables[f"{prefix}_exact_grid"],
        quantiles=tables[f"{prefix}_exact_quantiles"],
        source="frozen-exact-inverted-ecdf",
        target_count=CTA_PROFILES["consumed"].target_pool,
    )
    kll = QuantileAtlas(
        directions=directions,
        grid=tables[f"{prefix}_kll_grid"],
        quantiles=tables[f"{prefix}_kll_quantiles"],
        source="frozen-apache-kll-k128",
        target_count=CTA_PROFILES["consumed"].target_pool,
        normalized_rank_error=float(metadata["kll_normalized_rank_error"]),
        sketch_payloads=payloads,
    )
    if len(payloads) != len(directions):
        raise RuntimeError("frozen target omitted KLL directions")
    return exact, kll, metadata


def subset_atlas(atlas: QuantileAtlas, count: int) -> QuantileAtlas:
    payloads = None
    if atlas.sketch_payloads is not None:
        payloads = atlas.sketch_payloads[:count]
    return QuantileAtlas(
        directions=atlas.directions[:count],
        grid=atlas.grid,
        quantiles=atlas.quantiles[:count],
        source=f"{atlas.source}-first{count}",
        target_count=atlas.target_count,
        normalized_rank_error=atlas.normalized_rank_error,
        sketch_payloads=payloads,
    )


def stratified_draws(registry: dict) -> np.ndarray:
    cells: dict[tuple[int, str], list[int]] = {}
    for position, target in enumerate(registry["targets"]):
        cells.setdefault((int(target["dimension"]), target["family"]), []).append(
            position
        )
    if len(cells) != 16 or any(len(indices) != 4 for indices in cells.values()):
        raise RuntimeError("bootstrap cells differ from frozen 16 x 4 design")
    rng = np.random.default_rng(int(registry["bootstrap_seed"]))
    draws = np.empty((BOOTSTRAP_DRAWS, len(registry["targets"])), dtype=int)
    offset = 0
    for key in sorted(cells):
        indices = np.asarray(cells[key], dtype=int)
        draws[:, offset : offset + len(indices)] = rng.choice(
            indices, size=(BOOTSTRAP_DRAWS, len(indices)), replace=True
        )
        offset += len(indices)
    return draws


def comparison(
    numerator: np.ndarray, denominator: np.ndarray, draws: np.ndarray
) -> dict[str, float]:
    logs = np.log((numerator + 1e-12) / (denominator + 1e-12))
    estimates = np.exp(np.mean(logs[draws], axis=1))
    return {
        "ratio": float(np.exp(np.mean(logs))),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def analyze(rows: list[dict[str, Any]], registry: dict) -> dict[str, Any]:
    targets = [target["name"] for target in registry["targets"]]
    reduced: dict[tuple[str, str, str], float] = {}
    for target in targets:
        for arm in ARMS:
            selected = [
                row for row in rows if row["target"] == target and row["arm"] == arm
            ]
            if len(selected) != len(INITIALIZATIONS):
                raise RuntimeError("target reduction has incomplete initializations")
            for metric in (*METRICS, "mode_coverage", "rare_mass_error"):
                values = np.asarray([float(row[metric]) for row in selected])
                reduced[(target, arm, metric)] = (
                    float(np.nanmedian(values))
                    if not np.all(np.isnan(values))
                    else math.nan
                )
    arrays = {
        (arm, metric): np.asarray(
            [reduced[(target, arm, metric)] for target in targets]
        )
        for arm in ARMS
        for metric in METRICS
    }
    draws = stratified_draws(registry)
    comparisons: dict[str, dict[str, float]] = {}
    for candidate in ("cta-exact-hybrid", "cta-kll-hybrid"):
        for baseline in ("paper-neural", "minibatch-sw", "kll-small-rsr"):
            for metric in METRICS:
                key = f"{candidate}_vs_{baseline}_{metric}"
                comparisons[key] = comparison(
                    arrays[(candidate, metric)], arrays[(baseline, metric)], draws
                )
    envelope: dict[str, np.ndarray] = {}
    for metric in METRICS:
        envelope[metric] = np.minimum.reduce(
            [
                arrays[(baseline, metric)]
                for baseline in ("paper-neural", "minibatch-sw", "kll-small-rsr")
            ]
        )
        comparisons[f"cta-exact-hybrid_vs_envelope_{metric}"] = comparison(
            arrays[("cta-exact-hybrid", metric)], envelope[metric], draws
        )
        comparisons[f"cta-kll-hybrid_vs_exact_{metric}"] = comparison(
            arrays[("cta-kll-hybrid", metric)],
            arrays[("cta-exact-hybrid", metric)],
            draws,
        )

    exact_both_paper = int(
        np.sum(
            (arrays[("cta-exact-hybrid", "ed2")] < arrays[("paper-neural", "ed2")])
            & (
                arrays[("cta-exact-hybrid", "heldout_sw1")]
                < arrays[("paper-neural", "heldout_sw1")]
            )
        )
    )
    exact_both_envelope = int(
        np.sum(
            (arrays[("cta-exact-hybrid", "ed2")] < envelope["ed2"])
            & (arrays[("cta-exact-hybrid", "heldout_sw1")] < envelope["heldout_sw1"])
        )
    )
    kll_both_paper = int(
        np.sum(
            (arrays[("cta-kll-hybrid", "ed2")] < arrays[("paper-neural", "ed2")])
            & (
                arrays[("cta-kll-hybrid", "heldout_sw1")]
                < arrays[("paper-neural", "heldout_sw1")]
            )
        )
    )
    cell_exact_paper: dict[str, float] = {}
    cell_kll_exact: dict[str, float] = {}
    for dimension in registry["dimensions"]:
        for family in registry["families"]:
            positions = [
                index
                for index, target in enumerate(registry["targets"])
                if int(target["dimension"]) == int(dimension)
                and target["family"] == family
            ]
            label = f"d{dimension}-{family}"
            cell_exact_paper[label] = float(
                np.exp(
                    np.mean(
                        np.log(
                            (arrays[("cta-exact-hybrid", "ed2")][positions] + 1e-12)
                            / (arrays[("paper-neural", "ed2")][positions] + 1e-12)
                        )
                    )
                )
            )
            cell_kll_exact[label] = float(
                np.exp(
                    np.mean(
                        np.log(
                            (arrays[("cta-kll-hybrid", "ed2")][positions] + 1e-12)
                            / (arrays[("cta-exact-hybrid", "ed2")][positions] + 1e-12)
                        )
                    )
                )
            )
    rare_targets = [
        target["name"]
        for target in registry["targets"]
        if target["family"] == "rare-gmm"
    ]
    rare = {}
    for arm in ("paper-neural", "cta-exact-hybrid", "cta-kll-hybrid"):
        rare[arm] = {
            "median_mode_coverage": float(
                np.nanmedian(
                    [reduced[(target, arm, "mode_coverage")] for target in rare_targets]
                )
            ),
            "median_rare_mass_error": float(
                np.nanmedian(
                    [
                        reduced[(target, arm, "rare_mass_error")]
                        for target in rare_targets
                    ]
                )
            ),
        }
    divergence = {
        arm: int(sum(int(row["diverged"]) for row in rows if row["arm"] == arm))
        for arm in ARMS
    }
    c = comparisons
    paper_rare_error = rare["paper-neural"]["median_rare_mass_error"]
    exact_gates = {
        "paper_ed2": c["cta-exact-hybrid_vs_paper-neural_ed2"]["ratio"] < 0.75
        and c["cta-exact-hybrid_vs_paper-neural_ed2"]["ci_high"] < 1.0,
        "paper_sw1": c["cta-exact-hybrid_vs_paper-neural_heldout_sw1"]["ratio"] < 0.80
        and c["cta-exact-hybrid_vs_paper-neural_heldout_sw1"]["ci_high"] < 1.0,
        "envelope_ed2": c["cta-exact-hybrid_vs_envelope_ed2"]["ratio"] < 0.90
        and c["cta-exact-hybrid_vs_envelope_ed2"]["ci_high"] < 1.0,
        "envelope_sw1": c["cta-exact-hybrid_vs_envelope_heldout_sw1"]["ratio"] < 0.90
        and c["cta-exact-hybrid_vs_envelope_heldout_sw1"]["ci_high"] < 1.0,
        "paper_both_wins": exact_both_paper >= 48,
        "envelope_both_wins": exact_both_envelope >= 40,
        "cell_robustness": max(cell_exact_paper.values()) <= 1.10,
        "zero_divergence": divergence["cta-exact-hybrid"] == 0,
        "rare_modes": rare["cta-exact-hybrid"]["median_mode_coverage"]
        >= rare["paper-neural"]["median_mode_coverage"]
        and (
            paper_rare_error == 0.0
            or rare["cta-exact-hybrid"]["median_rare_mass_error"]
            <= 0.80 * paper_rare_error
        ),
    }
    kll_gates = {
        "retain_ed2": c["cta-kll-hybrid_vs_exact_ed2"]["ratio"] <= 1.15
        and c["cta-kll-hybrid_vs_exact_ed2"]["ci_high"] <= 1.25,
        "retain_sw1": c["cta-kll-hybrid_vs_exact_heldout_sw1"]["ratio"] <= 1.15
        and c["cta-kll-hybrid_vs_exact_heldout_sw1"]["ci_high"] <= 1.25,
        "paper_both_wins": kll_both_paper >= 45,
        "cell_robustness": max(cell_kll_exact.values()) <= 1.25,
        "zero_divergence": divergence["cta-kll-hybrid"] == 0,
    }
    return {
        "comparisons": comparisons,
        "wins": {
            "exact_both_vs_paper": exact_both_paper,
            "exact_both_vs_envelope": exact_both_envelope,
            "kll_both_vs_paper": kll_both_paper,
        },
        "cell_exact_vs_paper_ed2": cell_exact_paper,
        "cell_kll_vs_exact_ed2": cell_kll_exact,
        "rare": rare,
        "divergence": divergence,
        "exact_gates": exact_gates,
        "exact_pass": all(exact_gates.values()),
        "kll_gates": kll_gates,
        "kll_pass": all(kll_gates.values()),
    }


def write_artifact(
    directory: Path,
    rows: list[dict[str, Any]],
    outputs: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    results: dict,
    registry: dict,
    freeze: dict,
    started: float,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "outputs.npz", **outputs)
    np.savez_compressed(directory / "references.npz", **references)
    (directory / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen_files = (REGISTRY, REGISTRY_HASH, FREEZE, TABLES, STATES, STATE_INDEX)
    for path in frozen_files:
        shutil.copy2(path, directory / path.name)
    for relative in freeze["source_sha256"]:
        source = HERE.parent / relative
        destination = directory / "source_snapshots" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = {
        "schema": "neural-conditioned-confirmatory-run-v1",
        "command": sys.argv,
        "registry_sha256": freeze["registry_sha256"],
        "freeze_sha256": sha256_file(FREEZE),
        "rows": len(rows),
        "outputs": len(outputs),
        "references": len(references),
        "wall_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "frozen_file_sha256": {path.name: sha256_file(path) for path in frozen_files},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cells = {(row["target"], row["initialization"], row["arm"]) for row in rows}
    expected = len(registry["targets"]) * len(INITIALIZATIONS) * len(ARMS)
    if len(rows) != expected or len(cells) != expected:
        raise RuntimeError("confirmatory artifact has incomplete/duplicate cells")
    if len(outputs) != expected or len(references) != len(registry["targets"]):
        raise RuntimeError("confirmatory saved arrays have incomplete coverage")
    if any(int(row["generator_example_evals_training"]) != 20480 for row in rows):
        raise RuntimeError("confirmatory row missed generator-example budget")
    audit = {
        "status": "pass",
        "rows": len(rows),
        "unique_cells": len(cells),
        "outputs": len(outputs),
        "references": len(references),
        "exact_pass": results["exact_pass"],
        "kll_pass": results["kll_pass"],
    }
    (directory / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    registry, freeze, state_index, state = load_and_verify()
    baseline_profile = BASELINE_PROFILES["standard"]
    cta_profile = CTA_PROFILES["consumed"]
    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    with np.load(TABLES) as tables:
        for target in registry["targets"]:
            target_started = time.perf_counter()
            dimension = int(target["dimension"])
            data = normalized_target_data(target, 0, cta_profile)
            exact, kll, metadata = frozen_atlases(
                target, tables, freeze, state_index, state
            )
            prefix = f"t{int(target['index']):03d}"
            frozen_directions = np.asarray(tables[f"{prefix}_directions"], dtype=float)
            recomputed = exact_target_atlas(
                data.pool, frozen_directions, exact.knot_count
            )
            if not np.array_equal(recomputed.quantiles, exact.quantiles):
                raise RuntimeError("frozen exact atlas does not match regenerated pool")
            base_exact = subset_atlas(exact, 64)
            base_kll = subset_atlas(kll, 64)
            base_directions = np.asarray(target["training_directions"], dtype=float)
            if not np.allclose(base_exact.directions, base_directions, atol=2e-15):
                raise RuntimeError("conditioned frame did not preserve registered base")
            heldout = np.asarray(target["heldout_directions"], dtype=float)
            references[safe_key(target["name"])] = data.reference.astype(np.float32)

            latent_seed = mixed_seed(int(target["seeds"]["training_latent"]), 0, 20)
            latent_bank = torch.tensor(
                np.random.default_rng(latent_seed).normal(
                    size=(baseline_profile.generator_budget, max(4, dimension))
                ),
                dtype=torch.float32,
            )
            order_rng = np.random.default_rng(
                mixed_seed(int(target["seeds"]["minibatch_order"]), 0, 31)
            )
            target_order = order_rng.permutation(len(data.pool))[
                : cta_profile.macro_steps * PARTICLE_POPULATION
            ]
            target_batches = data.pool[target_order].reshape(
                cta_profile.macro_steps, PARTICLE_POPULATION, dimension
            )
            permutations = tuple(
                torch.tensor(
                    order_rng.permutation(PARTICLE_POPULATION), dtype=torch.long
                )
                for _ in range(cta_profile.macro_steps)
            )
            evaluation_seed = mixed_seed(
                int(target["seeds"]["evaluation_latent"]), 0, 21
            )
            evaluation_latent = torch.tensor(
                np.random.default_rng(evaluation_seed).normal(
                    size=(cta_profile.evaluation_samples, max(4, dimension))
                ),
                dtype=torch.float32,
            )
            for initialization in INITIALIZATIONS:
                purpose = 11 if initialization == "concentrated" else 12
                model_seed = mixed_seed(int(target["seeds"]["model"]), 0, purpose)
                base = Generator(dimension, initialization, model_seed).to(
                    dtype=torch.float32
                )
                for arm in ARMS:
                    if arm in ("paper-neural", "minibatch-sw", "kll-small-rsr"):
                        model, ledger = train_neural_arm(
                            arm,
                            base,
                            latent_bank,
                            data.pool,
                            base_directions,
                            base_exact,
                            base_kll,
                            baseline_profile,
                            learning_rate=ADAM_LR * 16.0,
                        )
                    else:
                        model, ledger_dict = train_cta_arm(
                            arm,
                            base,
                            latent_bank[
                                : cta_profile.macro_steps * PARTICLE_POPULATION
                            ],
                            target_batches,
                            exact,
                            kll,
                            permutations,
                        )
                        ledger_dict["diverged"] = 0
                        ledger = ledger_dict
                    with torch.no_grad():
                        final = model(evaluation_latent).numpy()
                    metrics = evaluate_output(final, data, base_directions, heldout)
                    ledger_values = (
                        asdict(ledger) if not isinstance(ledger, dict) else ledger
                    )
                    row = {
                        "arm": arm,
                        "target": target["name"],
                        "family": target["family"],
                        "dimension": dimension,
                        "instance": target["instance"],
                        "initialization": initialization,
                        "learning_rate": LEARNING_RATE,
                        "conditioned_direction_count": metadata[
                            "conditioned_direction_count"
                        ],
                        "quadratic_condition_number": metadata[
                            "quadratic_condition_number"
                        ],
                        **metrics,
                        **ledger_values,
                        "model_parameters": sum(
                            parameter.numel() for parameter in model.parameters()
                        ),
                    }
                    rows.append(row)
                    outputs[safe_key(f"{arm}__{target['name']}__{initialization}")] = (
                        final.astype(np.float32)
                    )
            print(
                f"completed {target['name']} ({len(rows)}/640 rows) in "
                f"{time.perf_counter() - target_started:.2f}s",
                flush=True,
            )
    results = analyze(rows, registry)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    directory = RUN_ROOT / f"{stamp}-confirmatory"
    write_artifact(
        directory, rows, outputs, references, results, registry, freeze, started
    )
    print(
        json.dumps(
            {"exact_pass": results["exact_pass"], "kll_pass": results["kll_pass"]}
        )
    )
    print(f"artifact: {directory}")


if __name__ == "__main__":
    main()
