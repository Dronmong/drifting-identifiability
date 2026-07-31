"""Merge a complete set of Phase-1 target-shard artifacts.

The merge performs structural, hash, key-disjointness, row-count, and budget
checks before creating a new immutable aggregate artifact.  It does not rerun
or alter any statistical result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

from run_neural_pooled_rank_development import (
    ARMS,
    CEILING,
    PROFILES,
    REGISTRY,
    REGISTRY_HASH,
    ROOT,
    RUN_ROOT,
    git,
    load_registry,
    sha256_file,
    summarize,
)


HERE = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def append_npz(destination: dict[str, np.ndarray], path: Path, *, kind: str) -> None:
    with np.load(path) as payload:
        for key in payload.files:
            if key in destination:
                raise RuntimeError(f"duplicate {kind} key {key}")
            destination[key] = payload[key]


def typed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    integer_fields = {
        "dimension",
        "replication",
        "completed_updates",
        "unique_latent_samples",
        "generator_example_evals_training",
        "generator_forward_calls_training",
        "unique_target_observations",
        "target_example_accesses",
        "projection_scalar_products",
        "paper_kernel_pairs",
        "atlas_bytes",
        "kll_serialized_bytes",
        "diverged",
        "evaluation_generator_examples",
        "model_parameters",
    }
    text_fields = {"arm", "kind", "target", "family", "initialization"}
    result: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in text_fields:
                converted[key] = value
            elif key in integer_fields:
                converted[key] = int(value)
            else:
                converted[key] = float(value)
        result.append(converted)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    shards = [path.resolve() for path in args.shards]
    registry, registry_digest = load_registry()
    profile = PROFILES["standard"]
    expected_total_rows = (
        len(registry["targets"])
        * profile.replications
        * len(profile.initializations)
        * (len(ARMS) + 1)
    )

    manifests: list[dict[str, Any]] = []
    shard_indices: set[int] = set()
    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    kll_index: list[dict[str, Any]] = []
    kll_chunks: list[bytes] = []
    kll_offset = 0
    shard_hashes: dict[str, Any] = {}

    for shard in shards:
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        if manifest["profile"]["name"] != "standard":
            raise RuntimeError(f"non-standard shard {shard}")
        if manifest["registry_sha256"] != registry_digest:
            raise RuntimeError(f"registry mismatch in {shard}")
        if manifest["actual_rows"] != manifest["expected_rows"]:
            raise RuntimeError(f"incomplete shard {shard}")
        index = int(manifest["shard_index"])
        if index in shard_indices:
            raise RuntimeError(f"duplicate shard index {index}")
        shard_indices.add(index)
        manifests.append(manifest)

        shard_rows = read_rows(shard / "rows.csv")
        if len(shard_rows) != manifest["actual_rows"]:
            raise RuntimeError(f"row-file mismatch in {shard}")
        rows.extend(shard_rows)
        append_npz(outputs, shard / "outputs.npz", kind="output")
        append_npz(references, shard / "references.npz", kind="reference")

        binary = (shard / "kll_states.bin").read_bytes()
        index_rows = json.loads(
            (shard / "kll_state_index.json").read_text(encoding="utf-8")
        )
        for item in index_rows:
            local_offset = int(item["offset"])
            length = int(item["length"])
            payload = binary[local_offset : local_offset + length]
            if len(payload) != length:
                raise RuntimeError(f"truncated KLL state in {shard}")
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise RuntimeError(f"KLL hash mismatch in {shard}")
            merged = dict(item)
            merged["offset"] = kll_offset + local_offset
            merged["source_shard"] = shard.name
            kll_index.append(merged)
        kll_chunks.append(binary)
        kll_offset += len(binary)
        shard_hashes[shard.name] = {
            name: sha256_file(shard / name)
            for name in (
                "manifest.json",
                "rows.csv",
                "outputs.npz",
                "references.npz",
                "kll_states.bin",
                "kll_state_index.json",
            )
        }

    shard_count_values = {int(manifest["shard_count"]) for manifest in manifests}
    if len(shard_count_values) != 1:
        raise RuntimeError("inconsistent shard counts")
    shard_count = shard_count_values.pop()
    if shard_indices != set(range(shard_count)) or len(shards) != shard_count:
        raise RuntimeError("shard set is incomplete")
    if len(rows) != expected_total_rows:
        raise RuntimeError("aggregate row count is wrong")
    expected_targets = {target["name"] for target in registry["targets"]}
    observed_targets = {row["target"] for row in rows}
    if observed_targets != expected_targets:
        raise RuntimeError("aggregate target set is wrong")
    for row in rows:
        if row["kind"] == "neural" and (
            int(row["generator_example_evals_training"]) != profile.generator_budget
        ):
            raise RuntimeError("merged neural row missed generator budget")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = RUN_ROOT / f"{stamp}-standard-merged"
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(REGISTRY, destination / REGISTRY.name)
    shutil.copy2(REGISTRY_HASH, destination / REGISTRY_HASH.name)
    sources = [
        HERE / "merge_neural_pooled_rank_shards.py",
        HERE / "run_neural_pooled_rank_development.py",
        HERE / "neural_pooled_rank.py",
        HERE / "NeuralPooledRankPhase1Protocol.md",
    ]
    for source in sources:
        target = destination / "source_snapshots" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    with (destination / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(destination / "outputs.npz", **outputs)
    np.savez_compressed(destination / "references.npz", **references)
    (destination / "kll_states.bin").write_bytes(b"".join(kll_chunks))
    (destination / "kll_state_index.json").write_text(
        json.dumps(kll_index, indent=2) + "\n", encoding="utf-8"
    )
    summary = summarize(typed_rows(rows))
    (destination / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": "neural-pooled-rank-standard-merged-v1",
        "profile": profile.__dict__,
        "registry_sha256": registry_digest,
        "git_commit_at_merge": git("rev-parse", "HEAD"),
        "git_dirty_at_merge": bool(git("status", "--porcelain")),
        "command": sys.argv,
        "shard_count": shard_count,
        "shard_indices": sorted(shard_indices),
        "source_shards": [str(path) for path in shards],
        "source_shard_hashes": shard_hashes,
        "actual_rows": len(rows),
        "expected_rows": expected_total_rows,
        "output_count": len(outputs),
        "reference_count": len(references),
        "kll_state_count": len(kll_index),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in sources
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "Neural pooled-rank merged standard development run",
        f"  registry {registry_digest}",
        f"  {len(rows)} rows from {shard_count} complete target shards",
    ]
    for arm in (*ARMS, CEILING):
        values = summary["by_arm"][arm]
        lines.append(
            f"  {arm:28s} ED2={values['median_ed2']:.6f} "
            f"heldoutSW1={values['median_heldout_sw1']:.6f} "
            f"div={values['divergences']}"
        )
    (destination / "summary.md").write_text(
        "```text\n" + "\n".join(lines) + "\n```\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"ARTIFACT={destination}")


if __name__ == "__main__":
    main()
