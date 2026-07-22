"""Correctness and stress audit for the promoted Apache projected KLL.

The audit compares Apache DataSketches 5.2.0 against exact empirical ranks and
the development fixed-capacity compactor.  It does not use generator endpoint
metrics and is safe to run before freezing the confirmatory registry.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNROOT = HERE / "kll_audit_runs"
sys.path.insert(0, str(HERE))

import datasketches  # noqa: E402

from persistent_quantile_transport import midpoint_grid  # noqa: E402
from projected_quantile_accumulators import (  # noqa: E402
    RandomCompactorSketch,
    inverted_empirical_quantiles,
)
from standard_projected_kll import (  # noqa: E402
    REQUIRED_DATASKETCHES_VERSION,
    empirical_rank_error,
    invariant_tests,
    require_datasketches,
)

MASTER = 20260724


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AuditConfig:
    n: int
    knots: int
    k: int
    repeats: int
    theoretical_rank_error: float


class Artifact:
    def __init__(self, config: AuditConfig) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = RUNROOT / f"{stamp}-k{config.k}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.lines: list[str] = []
        sources = [
            HERE / "audit_projected_kll.py",
            HERE / "standard_projected_kll.py",
            HERE / "projected_quantile_accumulators.py",
            HERE / "PSQTAccumulatorConfirmatoryRoadmap.md",
        ]
        self.manifest = {
            "status": "pre-confirmation-engineering-audit",
            "master_seed": MASTER,
            "config": asdict(config),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status": _git("status", "--short").splitlines(),
            "python": sys.version,
            "numpy": np.__version__,
            "datasketches": importlib.metadata.version("datasketches"),
            "platform": platform.platform(),
            "command": sys.argv,
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path) for path in sources
            },
        }
        for source in sources:
            shutil.copy2(source, self.directory / f"source_{source.name}")

    def log(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def finish(self) -> None:
        self.manifest["wall_seconds"] = time.time() - self.started
        with (self.directory / "manifest.json").open(
                "w", encoding="utf-8") as stream:
            json.dump(self.manifest, stream, indent=2)
        with (self.directory / "RESULTS.md").open(
                "w", encoding="utf-8") as stream:
            stream.write("\n".join(self.lines) + "\n")


def stress_streams(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(MASTER)
    streams = {
        "gaussian": rng.normal(size=n),
        "student-t3": rng.standard_t(3, size=n),
        "lognormal-centered": rng.lognormal(0.0, 0.9, size=n) - 1.5,
        "equal-separated": np.where(
            rng.random(n) < 0.5,
            rng.normal(-2.0, 0.05, n), rng.normal(2.0, 0.05, n)),
        "rare-01": np.where(
            rng.random(n) < 0.01,
            rng.normal(3.0, 0.05, n), rng.normal(-1.0, 0.10, n)),
        "rare-02": np.where(
            rng.random(n) < 0.02,
            rng.normal(3.0, 0.05, n), rng.normal(-1.0, 0.10, n)),
        "rare-05": np.where(
            rng.random(n) < 0.05,
            rng.normal(3.0, 0.05, n), rng.normal(-1.0, 0.10, n)),
        "rare-10": np.where(
            rng.random(n) < 0.10,
            rng.normal(3.0, 0.05, n), rng.normal(-1.0, 0.10, n)),
        "repeated-discrete": rng.choice(
            np.asarray([-4.0, -1.0, 0.0, 7.0]), n,
            p=np.asarray([0.02, 0.48, 0.45, 0.05])),
    }
    return {name: np.asarray(values, dtype=np.float32)
            for name, values in streams.items()}


def ordered(values: np.ndarray, order: str,
            rng: np.random.Generator) -> np.ndarray:
    if order == "shuffled":
        return values[rng.permutation(len(values))]
    if order == "ascending":
        return np.sort(values)
    if order == "alternating-extremes":
        sorted_values = np.sort(values)
        indices = np.empty(len(values), dtype=int)
        left, right = 0, len(values) - 1
        for index in range(len(values)):
            if index % 2 == 0:
                indices[index] = left
                left += 1
            else:
                indices[index] = right
                right -= 1
        return sorted_values[indices]
    raise ValueError(order)


def apache_row(values: np.ndarray, grid: np.ndarray, k: int,
               *, chunks: int = 1, merged: bool = False) -> tuple[dict, bytes]:
    if merged:
        parts = []
        for part in np.array_split(values, chunks):
            sketch = datasketches.kll_floats_sketch(k)
            sketch.update(np.ascontiguousarray(part, dtype=np.float32))
            parts.append(sketch)
        sketch = datasketches.kll_floats_sketch(k)
        for part in parts:
            sketch.merge(part)
    else:
        sketch = datasketches.kll_floats_sketch(k)
        for part in np.array_split(values, chunks):
            sketch.update(np.ascontiguousarray(part, dtype=np.float32))
    estimates = np.asarray(
        sketch.get_quantiles(grid.tolist(), inclusive=True), dtype=np.float32)
    exact = inverted_empirical_quantiles(values, grid).astype(np.float32)
    errors = empirical_rank_error(values, grid, estimates)
    payload = bytes(sketch.serialize())
    replay = datasketches.kll_floats_sketch.deserialize(payload)
    replayed = np.asarray(
        replay.get_quantiles(grid.tolist(), inclusive=True), dtype=np.float32)
    return {
        "implementation": "apache",
        "chunks": chunks,
        "merged": int(merged),
        "n": int(sketch.n),
        "retained_items": int(sketch.num_retained),
        "serialized_bytes": len(payload),
        "max_rank_error": float(errors.max()),
        "mean_rank_error": float(errors.mean()),
        "quantile_rmse": float(np.sqrt(np.mean((estimates - exact) ** 2))),
        "monotone": int(np.all(np.diff(estimates) >= 0.0)),
        "observed_support": int(np.all(np.isin(estimates, values))),
        "serialized_replay": int(np.array_equal(estimates, replayed)),
    }, payload


def local_row(values: np.ndarray, grid: np.ndarray, k: int,
              seed: int, *, chunks: int = 1) -> dict:
    sketch = RandomCompactorSketch(k, seed=seed)
    for part in np.array_split(values, chunks):
        sketch.update(part)
    estimates = sketch.quantiles(grid).astype(np.float32)
    exact = inverted_empirical_quantiles(values, grid).astype(np.float32)
    errors = empirical_rank_error(values, grid, estimates)
    return {
        "implementation": "local-fixed-capacity",
        "chunks": chunks,
        "merged": 0,
        "n": sketch.count,
        "retained_items": sketch.retained_items,
        "serialized_bytes": sketch.retained_items * 4,
        "max_rank_error": float(errors.max()),
        "mean_rank_error": float(errors.mean()),
        "quantile_rmse": float(np.sqrt(np.mean((estimates - exact) ** 2))),
        "monotone": int(np.all(np.diff(estimates) >= 0.0)),
        "observed_support": int(np.all(np.isin(estimates, values))),
        "serialized_replay": 1,
    }


def percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=19200)
    parser.add_argument("--knots", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    require_datasketches()
    if importlib.metadata.version("datasketches") != \
            REQUIRED_DATASKETCHES_VERSION:
        raise RuntimeError("unfrozen datasketches version")
    theoretical = float(
        datasketches.kll_floats_sketch.get_normalized_rank_error(
            args.k, False))
    config = AuditConfig(
        args.n, args.knots, args.k, args.repeats, theoretical)
    artifact = Artifact(config)
    artifact.log("# Apache projected KLL pre-confirmation audit")
    artifact.log()
    artifact.log("Status: engineering audit; no endpoint targets used.")
    artifact.log()
    invariant_tests()
    artifact.log("- projected Apache KLL invariants: PASS")
    artifact.log(
        f"- official single-sided normalized rank error at k={args.k}: "
        f"`{theoretical:.6f}`")
    artifact.log(
        "- exact fixed-seed replay: unavailable in Apache Python API; "
        "serialized trained-state replay is audited instead")

    grid = midpoint_grid(args.knots)
    rows: list[dict] = []
    stream_payload = stress_streams(args.n)
    for stream_index, (name, base) in enumerate(stream_payload.items()):
        for order_index, order_name in enumerate((
                "shuffled", "ascending", "alternating-extremes")):
            order_rng = np.random.default_rng(
                MASTER + 1009 * stream_index + 17 * order_index)
            values = ordered(base, order_name, order_rng)
            for repeat in range(args.repeats):
                standard, _ = apache_row(values, grid, args.k)
                rows.append({
                    "stream": name, "order": order_name,
                    "repeat": repeat, **standard,
                })
                local = local_row(
                    values, grid, args.k,
                    MASTER + 1_000_003 * stream_index +
                    10_007 * order_index + repeat)
                rows.append({
                    "stream": name, "order": order_name,
                    "repeat": repeat, **local,
                })
            # Partition sensitivity and explicit mergeability audit.
            chunked, _ = apache_row(values, grid, args.k, chunks=17)
            rows.append({
                "stream": name, "order": order_name,
                "repeat": -1, **chunked,
            })
            merged, _ = apache_row(
                values, grid, args.k, chunks=17, merged=True)
            rows.append({
                "stream": name, "order": order_name,
                "repeat": -2, **merged,
            })

    memory_rows = []
    memory_rng = np.random.default_rng(MASTER + 99)
    for n in (1000, 10000, 100000):
        values = memory_rng.normal(size=n).astype(np.float32)
        row, _ = apache_row(values, grid, args.k)
        memory_rows.append({"n": n, **row})

    with (artifact.directory / "rows.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (artifact.directory / "memory_scaling.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(memory_rows[0]))
        writer.writeheader()
        writer.writerows(memory_rows)

    apache = [row for row in rows if row["implementation"] == "apache"]
    local = [
        row for row in rows
        if row["implementation"] == "local-fixed-capacity"]
    apache_errors = [float(row["max_rank_error"]) for row in apache]
    local_errors = [float(row["max_rank_error"]) for row in local]
    gates = {
        "counts": all(int(row["n"]) == args.n for row in apache),
        "monotone": all(int(row["monotone"]) == 1 for row in apache),
        "observed_support": all(
            int(row["observed_support"]) == 1 for row in apache),
        "serialized_replay": all(
            int(row["serialized_replay"]) == 1 for row in apache),
        "max_rank_error_le_3x_official": (
            max(apache_errors) <= 3.0 * theoretical),
        "p95_rank_error_le_2x_official": (
            percentile(apache_errors, 0.95) <= 2.0 * theoretical),
        "bounded_state_at_100k": (
            int(memory_rows[-1]["serialized_bytes"]) <
            4 * int(memory_rows[-1]["n"])),
    }
    summary = {
        "config": asdict(config),
        "apache": {
            "max_rank_error": max(apache_errors),
            "p95_max_rank_error": percentile(apache_errors, 0.95),
            "median_max_rank_error": percentile(apache_errors, 0.5),
            "median_quantile_rmse": percentile([
                float(row["quantile_rmse"]) for row in apache], 0.5),
            "median_retained_items": percentile([
                float(row["retained_items"]) for row in apache], 0.5),
            "median_serialized_bytes": percentile([
                float(row["serialized_bytes"]) for row in apache], 0.5),
        },
        "local_fixed_capacity": {
            "max_rank_error": max(local_errors),
            "p95_max_rank_error": percentile(local_errors, 0.95),
            "median_max_rank_error": percentile(local_errors, 0.5),
            "median_quantile_rmse": percentile([
                float(row["quantile_rmse"]) for row in local], 0.5),
            "median_retained_items": percentile([
                float(row["retained_items"]) for row in local], 0.5),
        },
        "memory_scaling": memory_rows,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "reproducibility_boundary": (
            "Apache DataSketches Python KLL does not expose its compaction "
            "RNG seed; final serialized states must be preserved."),
    }
    with (artifact.directory / "summary.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    artifact.log()
    artifact.log("## Rank audit")
    artifact.log(
        "- Apache maximum observed rank error: "
        f"`{summary['apache']['max_rank_error']:.6f}`")
    artifact.log(
        "- Apache 95th percentile maximum error: "
        f"`{summary['apache']['p95_max_rank_error']:.6f}`")
    artifact.log(
        "- Apache median maximum error: "
        f"`{summary['apache']['median_max_rank_error']:.6f}`")
    artifact.log(
        "- local fixed-capacity median maximum error: "
        f"`{summary['local_fixed_capacity']['median_max_rank_error']:.6f}`")
    artifact.log("- gates:")
    for gate, passed in gates.items():
        artifact.log(f"  - `{gate}`: **{'PASS' if passed else 'FAIL'}**")
    artifact.log(
        "- engineering audit: "
        f"**{'PASS' if summary['all_gates_pass'] else 'FAIL'}**")
    artifact.log()
    artifact.log("## Promotion decision")
    artifact.log()
    artifact.log(
        "Use Apache DataSketches KLL k=128 for the quality arm. Preserve "
        "serialized trained states because its Python API does not expose a "
        "seed for randomized compactions. The local fixed-capacity compactor "
        "remains development history, not the promoted implementation.")
    artifact.finish()
    print(f"artifact: {artifact.directory}", flush=True)


if __name__ == "__main__":
    main()
