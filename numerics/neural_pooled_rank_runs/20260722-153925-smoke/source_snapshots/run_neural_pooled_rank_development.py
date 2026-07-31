"""Phase-1 multidimensional neural pooled-rank development runner.

Usage from the repository root:

    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      --with datasketches==5.2.0 \
      python numerics/run_neural_pooled_rank_development.py --profile smoke

The smoke profile validates engineering and accounting.  It is not a
confirmatory performance experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lowdim_drift import drift_paper, energy_distance2
from neural_pooled_rank import (
    QuantileAtlas,
    apache_kll_target_atlas,
    exact_target_atlas,
    frame_diagnostics,
    rank_matched_loss,
    run_sort_rerun_backward,
)
from projected_quantile_accumulators import (
    projected_quantile_table,
    reconstruct_from_quantile_table,
)
from standard_projected_kll import ApacheKLLProjectedAccumulator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGISTRY = HERE / "neural_pooled_rank_development_registry.json"
REGISTRY_HASH = REGISTRY.with_suffix(REGISTRY.suffix + ".sha256")
RUN_ROOT = HERE / "neural_pooled_rank_runs"
ARMS = (
    "paper-neural",
    "minibatch-sw",
    "exact-atlas-rsr",
    "kll-atlas-rsr",
    "kll-small-rsr",
    "kll-paper-hybrid",
)
CEILING = "exact-free-particle-ceiling"
PAPER_TAU = 1.0
PAPER_GAIN = 0.15
ADAM_LR = 1e-3
HIDDEN = 64
TRAIN_DIRECTIONS = 64
HELDOUT_DIRECTIONS = 128


@dataclass(frozen=True)
class Profile:
    name: str
    target_pool: int
    generator_budget: int
    ordinary_batch: int
    large_population: int
    microbatch: int
    atlas_knots: int
    evaluation_samples: int
    free_particle_sweeps: int
    replications: int
    initializations: tuple[str, ...]
    smoke_only: bool

    def validate(self) -> None:
        divisors = (
            self.ordinary_batch,
            2 * self.ordinary_batch,
            2 * self.large_population,
        )
        if any(self.generator_budget % divisor != 0 for divisor in divisors):
            raise ValueError("generator budget does not divide registered batches")
        if self.target_pool != self.generator_budget:
            raise ValueError("target pool and generator budget must match")
        if self.large_population % self.microbatch != 0:
            raise ValueError("large population must divide into microbatches")


PROFILES = {
    "smoke": Profile(
        "smoke", 2048, 2048, 32, 128, 32, 64, 512, 30, 1, ("concentrated",), True
    ),
    "standard": Profile(
        "standard",
        20480,
        20480,
        64,
        512,
        64,
        128,
        2048,
        100,
        3,
        ("concentrated", "broad"),
        False,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_registry() -> tuple[dict, str]:
    payload = REGISTRY.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = REGISTRY_HASH.read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise RuntimeError("development registry hash does not match sidecar")
    registry = json.loads(payload)
    if (
        registry.get("schema") != "neural-pooled-rank-development-v1"
        or registry.get("target_count") != 16
        or registry.get("rejection_count") != 0
    ):
        raise RuntimeError("unexpected neural pooled-rank registry schema")
    return registry, digest


class RunArtifact:
    def __init__(
        self, profile: Profile, registry_digest: str, expected_rows: int
    ) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = RUN_ROOT / f"{stamp}-{profile.name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.lines: list[str] = []
        self.kll_index: list[dict[str, Any]] = []
        self.kll_stream = (self.directory / "kll_states.bin").open("wb")
        sources = [
            HERE / "run_neural_pooled_rank_development.py",
            HERE / "neural_pooled_rank.py",
            HERE / "neural_pooled_rank_tests.py",
            HERE / "generate_neural_pooled_rank_registry.py",
            HERE / "NeuralPooledRankPhase1Protocol.md",
            HERE / "KLLPSQTNeuralAmortizationResearch.md",
            HERE / "lowdim_drift.py",
            HERE / "projected_quantile_accumulators.py",
            HERE / "standard_projected_kll.py",
        ]
        status = git("status", "--porcelain")
        self.manifest = {
            "profile": asdict(profile),
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

    def record_kll(self, target: str, replication: int, atlas: QuantileAtlas) -> None:
        if atlas.sketch_payloads is None:
            raise ValueError("cannot record an atlas without KLL payloads")
        for direction, payload in enumerate(atlas.sketch_payloads):
            offset = self.kll_stream.tell()
            self.kll_stream.write(payload)
            self.kll_index.append(
                {
                    "target": target,
                    "replication": replication,
                    "direction": direction,
                    "offset": offset,
                    "length": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "observations": atlas.target_count,
                }
            )

    def finish(
        self,
        rows: list[dict[str, Any]],
        outputs: dict[str, np.ndarray],
        references: dict[str, np.ndarray],
        summary: dict,
    ) -> None:
        self.kll_stream.close()
        if len(rows) != self.manifest["expected_rows"]:
            raise RuntimeError("row count does not match registered design")
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
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.directory / "summary.md").write_text(
            "```text\n" + "\n".join(self.lines) + "\n```\n", encoding="utf-8"
        )


def mixed_seed(seed: int, replication: int, purpose: int) -> int:
    sequence = np.random.SeedSequence([seed, replication, purpose])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def sample_target(target: dict, count: int, rng: np.random.Generator) -> np.ndarray:
    dimension = int(target["dimension"])
    family = str(target["family"])
    parameters = target["parameters"]
    if family in ("balanced-gmm", "rare-gmm"):
        centers = np.asarray(parameters["centers"], dtype=float)
        weights = np.asarray(parameters["weights"], dtype=float)
        sigmas = np.asarray(parameters["sigmas"], dtype=float)
        components = rng.choice(len(weights), size=count, p=weights)
        return centers[components] + (
            rng.normal(size=(count, dimension)) * sigmas[components, None]
        )
    if family == "correlated-t":
        rotation = np.asarray(parameters["rotation"], dtype=float)
        scales = np.asarray(parameters["axis_scales"], dtype=float)
        degrees = float(parameters["degrees_of_freedom"])
        values = rng.standard_t(degrees, size=(count, dimension)) * scales
        if dimension > 1:
            values[:, 0] += float(parameters["skew_strength"]) * (
                np.abs(values[:, 1]) - np.mean(np.abs(values[:, 1]))
            )
        return values @ rotation.T
    if family == "nonlinear":
        rotation = np.asarray(parameters["rotation"], dtype=float)
        curve = float(parameters["curve_strength"])
        noise = float(parameters["noise_scale"])
        latent = rng.normal(size=(count, dimension))
        values = noise * rng.normal(size=(count, dimension))
        values[:, 0] += latent[:, 0]
        values[:, 1] += curve * (latent[:, 0] ** 2 - 1.0) + 0.35 * latent[:, 1]
        for coordinate in range(2, dimension):
            values[:, coordinate] += 0.45 * latent[:, coordinate] + 0.30 * np.sin(
                (coordinate + 1) * latent[:, 0] / dimension
            )
        return values @ rotation.T
    raise ValueError(f"unknown target family {family}")


@dataclass(frozen=True)
class NormalizedTarget:
    pool: np.ndarray
    reference: np.ndarray
    center: np.ndarray
    scale: float
    normalized_centers: np.ndarray | None
    weights: np.ndarray | None


def normalized_target_data(
    target: dict, replication: int, profile: Profile
) -> NormalizedTarget:
    seeds = target["seeds"]
    pool = sample_target(
        target,
        profile.target_pool,
        np.random.default_rng(mixed_seed(int(seeds["target_pool"]), replication, 1)),
    )
    reference = sample_target(
        target,
        profile.evaluation_samples,
        np.random.default_rng(
            mixed_seed(int(seeds["evaluation_target"]), replication, 2)
        ),
    )
    center = pool.mean(axis=0)
    scale = float(np.sqrt(np.mean((pool - center) ** 2)))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("target normalization scale is degenerate")
    pool = (pool - center) / scale
    reference = (reference - center) / scale
    parameters = target["parameters"]
    normalized_centers = None
    weights = None
    if "centers" in parameters:
        normalized_centers = (
            np.asarray(parameters["centers"], dtype=float) - center
        ) / scale
        weights = np.asarray(parameters["weights"], dtype=float)
    return NormalizedTarget(
        pool=pool,
        reference=reference,
        center=center,
        scale=scale,
        normalized_centers=normalized_centers,
        weights=weights,
    )


class Generator(nn.Module):
    def __init__(self, dimension: int, initialization: str, seed: int) -> None:
        super().__init__()
        self.latent_dimension = max(4, dimension)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers = nn.Sequential(
                nn.Linear(self.latent_dimension, HIDDEN),
                nn.Tanh(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.Tanh(),
                nn.Linear(HIDDEN, dimension),
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    nn.init.zeros_(module.bias)
            output = self.layers[-1]
            factor = {"concentrated": 0.05, "broad": 0.50}.get(initialization)
            if factor is None:
                raise ValueError(f"unknown initialization {initialization}")
            with torch.no_grad():
                output.weight.mul_(factor)

    def forward(self, latent: Tensor) -> Tensor:
        return self.layers(latent)


@torch.no_grad()
def torch_paper_field(
    query: Tensor, positive: Tensor, *, tau: float = PAPER_TAU, mask: bool = True
) -> Tensor:
    if (
        query.ndim != 2
        or positive.ndim != 2
        or query.shape[1] != positive.shape[1]
        or tau <= 0.0
    ):
        raise ValueError("invalid paper-field inputs")
    positive_distance = torch.linalg.vector_norm(
        query[:, None, :] - positive[None, :, :], dim=2
    )
    negative_distance = torch.linalg.vector_norm(
        query[:, None, :] - query[None, :, :], dim=2
    )
    logits = torch.cat(
        [
            -positive_distance / tau,
            -negative_distance / tau,
        ],
        dim=1,
    )
    if mask:
        rows = torch.arange(len(query), device=query.device)
        logits[rows, len(positive) + rows] -= 1e6 / tau
    row = torch.softmax(logits, dim=1)
    column = torch.softmax(logits, dim=0)
    affinity = torch.sqrt(row * column)
    positive_affinity = affinity[:, : len(positive)]
    negative_affinity = affinity[:, len(positive) :]
    positive_weight = positive_affinity * negative_affinity.sum(dim=1, keepdim=True)
    negative_weight = negative_affinity * positive_affinity.sum(dim=1, keepdim=True)
    return positive_weight @ positive - negative_weight @ query


@dataclass
class TrainingLedger:
    completed_updates: int = 0
    unique_latent_samples: int = 0
    generator_example_evals_training: int = 0
    generator_forward_calls_training: int = 0
    unique_target_observations: int = 0
    target_example_accesses: int = 0
    projection_scalar_products: int = 0
    sort_work: float = 0.0
    paper_kernel_pairs: int = 0
    atlas_bytes: int = 0
    kll_serialized_bytes: int = 0
    diverged: int = 0
    wall_seconds: float = 0.0


def atlas_bytes(atlas: QuantileAtlas) -> int:
    return int(atlas.directions.nbytes + atlas.grid.nbytes + atlas.quantiles.nbytes)


def model_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )


def train_neural_arm(
    arm: str,
    base_model: Generator,
    latent_bank: Tensor,
    target_pool: np.ndarray,
    directions: np.ndarray,
    exact_atlas: QuantileAtlas,
    kll_atlas: QuantileAtlas,
    profile: Profile,
) -> tuple[Generator, TrainingLedger]:
    model = Generator(exact_atlas.dimension, "concentrated", 0).to(dtype=torch.float32)
    model.load_state_dict(base_model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    ledger = TrainingLedger(unique_target_observations=len(target_pool))
    started = time.perf_counter()
    direction_count = len(directions)

    if arm in ("paper-neural", "minibatch-sw"):
        batch = profile.ordinary_batch
        updates = profile.generator_budget // batch
        for step in range(updates):
            start, stop = step * batch, (step + 1) * batch
            latent = latent_bank[start:stop]
            positive_np = target_pool[start:stop]
            positive = torch.tensor(positive_np, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            generated = model(latent)
            if arm == "paper-neural":
                field = torch_paper_field(generated.detach(), positive)
                loss = -PAPER_GAIN * torch.mean(torch.sum(generated * field, dim=1))
                ledger.paper_kernel_pairs += batch * (batch + batch)
            else:
                batch_atlas = exact_target_atlas(
                    positive_np, directions, knot_count=batch
                )
                loss = rank_matched_loss(generated, batch_atlas).loss
                ledger.projection_scalar_products += 2 * batch * direction_count
                ledger.sort_work += 2.0 * direction_count * batch * math.log2(batch)
            loss.backward()
            optimizer.step()
            ledger.completed_updates += 1
            if not model_finite(model):
                ledger.diverged = 1
                break
        ledger.unique_latent_samples = updates * batch
        ledger.generator_example_evals_training = updates * batch
        ledger.generator_forward_calls_training = updates
        ledger.target_example_accesses = updates * batch
    else:
        population = (
            profile.ordinary_batch
            if arm == "kll-small-rsr"
            else profile.large_population
        )
        updates = profile.generator_budget // (2 * population)
        atlas = exact_atlas if arm == "exact-atlas-rsr" else kll_atlas
        ledger.atlas_bytes = atlas_bytes(atlas)
        if atlas.sketch_payloads is not None:
            ledger.kll_serialized_bytes = sum(
                len(payload) for payload in atlas.sketch_payloads
            )
        ledger.target_example_accesses = len(target_pool)
        ledger.projection_scalar_products += len(target_pool) * direction_count
        if arm == "exact-atlas-rsr":
            ledger.sort_work += (
                direction_count * len(target_pool) * math.log2(len(target_pool))
            )
        for step in range(updates):
            start, stop = step * population, (step + 1) * population
            latent = latent_bank[start:stop]
            optimizer.zero_grad(set_to_none=True)
            builder = None
            rank_weight = 1.0
            field_weight = 1.0
            if arm == "kll-paper-hybrid":
                positive_np = target_pool[start:stop]
                positive = torch.tensor(positive_np, dtype=torch.float32)

                def builder(
                    run_features: Tensor, positive: Tensor = positive
                ) -> Tensor:
                    return torch_paper_field(run_features, positive)

                progress = step / max(updates - 1, 1)
                rank_weight = 1.0 - 0.75 * progress
                field_weight = PAPER_GAIN * (0.25 + 0.75 * progress)
                ledger.target_example_accesses += population
                ledger.paper_kernel_pairs += population * (population + population)
            run_sort_rerun_backward(
                model,
                latent,
                atlas,
                microbatch=profile.microbatch,
                feature_field_builder=builder,
                rank_loss_weight=rank_weight,
                feature_field_weight=field_weight,
            )
            optimizer.step()
            ledger.completed_updates += 1
            if not model_finite(model):
                ledger.diverged = 1
                break
        ledger.unique_latent_samples = updates * population
        ledger.generator_example_evals_training = updates * 2 * population
        chunks = math.ceil(population / profile.microbatch)
        ledger.generator_forward_calls_training = updates * 2 * chunks
        ledger.projection_scalar_products += (
            updates * 2 * (population * direction_count)
        )
        ledger.sort_work += (
            updates * direction_count * population * math.log2(population)
        )
    ledger.wall_seconds = time.perf_counter() - started
    return model, ledger


def heldout_sw1(
    generated: np.ndarray, reference: np.ndarray, directions: np.ndarray
) -> float:
    probabilities = np.linspace(0.01, 0.99, 99)
    projected_generated = generated @ directions.T
    projected_reference = reference @ directions.T
    q_generated = np.quantile(
        projected_generated, probabilities, axis=0, method="linear"
    )
    q_reference = np.quantile(
        projected_reference, probabilities, axis=0, method="linear"
    )
    return float(np.mean(np.abs(q_generated - q_reference)))


def training_quantile_rmse(
    generated: np.ndarray, reference: np.ndarray, directions: np.ndarray
) -> float:
    knots = min(128, len(generated), len(reference))
    probabilities = (np.arange(knots) + 0.5) / knots
    generated_table = projected_quantile_table(
        generated, directions, probabilities, method="linear"
    )
    reference_table = projected_quantile_table(
        reference, directions, probabilities, method="linear"
    )
    return float(np.sqrt(np.mean((generated_table - reference_table) ** 2)))


def covariance_effective_rank(values: np.ndarray) -> float:
    covariance = np.cov(values, rowvar=False)
    if np.ndim(covariance) == 0:
        eigenvalues = np.asarray([float(covariance)])
    else:
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    denominator = float(np.sum(eigenvalues**2))
    return float(np.sum(eigenvalues)) ** 2 / denominator if denominator > 0.0 else 0.0


def mode_diagnostics(
    values: np.ndarray, data: NormalizedTarget
) -> tuple[float, float, float]:
    if data.normalized_centers is None or data.weights is None:
        return float("nan"), float("nan"), float("nan")
    distances = np.linalg.norm(
        values[:, None, :] - data.normalized_centers[None, :, :], axis=2
    )
    labels = np.argmin(distances, axis=1)
    mass = np.bincount(labels, minlength=len(data.weights)).astype(float) / len(values)
    coverage = float(np.mean(mass >= 0.5 * data.weights))
    error = float(np.sum(np.abs(mass - data.weights)))
    rare_error = (
        float(abs(mass[-1] - data.weights[-1]))
        if data.weights[-1] <= 0.10
        else float("nan")
    )
    return coverage, error, rare_error


def evaluate_output(
    values: np.ndarray,
    data: NormalizedTarget,
    training_directions: np.ndarray,
    heldout_directions: np.ndarray,
) -> dict[str, float]:
    if not np.all(np.isfinite(values)):
        return {
            key: float("inf")
            for key in (
                "ed2",
                "heldout_sw1",
                "training_quantile_rmse",
                "output_rms",
                "covariance_effective_rank",
                "mode_coverage",
                "mode_mass_error",
                "rare_mass_error",
            )
        }
    coverage, mass_error, rare_error = mode_diagnostics(values, data)
    return {
        "ed2": max(0.0, float(energy_distance2(values, data.reference))),
        "heldout_sw1": heldout_sw1(values, data.reference, heldout_directions),
        "training_quantile_rmse": training_quantile_rmse(
            values, data.reference, training_directions
        ),
        "output_rms": float(np.sqrt(np.mean(values**2))),
        "covariance_effective_rank": covariance_effective_rank(values),
        "mode_coverage": coverage,
        "mode_mass_error": mass_error,
        "rare_mass_error": rare_error,
    }


def safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def phase_invariants(registry: dict, log) -> None:
    from neural_pooled_rank_tests import (
        atoms_gaps_and_rare_modes_test,
        exact_atlas_test,
        finite_difference_and_psqt_test,
        frame_diagnostic_test,
        kll_no_compaction_test,
        rsr_gradient_test,
        stochastic_replay_test,
    )

    for test in (
        exact_atlas_test,
        kll_no_compaction_test,
        finite_difference_and_psqt_test,
        rsr_gradient_test,
        stochastic_replay_test,
        atoms_gaps_and_rare_modes_test,
        frame_diagnostic_test,
    ):
        test()
    log("  [PASS] complete Phase-0 regression suite")

    rng = np.random.default_rng(991)
    for mask in (False, True):
        query = rng.normal(size=(11, 4))
        positive = rng.normal(size=(13, 4))
        expected = drift_paper(query, positive, PAPER_TAU, mask)
        observed = torch_paper_field(
            torch.tensor(query, dtype=torch.float64),
            torch.tensor(positive, dtype=torch.float64),
            mask=mask,
        ).numpy()
        error = float(np.max(np.abs(expected - observed)))
        if error > 2e-14:
            raise AssertionError(f"Torch paper-field mismatch {error}")
    matched = rng.normal(size=(9, 3))
    cancellation = torch_paper_field(
        torch.tensor(matched, dtype=torch.float64),
        torch.tensor(matched, dtype=torch.float64),
        mask=False,
    )
    if float(torch.max(torch.abs(cancellation))) > 2e-14:
        raise AssertionError("Torch matched paper field did not cancel")
    log("  [PASS] Torch paper field agrees with repository NumPy port")

    worst_training = 0.0
    worst_heldout = 0.0
    for target in registry["targets"]:
        dimension = int(target["dimension"])
        training = frame_diagnostics(target["training_directions"])
        heldout = frame_diagnostics(target["heldout_directions"])
        if training.rank != dimension or heldout.rank != dimension:
            raise AssertionError("registered direction bank is rank deficient")
        worst_training = max(worst_training, training.spectral_tightness_error)
        worst_heldout = max(worst_heldout, heldout.spectral_tightness_error)
    if worst_training > 2e-12 or worst_heldout > 2e-12:
        raise AssertionError("registered orthogonal blocks are not tight")
    log("  [PASS] all registered training/held-out frames are tight")

    first = Generator(4, "concentrated", 137)
    second = Generator(4, "concentrated", 137)
    if any(
        not torch.equal(a, b)
        for a, b in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    ):
        raise AssertionError("paired model initialization is not exact")
    log("  [PASS] paired initial generator states are bitwise equal")


def summarize(rows: list[dict[str, Any]]) -> dict:
    result: dict[str, Any] = {"by_arm": {}, "by_dimension": {}}
    for arm in (*ARMS, CEILING):
        selected = [row for row in rows if row["arm"] == arm]
        result["by_arm"][arm] = {
            "rows": len(selected),
            "median_ed2": float(np.median([row["ed2"] for row in selected])),
            "median_heldout_sw1": float(
                np.median([row["heldout_sw1"] for row in selected])
            ),
            "divergences": int(sum(row["diverged"] for row in selected)),
            "median_wall_seconds": float(
                np.median([row["wall_seconds"] for row in selected])
            ),
        }
    for dimension in (2, 4, 8, 16):
        result["by_dimension"][str(dimension)] = {}
        for arm in ARMS:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and row["dimension"] == dimension
            ]
            result["by_dimension"][str(dimension)][arm] = {
                "median_ed2": float(np.median([row["ed2"] for row in selected])),
                "median_heldout_sw1": float(
                    np.median([row["heldout_sw1"] for row in selected])
                ),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    profile.validate()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    registry, registry_digest = load_registry()
    targets = [
        target
        for target in registry["targets"]
        if not profile.smoke_only or target["smoke"]
    ]
    rows_per_cell = len(ARMS) + 1
    expected_rows = (
        len(targets)
        * profile.replications
        * len(profile.initializations)
        * rows_per_cell
    )
    run = RunArtifact(profile, registry_digest, expected_rows)
    run.log("Neural pooled-rank Phase-1 development")
    run.log(f"  registry {registry_digest}")
    run.log(
        f"  {len(targets)} targets x {profile.replications} reps x "
        f"{len(profile.initializations)} inits x {rows_per_cell} rows"
    )
    phase_invariants(registry, run.log)

    rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for target in targets:
        target_started = time.perf_counter()
        dimension = int(target["dimension"])
        training_directions = np.asarray(target["training_directions"], dtype=float)
        heldout_directions = np.asarray(target["heldout_directions"], dtype=float)
        for replication in range(profile.replications):
            data = normalized_target_data(target, replication, profile)
            exact_started = time.perf_counter()
            exact_atlas = exact_target_atlas(
                data.pool, training_directions, profile.atlas_knots
            )
            exact_build_seconds = time.perf_counter() - exact_started
            kll_started = time.perf_counter()
            kll_atlas = apache_kll_target_atlas(
                data.pool, training_directions, profile.atlas_knots, k=128
            )
            kll_build_seconds = time.perf_counter() - kll_started
            restored = ApacheKLLProjectedAccumulator.from_serialized(
                dimension,
                training_directions,
                profile.atlas_knots,
                list(kll_atlas.sketch_payloads or ()),
                k=128,
            )
            if not np.array_equal(restored.table(), kll_atlas.quantiles):
                raise AssertionError("serialized KLL atlas did not replay")
            run.record_kll(target["name"], replication, kll_atlas)
            reference_key = safe_key(f"{target['name']}__r{replication}")
            references[reference_key] = data.reference.astype(np.float32)

            for initialization in profile.initializations:
                model_seed = mixed_seed(
                    int(target["seeds"]["model"]),
                    replication,
                    11 if initialization == "concentrated" else 12,
                )
                base_model = Generator(dimension, initialization, model_seed).to(
                    dtype=torch.float32
                )
                latent_seed = mixed_seed(
                    int(target["seeds"]["training_latent"]), replication, 20
                )
                latent_rng = np.random.default_rng(latent_seed)
                latent_bank = torch.tensor(
                    latent_rng.normal(
                        size=(profile.generator_budget, base_model.latent_dimension)
                    ),
                    dtype=torch.float32,
                )
                evaluation_seed = mixed_seed(
                    int(target["seeds"]["evaluation_latent"]), replication, 21
                )
                evaluation_latent = torch.tensor(
                    np.random.default_rng(evaluation_seed).normal(
                        size=(profile.evaluation_samples, base_model.latent_dimension)
                    ),
                    dtype=torch.float32,
                )

                for arm in ARMS:
                    model, ledger = train_neural_arm(
                        arm,
                        base_model,
                        latent_bank,
                        data.pool,
                        training_directions,
                        exact_atlas,
                        kll_atlas,
                        profile,
                    )
                    with torch.no_grad():
                        final = model(evaluation_latent).numpy()
                    metrics = evaluate_output(
                        final, data, training_directions, heldout_directions
                    )
                    build_seconds = (
                        exact_build_seconds
                        if arm == "exact-atlas-rsr"
                        else kll_build_seconds
                        if arm in ("kll-atlas-rsr", "kll-small-rsr", "kll-paper-hybrid")
                        else 0.0
                    )
                    row = {
                        "arm": arm,
                        "kind": "neural",
                        "target": target["name"],
                        "family": target["family"],
                        "dimension": dimension,
                        "replication": replication,
                        "initialization": initialization,
                        **metrics,
                        **asdict(ledger),
                        "atlas_build_seconds": build_seconds,
                        "evaluation_generator_examples": profile.evaluation_samples,
                        "model_parameters": sum(
                            parameter.numel() for parameter in model.parameters()
                        ),
                    }
                    rows.append(row)
                    key = safe_key(
                        f"{arm}__{target['name']}__r{replication}__{initialization}"
                    )
                    outputs[key] = final.astype(np.float32)

                with torch.no_grad():
                    initial_particles = base_model(
                        latent_bank[: profile.large_population]
                    ).numpy()
                ceiling_started = time.perf_counter()
                ceiling, sort_proxy, projection_count = reconstruct_from_quantile_table(
                    initial_particles,
                    training_directions,
                    exact_atlas.quantiles,
                    steps=profile.free_particle_sweeps,
                    step_size=0.5,
                )
                ceiling_wall = time.perf_counter() - ceiling_started
                ceiling_metrics = evaluate_output(
                    ceiling, data, training_directions, heldout_directions
                )
                ceiling_ledger = TrainingLedger(
                    completed_updates=profile.free_particle_sweeps,
                    unique_latent_samples=profile.large_population,
                    generator_example_evals_training=profile.large_population,
                    generator_forward_calls_training=1,
                    unique_target_observations=len(data.pool),
                    target_example_accesses=len(data.pool),
                    projection_scalar_products=(
                        len(data.pool) * TRAIN_DIRECTIONS + projection_count
                    ),
                    sort_work=(
                        TRAIN_DIRECTIONS * len(data.pool) * math.log2(len(data.pool))
                        + sort_proxy
                    ),
                    atlas_bytes=atlas_bytes(exact_atlas),
                    wall_seconds=ceiling_wall,
                )
                rows.append(
                    {
                        "arm": CEILING,
                        "kind": "free-particle-diagnostic",
                        "target": target["name"],
                        "family": target["family"],
                        "dimension": dimension,
                        "replication": replication,
                        "initialization": initialization,
                        **ceiling_metrics,
                        **asdict(ceiling_ledger),
                        "atlas_build_seconds": exact_build_seconds,
                        "evaluation_generator_examples": 0,
                        "model_parameters": 0,
                    }
                )
                ceiling_key = safe_key(
                    f"{CEILING}__{target['name']}__r{replication}__{initialization}"
                )
                outputs[ceiling_key] = ceiling.astype(np.float32)
        run.log(
            f"  completed {target['name']} in "
            f"{time.perf_counter() - target_started:.2f}s"
        )

    for row in rows:
        if row["kind"] == "neural" and (
            row["generator_example_evals_training"] != profile.generator_budget
        ):
            raise AssertionError("a neural arm missed the generator budget")
        if row["kind"] == "neural" and row["diverged"]:
            run.log(f"  WARNING divergence: {row['arm']} {row['target']}")
    summary = summarize(rows)
    for arm in (*ARMS, CEILING):
        values = summary["by_arm"][arm]
        run.log(
            f"  {arm:28s} ED2={values['median_ed2']:.6f} "
            f"heldoutSW1={values['median_heldout_sw1']:.6f} "
            f"div={values['divergences']}"
        )
    run.finish(rows, outputs, references, summary)
    run.log(f"artifact {run.directory}")
    print(f"ARTIFACT={run.directory}")


if __name__ == "__main__":
    main()
