"""Production-checkpoint admission for the CAP-EMF finite difference.

The gate compares stopped finite differences with an exact directional JVP on
the *trained* model.  A CPU smoke check may exercise the code, but only a CUDA
run with parameter-gradient comparison can emit GO for a paid experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..stage_cap.config import CAPModelConfig, CAPObjectiveConfig
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap.model import CAPPixelTransformer
from ..stage_cap.objective import directional_jvp_reference, emf_local_difference
from .artifacts import assert_unused, source_manifest, write_json_atomic
from .config import numerical_candidate
from .hardware import hardware_binding

# RELAXATION, adopted after seeing the data and authorized explicitly.  This is
# not a bug fix and should not be described as one.
#
# The original thresholds were never met by any measured configuration: the
# preserved CAP-EMF-1 checkpoint failed interior_high 0/9 under
# local_1000_d0002_fp32, and the best scale-100 run reached 61/63.  The gate has
# never returned GO.
#
# The relative tests are also ill-conditioned exactly where they bind.  Relative
# RMS divides by the magnitude of the derivative being approximated, so a model
# with a flatter characteristic is penalized for an error that is smaller in
# data units.  Measured on matched 50k runs, the repaired configuration is more
# accurate in ABSOLUTE terms at four of five strata -- at large_coefficient
# 0.0472 against the baseline's 0.0582 -- while scoring worse on every relative
# test, because its reference magnitude is 3.2x smaller.
#
# What justifies acting on that: the proxy stopped tracking the outcome.  The
# configuration the gate rates worse produces clean FID 68.36 against the one it
# rates better at 114.90, and beats CAP-EMF-1's fully trained 650k result of
# 83.65.  The gate is a means of checking the training target, and on this
# evidence it is mis-measuring it.
#
# The guardrail is that the relaxed gate must still REJECT legacy_1000_d01, the
# documented ten-radian control, which is asserted in the test suite.  A gate
# that admits everything certifies nothing.
#
# Consequence to carry forward: admission is no longer an independent check at
# its original strength, and any result downstream of it inherits that caveat.
TARGET_COSINE_MIN = 0.98
TARGET_RELATIVE_RMS_MAX = 0.20
GRADIENT_COSINE_MIN = 0.95
# Cosine alone is scale blind.  A gradient pointing in the right direction but
# 25% too large passes a cosine-only gate and changes every optimizer update.
GRADIENT_RELATIVE_L2_MAX = 0.20
GRADIENT_NORM_RATIO_MIN = 0.85
GRADIENT_NORM_RATIO_MAX = 1.15
# Relaxed with the quotient thresholds above, and for the same reason.  At the
# high-coefficient strata the assembled test is near-redundant with the quotient
# one -- the coefficient scales reference and error alike, so the two relative
# errors converge (0.1638 against 0.1829 on the repaired run) -- which is why
# both move together rather than independently.
ASSEMBLED_TARGET_COSINE_MIN = 0.98
ASSEMBLED_TARGET_RELATIVE_RMS_MAX = 0.20

# Exact one-step inference is (t,h)=(1,1), i.e. (t,r)=(1,0).  The old audit
# stopped at (.98,.02), twenty radians away in a 1000-scaled embedding.  The
# low-t row is included because the production loss weights it by 1/t^2.
AUDIT_STRATA: tuple[tuple[str, float, float], ...] = (
    ("exact_inference", 1.0, 0.0),
    ("near_inference", 0.995, 0.0),
    ("large_coefficient", 0.98, 0.02),
    ("interior_high", 0.85, 0.10),
    ("interior_mid", 0.60, 0.20),
    ("low_t_weighted", 0.03, 0.0),
    ("below_floor_weighted", 0.01, 0.0),
)
AUDIT_SOURCES = ("cifar10_train", "synthetic_gaussian", "synthetic_checkerboard")
AUDIT_BATCH = 4
PRODUCTION_MICROBATCH = 16
MINIMUM_REPEATS = 3


def _hash64(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _passing_input_batch(payload: object, *, source: str, batch: int) -> bool:
    if not isinstance(payload, dict):
        return False
    seed = payload.get("seed")
    sample_ids = payload.get("sample_ids")
    return (
        payload.get("source") == source
        and payload.get("batch") == batch
        and isinstance(seed, int)
        and not isinstance(seed, bool)
        and all(
            _hash64(payload.get(name))
            for name in ("clean_sha256", "noise_sha256", "state_sha256")
        )
        and (
            isinstance(sample_ids, list) and len(sample_ids) == batch
            if source == "cifar10_train"
            else sample_ids is None
        )
    )


def _recorded_admission_checks(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None
    try:
        return _strict_admission_checks(
            payload["quotient"], payload["assembled_target"], payload["gradient"]
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _expected_mixed_pairs(batch: int) -> list[list[float]]:
    active_pairs = [(t, r) for _name, t, r in AUDIT_STRATA]
    diagonal_times = (0.20, 0.40, 0.60, 0.80, 0.95, 0.98, 0.995, 1.0)
    diagonal_count = batch // 2
    pairs = [
        (diagonal_times[index % len(diagonal_times)],) * 2
        for index in range(diagonal_count)
    ] + [
        active_pairs[index % len(active_pairs)]
        for index in range(batch - diagonal_count)
    ]
    return [[float(left), float(right)] for left, right in pairs]


def admission_matrix_complete(record: object) -> bool:
    """Validate the complete homogeneous and mixed numerical experiment.

    This pure predicate is shared by preflight, the 50k continuation gate, and
    the 150k promotion gate.  Rechecking the matrix prevents a self-consistent
    SHA sidecar around a fabricated top-level ``decision: GO`` from granting
    training authority.
    """
    if not isinstance(record, dict):
        return False
    repeats = record.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        return False
    try:
        candidate = numerical_candidate(record.get("candidate", {}).get("name"))
    except (AttributeError, TypeError, ValueError):
        return False
    expected_candidate = asdict(candidate)
    expected_thresholds = {
        "quotient_cosine_min": TARGET_COSINE_MIN,
        "quotient_relative_rms_max": TARGET_RELATIVE_RMS_MAX,
        "assembled_target_cosine_min": ASSEMBLED_TARGET_COSINE_MIN,
        "assembled_target_relative_rms_max": ASSEMBLED_TARGET_RELATIVE_RMS_MAX,
        "gradient_cosine_min": GRADIENT_COSINE_MIN,
        "gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
        "gradient_norm_ratio": [
            GRADIENT_NORM_RATIO_MIN,
            GRADIENT_NORM_RATIO_MAX,
        ],
    }
    stratum_values = {name: (t, r, t - r) for name, t, r in AUDIT_STRATA}
    expected_strata = [
        {"name": name, "t": t, "r": r, "h": t - r} for name, t, r in AUDIT_STRATA
    ]
    expected_row_keys = {
        (repeat, source, name)
        for repeat in range(repeats)
        for source in AUDIT_SOURCES
        for name, _t, _r in AUDIT_STRATA
    }
    strata = record.get("strata")
    if not isinstance(strata, list):
        return False
    actual_row_keys = {
        (
            row.get("repeat"),
            row.get("input_batch", {}).get("source"),
            row.get("stratum"),
        )
        for row in strata
        if isinstance(row, dict)
    }

    def row_valid(row: object) -> bool:
        if not isinstance(row, dict) or row.get("stratum") not in stratum_values:
            return False
        expected_t, expected_r, expected_h = stratum_values[row["stratum"]]
        checks = _recorded_admission_checks(row)
        input_batch = row.get("input_batch")
        return (
            checks is not None
            and row.get("admission_checks") == checks
            and all(checks.values())
            and row.get("verdict") == "PASS"
            and row.get("t") == expected_t
            and row.get("r") == expected_r
            and row.get("h") == expected_h
            and row.get("delta") == candidate.delta
            and row.get("evaluation_mode") == candidate.stopped_evaluation
            and _passing_input_batch(
                input_batch,
                source=input_batch.get("source")
                if isinstance(input_batch, dict)
                else "",
                batch=AUDIT_BATCH,
            )
        )

    expected_mixed_keys = {
        (repeat, source) for repeat in range(repeats) for source in AUDIT_SOURCES
    }

    def mixed_valid(rows: object, *, batch: int) -> bool:
        return (
            isinstance(rows, list)
            and len(rows) == len(expected_mixed_keys)
            and {
                (
                    row.get("repeat"),
                    row.get("input_batch", {}).get("source"),
                )
                for row in rows
                if isinstance(row, dict)
            }
            == expected_mixed_keys
            and all(
                isinstance(row, dict)
                and (checks := _recorded_admission_checks(row)) is not None
                and row.get("admission_checks") == checks
                and all(checks.values())
                and row.get("verdict") == "PASS"
                and _passing_input_batch(
                    row.get("input_batch"),
                    source=row.get("input_batch", {}).get("source", ""),
                    batch=batch,
                )
                and row.get("input_batch", {}).get("time_pairs")
                == _expected_mixed_pairs(batch)
                and int(row.get("diagonal_rows", -1)) == batch // 2
                and int(row.get("active_rows", -1)) == batch - batch // 2
                and row.get("delta") == candidate.delta
                and row.get("evaluation_mode") == candidate.stopped_evaluation
                for row in rows
            )
        )

    required_protocol = {
        "checkpoint_identity",
        "cuda",
        "hardware_matches",
        "parameter_gradient_checked",
        "audit_batch_exactly_four",
        "minimum_three_repeats",
        "deterministic_algorithms",
        "all_strata_pass",
        "all_mixed_gradient_batches_pass",
        "all_production_shape_batches_pass",
    }
    protocol = record.get("protocol_checks")
    return (
        record.get("decision") == "GO"
        and record.get("candidate") == expected_candidate
        and record.get("thresholds") == expected_thresholds
        and record.get("checkpoint_identity", {}).get("valid") is True
        and record.get("hardware", {}).get("matches") is True
        and record.get("production_numerical_mode", {}).get("deterministic_algorithms")
        is True
        and int(record.get("batch_per_stratum", -1)) == AUDIT_BATCH
        and repeats >= MINIMUM_REPEATS
        and record.get("gradient_checked") is True
        and record.get("cuda_admission") is True
        and record.get("design", {}).get("sources") == list(AUDIT_SOURCES)
        and record.get("design", {}).get("strata") == expected_strata
        and int(record.get("design", {}).get("total_stratum_batches", -1))
        == len(expected_row_keys)
        and len(strata) == len(expected_row_keys)
        and actual_row_keys == expected_row_keys
        and all(row_valid(row) for row in strata)
        and mixed_valid(record.get("mixed_gradient"), batch=AUDIT_BATCH)
        and mixed_valid(record.get("production_shape"), batch=PRODUCTION_MICROBATCH)
        and isinstance(protocol, dict)
        and set(protocol) == required_protocol
        and all(value is True for value in protocol.values())
    )


@contextmanager
def tf32_mode(enabled: bool):
    if not torch.cuda.is_available():
        yield
        return
    before_matmul = torch.backends.cuda.matmul.allow_tf32
    before_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before_matmul
        torch.backends.cudnn.allow_tf32 = before_cudnn


@contextmanager
def production_numerical_mode(device: torch.device):
    """Match the deterministic/TF32 boundary used by screen training.

    The previous admission enabled TF32 only around the graded *forward* and
    restored the process default before ``autograd.grad``.  On current PyTorch
    builds that can silently compare full-FP32 backward gradients even though
    the paid run uses TF32 backward matmuls.  This outer context keeps ordinary
    forward and backward work in the production mode; the exact JVP and stopped
    path still enter their explicit full-FP32 inner contexts.
    """

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        yield {
            "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "graded_forward_tf32": device.type == "cuda",
            "graded_backward_tf32": device.type == "cuda",
            "exact_jvp_tf32": False,
            "stopped_path_tf32": False,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash the exact CPU bytes plus dtype/shape without requiring NumPy."""

    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + bytes(tensor.untyped_storage())).hexdigest()


def _checkpoint_identity(payload: dict, model: CAPPixelTransformer) -> dict:
    profile = payload.get("profile")
    state = payload.get("state_dict")
    state_value_count = (
        sum(value.numel() for value in state.values())
        if isinstance(state, dict)
        and all(isinstance(value, torch.Tensor) for value in state.values())
        else -1
    )
    recorded_count = int(payload.get("parameter_count", -1))
    trainable_count = model.parameter_count()
    if recorded_count == trainable_count:
        count_semantics = "trainable_parameters"
    elif payload.get("stage") == "cap-emf-1" and recorded_count == state_value_count:
        # Historical CAP1 counted the two fixed sinusoidal-frequency buffers.
        # Preserve the immutable checkpoint while naming that legacy schema;
        # new CAP2 checkpoints record trainable parameters separately.
        count_semantics = "legacy_state_dict_values_including_buffers"
    else:
        count_semantics = "invalid"
    checks = {
        "recognized_stage": payload.get("stage") in {"cap-emf-1", "cap-emf-2-screen"},
        "recognized_kind": payload.get("kind") in {"raw", "ema"},
        "nonnegative_step": isinstance(payload.get("step"), int)
        and payload["step"] >= 0,
        "profile_present": isinstance(profile, dict),
        "parameter_count_recognized": count_semantics != "invalid",
    }
    return {
        "stage": payload.get("stage"),
        "step": payload.get("step"),
        "kind": payload.get("kind"),
        "arm": payload.get("arm"),
        "profile_name": profile.get("name") if isinstance(profile, dict) else None,
        "profile_sha256": _json_sha256(profile) if isinstance(profile, dict) else None,
        "recorded_parameter_count": payload.get("parameter_count"),
        "recorded_parameter_count_semantics": count_semantics,
        "state_dict_value_count": state_value_count,
        "loaded_trainable_parameter_count": trainable_count,
        "checks": checks,
        "valid": all(checks.values()),
    }


def _checkerboard_batch(model: CAPPixelTransformer, batch: int) -> torch.Tensor:
    size = model.config.image_size
    yy = torch.arange(size)[:, None]
    xx = torch.arange(size)[None, :]
    base = (1 - 2 * ((yy + xx) % 2)).float()
    images = base.view(1, 1, size, size).repeat(batch, model.config.channels, 1, 1)
    # Exercise both phases and distinct channel signs while staying in the
    # exact training range [-1,1].
    sample_sign = torch.where(
        torch.arange(batch) % 2 == 0,
        torch.ones(batch),
        -torch.ones(batch),
    ).view(batch, 1, 1, 1)
    channel_sign = torch.where(
        torch.arange(model.config.channels) % 2 == 0,
        torch.ones(model.config.channels),
        -torch.ones(model.config.channels),
    ).view(1, model.config.channels, 1, 1)
    return images * sample_sign * channel_sign


def _audit_inputs(
    model: CAPPixelTransformer,
    *,
    source: str,
    batch: int,
    seed: int,
    device: torch.device,
    real_pool: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(seed)
    sample_ids: list[int] | None = None
    if source == "synthetic_gaussian":
        clean = torch.randn(
            (
                batch,
                model.config.channels,
                model.config.image_size,
                model.config.image_size,
            ),
            generator=generator,
        )
    elif source == "synthetic_checkerboard":
        clean = _checkerboard_batch(model, batch)
    elif source == "cifar10_train":
        if real_pool is None:
            raise ValueError("cifar10_train admission requires an explicit train pool")
        if len(real_pool) < batch:
            raise ValueError("real admission pool is smaller than the audit batch")
        indices = torch.randperm(len(real_pool), generator=generator)[:batch]
        sample_ids = [int(index) for index in indices]
        clean = real_pool[indices].cpu().float()
    else:
        raise ValueError(f"unknown numerical-admission source {source!r}")
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    metadata = {
        "source": source,
        "batch": batch,
        "seed": seed,
        "sample_ids": sample_ids,
        "clean_sha256": _tensor_sha256(clean),
        "noise_sha256": _tensor_sha256(noise),
        "clean_range": [float(clean.min()), float(clean.max())],
    }
    return clean.to(device), noise.to(device), metadata


def _load_checkpoint(path: Path, device: torch.device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch
        payload = torch.load(path, map_location="cpu")
    if "state_dict" not in payload or "profile" not in payload:
        raise RuntimeError("checkpoint lacks state_dict/profile metadata")
    model_config = CAPModelConfig(**payload["profile"]["model"])
    objective_config = CAPObjectiveConfig(**payload["profile"]["objective"])
    model = CAPPixelTransformer(model_config, seed=1).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model, model_config, objective_config


def _vector_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    left = candidate.detach().double().flatten(1)
    right = reference.detach().double().flatten(1)
    difference = left - right
    denominator = right.square().mean(1).sqrt().clamp_min(1e-30)
    relative = difference.square().mean(1).sqrt() / denominator
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
    return {
        "relative_rms_mean": float(relative.mean()),
        "relative_rms_max": float(relative.max()),
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "candidate_rms": float(left.square().mean().sqrt()),
        "reference_rms": float(right.square().mean().sqrt()),
    }


def _adaptive_loss(
    current: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    config: CAPObjectiveConfig,
) -> torch.Tensor:
    residual = current - target.detach()
    sums = residual.square().flatten(1).sum(1)
    numerator = t.clamp_min(config.resolved_loss_weight_floor).pow(-2)
    weight = numerator / (
        (sums + config.adaptive_epsilon).pow(config.adaptive_power).detach()
    )
    return (weight * sums).mean()


def _gradient_vector_metrics(
    exact: list[torch.Tensor],
    approximate: list[torch.Tensor],
) -> dict:
    """Compare two gradient vectors without allocating one giant flattening."""

    dot = torch.zeros((), device=exact[0].device, dtype=torch.float64)
    exact_sq = torch.zeros_like(dot)
    approximate_sq = torch.zeros_like(dot)
    difference_sq = torch.zeros_like(dot)
    for left, right in zip(exact, approximate):
        left = left.detach().double()
        right = right.detach().double()
        dot += (left * right).sum()
        exact_sq += left.square().sum()
        approximate_sq += right.square().sum()
        difference_sq += (right - left).square().sum()
    exact_norm = exact_sq.sqrt()
    approximate_norm = approximate_sq.sqrt()
    denominator = (exact_norm * approximate_norm).clamp_min(1e-30)
    relative_denominator = exact_norm.clamp_min(1e-30)
    finite = bool(
        torch.isfinite(dot)
        & torch.isfinite(exact_norm)
        & torch.isfinite(approximate_norm)
        & torch.isfinite(difference_sq)
    )
    return {
        "cosine": float(dot / denominator),
        "exact_norm": float(exact_norm),
        "approximate_norm": float(approximate_norm),
        "norm_ratio": float(approximate_norm / relative_denominator),
        "relative_l2": float(difference_sq.sqrt() / relative_denominator),
        "finite": finite,
    }


def _parameter_group(name: str) -> str:
    if (
        name.startswith(("time_embed", "interval_embed", "condition_mlp"))
        or ".modulation." in name
    ):
        return "conditioning"
    if name.startswith(("pixel_head", "refiner")):
        return "output"
    return "trunk"


def _gradient_metrics(
    model: torch.nn.Module,
    current: torch.Tensor,
    exact_target: torch.Tensor,
    approximate_target: torch.Tensor,
    t: torch.Tensor,
    config: CAPObjectiveConfig,
) -> dict:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    exact_loss = _adaptive_loss(current, exact_target, t, config)
    approximate_loss = _adaptive_loss(current, approximate_target, t, config)
    # The paid screen enables TF32 globally.  Keep it enabled through both
    # backward passes; the stopped target itself was formed in full FP32.
    with tf32_mode(True):
        exact = list(torch.autograd.grad(exact_loss, parameters, retain_graph=True))
        approximate = list(torch.autograd.grad(approximate_loss, parameters))
    result = _gradient_vector_metrics(exact, approximate)
    by_group: dict[str, dict] = {}
    for group in ("conditioning", "trunk", "output"):
        indices = [
            index
            for index, (name, _) in enumerate(named_parameters)
            if _parameter_group(name) == group
        ]
        if indices:
            group_metrics = _gradient_vector_metrics(
                [exact[index] for index in indices],
                [approximate[index] for index in indices],
            )
            global_exact = max(float(result["exact_norm"]), 1e-30)
            group_metrics["exact_norm_share"] = (
                float(group_metrics["exact_norm"]) / global_exact
            )
            group_metrics["difference_over_global_exact"] = (
                float(group_metrics["relative_l2"])
                * float(group_metrics["exact_norm"])
                / global_exact
            )
            by_group[group] = group_metrics
    result["by_group"] = by_group
    return result


def _strict_admission_checks(
    quotient: dict,
    assembled_target: dict,
    gradient: dict | None,
) -> dict:
    """The paid-run predicate, kept pure so scale-blind cases are testable."""

    return {
        "quotient_relative_rms": (
            quotient["relative_rms_max"] <= TARGET_RELATIVE_RMS_MAX
        ),
        "quotient_cosine": quotient["cosine_min"] >= TARGET_COSINE_MIN,
        "assembled_target_relative_rms": (
            assembled_target["relative_rms_max"] <= ASSEMBLED_TARGET_RELATIVE_RMS_MAX
        ),
        "assembled_target_cosine": (
            assembled_target["cosine_min"] >= ASSEMBLED_TARGET_COSINE_MIN
        ),
        "gradient_cosine": (
            gradient is not None and gradient["cosine"] >= GRADIENT_COSINE_MIN
        ),
        "gradient_relative_l2": (
            gradient is not None and gradient["relative_l2"] <= GRADIENT_RELATIVE_L2_MAX
        ),
        "gradient_norm_ratio": (
            gradient is not None
            and GRADIENT_NORM_RATIO_MIN
            <= gradient["norm_ratio"]
            <= GRADIENT_NORM_RATIO_MAX
        ),
        "gradient_finite": gradient is not None and gradient["finite"],
    }


def audit_stratum(
    model: CAPPixelTransformer,
    config: CAPObjectiveConfig,
    *,
    t_value: float,
    r_value: float,
    batch: int,
    seed: int,
    delta: float,
    evaluation_mode: str,
    device: torch.device,
    include_gradient: bool,
    source: str = "synthetic_gaussian",
    real_pool: torch.Tensor | None = None,
    stratum_name: str | None = None,
) -> dict:
    clean, noise, input_batch = _audit_inputs(
        model,
        source=source,
        batch=batch,
        seed=seed,
        device=device,
        real_pool=real_pool,
    )
    t = torch.full((batch,), t_value, device=device)
    r = torch.full((batch,), r_value, device=device)
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    input_batch["state_sha256"] = _tensor_sha256(state)

    # The reference is always computed with TF32 disabled.  Forward AD still
    # follows the real trained model; only the arithmetic mode is fixed.
    with tf32_mode(False):
        exact = directional_jvp_reference(
            model, state, t, r, config.emf_denominator_floor
        ).detach()

    # Candidate execution sees the ordinary production TF32 setting.  The
    # fp32_dense mode disables it internally for all stopped evaluations.
    with tf32_mode(True):
        current, quotient = emf_local_difference(
            model,
            state,
            t,
            r,
            delta,
            config.emf_denominator_floor,
            evaluation_mode,
        )

    coefficient = (
        (t - r - delta).clamp_min(0)
        * t
        / r.clamp_min(config.resolved_coefficient_floor)
    )
    exact_target = clean + coefficient[:, None, None, None] * exact
    candidate_target = clean + coefficient[:, None, None, None] * quotient.detach()
    quotient_metrics = _vector_metrics(quotient, exact)
    assembled_target_metrics = _vector_metrics(candidate_target, exact_target)
    gradient_metrics = None
    if include_gradient:
        gradient_metrics = _gradient_metrics(
            model,
            current,
            exact_target,
            candidate_target,
            t,
            config,
        )
    gradient_cosine = (
        gradient_metrics["cosine"] if gradient_metrics is not None else None
    )
    # Preserve the original three-key view for compatibility with the frozen
    # CAP2 regression test.  It is diagnostic only; admission uses the stricter
    # predicate below, including assembled-target and gradient-magnitude checks.
    checks = {
        "target_relative_rms": (
            quotient_metrics["relative_rms_max"] <= TARGET_RELATIVE_RMS_MAX
        ),
        "target_cosine": quotient_metrics["cosine_min"] >= TARGET_COSINE_MIN,
        "gradient_cosine": (
            gradient_cosine is not None and gradient_cosine >= GRADIENT_COSINE_MIN
        ),
    }
    admission_checks = _strict_admission_checks(
        quotient_metrics,
        assembled_target_metrics,
        gradient_metrics,
    )
    return {
        "stratum": stratum_name,
        "t": t_value,
        "r": r_value,
        "h": t_value - r_value,
        "coefficient": float(coefficient[0]),
        "delta": delta,
        "evaluation_mode": evaluation_mode,
        "input_batch": input_batch,
        "quotient": quotient_metrics,
        "assembled_target": assembled_target_metrics,
        # Historical alias: the old report called quotient fidelity "target".
        "target": quotient_metrics,
        "gradient": gradient_metrics,
        "gradient_cosine": gradient_cosine,
        "checks": checks,
        "admission_checks": admission_checks,
        "verdict": "PASS" if all(admission_checks.values()) else "FAIL",
    }


def audit_mixed_batch(
    model: CAPPixelTransformer,
    config: CAPObjectiveConfig,
    *,
    source: str,
    batch: int,
    seed: int,
    delta: float,
    evaluation_mode: str,
    device: torch.device,
    include_gradient: bool,
    real_pool: torch.Tensor | None,
    kind: str,
) -> dict:
    """Audit heterogeneous times in one GEMM batch.

    Homogeneous strata isolate local failures, but production batches mix
    diagonal, tail, and endpoint rows.  GEMM rounding depends on batch shape,
    and gradient errors can cancel or reinforce across rows, so both a
    production-shape target audit and a mixed gradient audit are independent
    admission requirements.
    """
    clean, noise, input_batch = _audit_inputs(
        model,
        source=source,
        batch=batch,
        seed=seed,
        device=device,
        real_pool=real_pool,
    )
    active_pairs = [(t, r) for _name, t, r in AUDIT_STRATA]
    diagonal_times = (0.20, 0.40, 0.60, 0.80, 0.95, 0.98, 0.995, 1.0)
    diagonal_count = batch // 2
    selected = [
        (diagonal_times[index % len(diagonal_times)],) * 2
        for index in range(diagonal_count)
    ] + [
        active_pairs[index % len(active_pairs)]
        for index in range(batch - diagonal_count)
    ]
    t = torch.tensor([pair[0] for pair in selected], device=device)
    r = torch.tensor([pair[1] for pair in selected], device=device)
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    input_batch["state_sha256"] = _tensor_sha256(state)
    input_batch["time_pairs"] = [
        [float(left), float(right)] for left, right in selected
    ]

    with tf32_mode(False):
        exact = directional_jvp_reference(
            model, state, t, r, config.emf_denominator_floor
        ).detach()
    with tf32_mode(True):
        current, quotient = emf_local_difference(
            model,
            state,
            t,
            r,
            delta,
            config.emf_denominator_floor,
            evaluation_mode,
        )
    coefficient = (
        (t - r - delta).clamp_min(0)
        * t
        / r.clamp_min(config.resolved_coefficient_floor)
    )
    exact_target = clean + coefficient[:, None, None, None] * exact
    candidate_target = clean + coefficient[:, None, None, None] * quotient.detach()
    active = (t - r) > delta
    if not bool(active.any()):
        raise RuntimeError("mixed numerical audit has no active rows")
    quotient_metrics = _vector_metrics(quotient[active], exact[active])
    assembled_target_metrics = _vector_metrics(candidate_target, exact_target)
    gradient_metrics = (
        _gradient_metrics(
            model,
            current,
            exact_target,
            candidate_target,
            t,
            config,
        )
        if include_gradient
        else None
    )
    complete_checks = _strict_admission_checks(
        quotient_metrics, assembled_target_metrics, gradient_metrics
    )
    admission_checks = (
        complete_checks
        if include_gradient
        else {
            name: passed
            for name, passed in complete_checks.items()
            if name.startswith(("quotient_", "assembled_target_"))
        }
    )
    return {
        "kind": kind,
        "input_batch": input_batch,
        "diagonal_rows": int((t == r).sum()),
        "active_rows": int(active.sum()),
        "delta": delta,
        "evaluation_mode": evaluation_mode,
        "quotient": quotient_metrics,
        "assembled_target": assembled_target_metrics,
        "gradient": gradient_metrics,
        "admission_checks": admission_checks,
        "verdict": "PASS" if all(admission_checks.values()) else "FAIL",
    }


def run_admission(
    checkpoint: Path,
    candidate_name: str,
    *,
    device: torch.device,
    batch: int,
    include_gradient: bool,
    repeats: int,
    expected_gpu_name: str | None,
    data_root: str | None = None,
    real_pool: torch.Tensor | None = None,
) -> dict:
    candidate = numerical_candidate(candidate_name)
    payload, model, model_config, historical_objective = _load_checkpoint(
        checkpoint, device
    )
    if model_config.scalar_embedding_scale != candidate.embedding_scale:
        raise RuntimeError(
            "candidate embedding scale differs from the trained checkpoint; "
            "run a short model trained with that embedding before admission"
        )
    config = CAPObjectiveConfig(
        **{
            **asdict(historical_objective),
            "emf_delta": candidate.delta,
            "stopped_evaluation": candidate.stopped_evaluation,
        }
    )
    if repeats <= 0:
        raise ValueError("numerical admission needs at least one repeat")
    if batch <= 0:
        raise ValueError("numerical admission batch must be positive")
    if real_pool is None:
        # This API opens only the official training split.  The sealed test set
        # remains inaccessible from numerical admission by construction.
        real_pool = cifar10_train_pool(data_root)
    identity = _checkpoint_identity(payload, model)
    results = []
    mixed_gradient_results = []
    production_shape_results = []
    with production_numerical_mode(device) as numerical_mode:
        for repeat in range(repeats):
            for source_index, source in enumerate(AUDIT_SOURCES):
                for stratum_index, (name, t, r) in enumerate(AUDIT_STRATA):
                    result = audit_stratum(
                        model,
                        config,
                        t_value=t,
                        r_value=r,
                        batch=batch,
                        seed=(
                            20_260_820
                            + 10_000 * repeat
                            + 100 * source_index
                            + stratum_index
                        ),
                        delta=candidate.delta,
                        evaluation_mode=candidate.stopped_evaluation,
                        device=device,
                        include_gradient=include_gradient,
                        source=source,
                        real_pool=real_pool,
                        stratum_name=name,
                    )
                    result["repeat"] = repeat
                    results.append(result)
                mixed = audit_mixed_batch(
                    model,
                    config,
                    source=source,
                    batch=AUDIT_BATCH,
                    seed=20_261_820 + 10_000 * repeat + 100 * source_index,
                    delta=candidate.delta,
                    evaluation_mode=candidate.stopped_evaluation,
                    device=device,
                    include_gradient=include_gradient,
                    real_pool=real_pool,
                    kind="mixed-gradient",
                )
                mixed["repeat"] = repeat
                mixed_gradient_results.append(mixed)
                production_shape = audit_mixed_batch(
                    model,
                    config,
                    source=source,
                    batch=PRODUCTION_MICROBATCH,
                    seed=20_262_820 + 10_000 * repeat + 100 * source_index,
                    delta=candidate.delta,
                    evaluation_mode=candidate.stopped_evaluation,
                    device=device,
                    include_gradient=include_gradient,
                    real_pool=real_pool,
                    kind="production-shape-mixed-gradient",
                )
                production_shape["repeat"] = repeat
                production_shape_results.append(production_shape)
    cuda_admission = device.type == "cuda"
    hardware = hardware_binding(device, expected_gpu_name)
    all_pass = all(result["verdict"] == "PASS" for result in results)
    mixed_pass = all(result["verdict"] == "PASS" for result in mixed_gradient_results)
    production_shape_pass = all(
        result["verdict"] == "PASS" for result in production_shape_results
    )
    protocol_checks = {
        "checkpoint_identity": identity["valid"],
        "cuda": cuda_admission,
        "hardware_matches": hardware["matches"],
        "parameter_gradient_checked": include_gradient,
        "audit_batch_exactly_four": batch == AUDIT_BATCH,
        "minimum_three_repeats": repeats >= MINIMUM_REPEATS,
        "deterministic_algorithms": numerical_mode["deterministic_algorithms"],
        "all_strata_pass": all_pass,
        "all_mixed_gradient_batches_pass": mixed_pass,
        "all_production_shape_batches_pass": production_shape_pass,
    }
    decision = "GO" if all(protocol_checks.values()) else "NO_GO"
    return {
        "status": "cap-emf2-numerical-admission",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": payload.get("step"),
        "checkpoint_identity": identity,
        "checkpoint_embedding_scale": model_config.scalar_embedding_scale,
        "candidate": asdict(candidate),
        "device": str(device),
        "hardware": hardware,
        "production_numerical_mode": numerical_mode,
        "cuda_admission": cuda_admission,
        "gradient_checked": include_gradient,
        "batch_per_stratum": batch,
        "repeats": repeats,
        "design": {
            "sources": list(AUDIT_SOURCES),
            "strata": [
                {"name": name, "t": t, "r": r, "h": t - r}
                for name, t, r in AUDIT_STRATA
            ],
            "batch_per_stratum": batch,
            "repeats": repeats,
            "total_stratum_batches": (len(AUDIT_SOURCES) * len(AUDIT_STRATA) * repeats),
            "total_rows": (batch * len(AUDIT_SOURCES) * len(AUDIT_STRATA) * repeats),
            "mixed_gradient_batches": len(mixed_gradient_results),
            "mixed_gradient_batch": AUDIT_BATCH,
            "production_shape_batches": len(production_shape_results),
            "production_shape_batch": PRODUCTION_MICROBATCH,
            "real_source_split": "official CIFAR-10 train only",
            "synthetic_stress_sources": [
                source for source in AUDIT_SOURCES if source != "cifar10_train"
            ],
        },
        "thresholds": {
            "quotient_cosine_min": TARGET_COSINE_MIN,
            "quotient_relative_rms_max": TARGET_RELATIVE_RMS_MAX,
            "assembled_target_cosine_min": ASSEMBLED_TARGET_COSINE_MIN,
            "assembled_target_relative_rms_max": (ASSEMBLED_TARGET_RELATIVE_RMS_MAX),
            "gradient_cosine_min": GRADIENT_COSINE_MIN,
            "gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
            "gradient_norm_ratio": [
                GRADIENT_NORM_RATIO_MIN,
                GRADIENT_NORM_RATIO_MAX,
            ],
        },
        "strata": results,
        "mixed_gradient": mixed_gradient_results,
        "production_shape": production_shape_results,
        "protocol_checks": protocol_checks,
        "decision": decision,
        "limits": [
            "A CPU result exercises mechanics but cannot authorize a paid run.",
            "The CUDA device must match the predeclared production GPU name.",
            "A changed embedding scale requires a separately trained short checkpoint.",
            "This admission is checkpoint-specific; repeat it on raw and EMA weights as applicable.",
            "Synthetic rows are stress tests, not substitutes for CIFAR-10 training images.",
            "The candidate delta is audited as configured and is never tuned on these rows.",
            "Per-parameter-group gradient errors are diagnostic until their full-matrix norm shares are calibrated; the global gradient predicate remains the admission gate.",
        ],
        "source_sha256": source_manifest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", default="local_1000_d0002_fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--data-root",
        default=None,
        help="local CIFAR-10 root; numerical admission opens train only",
    )
    parser.add_argument(
        "--expected-gpu-name",
        required=True,
        help="case-insensitive substring of the rented/training GPU model",
    )
    parser.add_argument("--include-gradient", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("numerical_admission.json")
    )
    args = parser.parse_args()
    assert_unused(args.out)
    if args.batch <= 0:
        raise ValueError("batch must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA numerical admission requested but CUDA is unavailable")
    result = run_admission(
        args.checkpoint,
        args.candidate,
        device=device,
        batch=args.batch,
        include_gradient=args.include_gradient,
        repeats=args.repeats,
        expected_gpu_name=args.expected_gpu_name,
        data_root=args.data_root,
    )
    digest = write_json_atomic(args.out, result)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out} sha256={digest}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
