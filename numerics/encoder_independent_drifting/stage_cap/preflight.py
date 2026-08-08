"""CAP-EMF-1 local port audit — protocol section 8.1.

Nine checks, **no quality run**.  The development laptop cannot train this
model and does not try; it verifies mechanics and produces the throughput
numbers the cloud benchmark will replace with measurements.

Check 7 (feature-tap parity) is here rather than deferred to ASFD on purpose:
retrofitting hooks after the run would change the trunk's source hash and break
exactly the kind of binding that blocked B2.5.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..diagnostics import write_json
from .artifacts import (
    DEFAULT_PREFLIGHT,
    PROTOCOL,
    assert_result_path_unused,
    file_sha256,
    profile_payload,
    source_manifest,
)

PARAMETER_CEILING_NOTE = (
    "Protocol 5.4: the port reports the exact count and it is frozen at that "
    "value. If the cloud benchmark exceeds budget, reduce width once - never "
    "the patch grid, the objective, or the training horizon."
)
from .config import (
    FEATURE_LEVELS,
    PARAMETER_CEILING,
    enable_tf32,
    examples_seen,
    profile,
)
from .diagnostics import haar_transform, rank_noncollapse
from .model import CAPPixelTransformer, one_step_sample
from .objective import (
    directional_jvp_reference,
    emf_local_difference,
    emf_loss,
    sample_time_triangle,
)
from .training import EMAState, train_cap_unit

Check = tuple[str, bool, str, dict]


def _tiny(seed: int = 11) -> tuple[CAPPixelTransformer, object]:
    small = profile("smoke")
    return CAPPixelTransformer(small.model, seed), small


def wake_output_path(model: CAPPixelTransformer, seed: int = 29) -> CAPPixelTransformer:
    """Randomize the deliberately zero-initialized output path.

    ``pixel_head`` and the refiner's final convolution are zero at
    initialization, and AdaLN-Zero zeroes every modulation, so a freshly
    constructed model is **the zero function** — a genuine and intended
    property, and the reason a derivative test on an untouched model is
    vacuous: both the difference quotient and the exact JVP are identically
    zero, so any comparison passes trivially and any monotonicity check fails.

    Any audit of the EMF derivative must run against a non-degenerate model.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.pixel_head.weight.normal_(0.0, 0.05, generator=generator)
        model.pixel_head.bias.normal_(0.0, 0.05, generator=generator)
        final_conv = model.refiner.body[-1]
        final_conv.weight.normal_(0.0, 0.05, generator=generator)
        final_conv.bias.normal_(0.0, 0.05, generator=generator)
        for block in list(model.encoder) + list(model.decoder):
            block.modulation.weight.normal_(0.0, 0.05, generator=generator)
            block.modulation.bias.normal_(0.0, 0.05, generator=generator)
    return model


def check_shapes() -> Check:
    """1. Shape identities at the frozen capability geometry."""
    frozen = profile("capability")
    model = CAPPixelTransformer(frozen.model, 3)
    batch = 2
    images = torch.randn(batch, 3, 32, 32)
    t = torch.full((batch,), 0.7)
    h = torch.full((batch,), 0.3)
    with torch.no_grad():
        out, features = model.forward_with_features(images, t, h)
    ok = out.shape == images.shape and frozen.model.tokens == 256
    detail = f"output {tuple(out.shape)}, {frozen.model.tokens} tokens"
    levels = {name: tuple(value.shape) for name, value in features.items()}
    for shape in levels.values():
        ok = ok and shape == (batch, 256, frozen.model.width)
    return "shape identities", ok, detail, {"levels": levels}


def check_emf_against_jvp() -> Check:
    """2. The stopped local difference converges to the directional JVP.

    Run against a woken model: an untouched one is the zero function and the
    comparison would be vacuous.  Convergence must be **first order** — a sign
    error, a clock error, or a wrong Euler velocity would break the rate even
    when the magnitudes happen to look plausible.
    """
    model, small = _tiny()
    model = wake_output_path(model.double().eval())
    config = small.objective
    state = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    t = torch.tensor([0.9, 0.7, 0.5], dtype=torch.float64)
    r = torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64)
    exact = directional_jvp_reference(
        model, state, t, r, config.emf_denominator_floor
    ).detach()
    magnitude = float(exact.abs().max())
    deltas = (1e-3, 1e-4, 1e-5)
    errors = {}
    for delta in deltas:
        _, quotient = emf_local_difference(
            model, state, t, r, delta, config.emf_denominator_floor
        )
        errors[f"{delta:g}"] = float((quotient - exact).abs().max()) / magnitude
    values = [errors[f"{delta:g}"] for delta in deltas]
    rates = [values[i] / values[i + 1] for i in range(len(values) - 1)]
    ok = (
        magnitude > 1e-6
        and all(4.0 <= rate <= 20.0 for rate in rates)
        and values[-1] < 1e-2
    )
    return (
        "EMF finite difference vs float64 JVP",
        bool(ok),
        (
            f"relative max error {errors}, first-order rates "
            f"{[round(rate, 2) for rate in rates]}"
        ),
        {
            "relative_errors": errors,
            "convergence_rates": rates,
            "jvp_magnitude": magnitude,
            "note": (
                "the production emf_delta of 0.01 is a stopped local "
                "difference by design, not an exact derivative; this check "
                "verifies the implementation converges to the right limit"
            ),
        },
    )


def check_one_call() -> Check:
    """3. The sampler makes exactly one network evaluation."""
    model, _ = _tiny()
    model.eval()
    calls = {"n": 0}
    handle = model.register_forward_pre_hook(
        lambda module, args: calls.__setitem__("n", calls["n"] + 1)
    )
    with torch.no_grad():
        sample = one_step_sample(model, torch.randn(2, 3, 8, 8))
    handle.remove()
    ok = calls["n"] == 1 and sample.shape == (2, 3, 8, 8)
    return (
        "one-call inference",
        ok,
        f"{calls['n']} model evaluation(s)",
        {"forward_calls": calls["n"]},
    )


def check_restart_determinism(tmp: Path) -> Check:
    """4. A resumed run reaches the same weights as an uninterrupted one."""
    small = profile("smoke")
    pool = torch.randn(32, 3, 8, 8)
    reference = train_cap_unit(pool, small, "cpu")
    partial_path = tmp / "recovery.pt"
    if partial_path.exists():
        partial_path.unlink()
    # Stop after two updates, then resume from the recovery file.
    half = replace(
        small, train=replace(small.train, updates=2, checkpoint_updates=(2,))
    )
    train_cap_unit(pool, half, "cpu", recovery_path=partial_path)
    resumed = train_cap_unit(pool, small, "cpu", recovery_path=partial_path)
    a = reference.history[-1]
    b = resumed.history[-1]
    ok = (
        reference.optimizer_updates == resumed.optimizer_updates
        and abs(a["raw_mse"] - b["raw_mse"]) < 1e-9
    )
    return (
        "restart determinism",
        ok,
        f"final raw_mse {a['raw_mse']:.12g} vs {b['raw_mse']:.12g}",
        {"uninterrupted": a["raw_mse"], "resumed": b["raw_mse"]},
    )


def check_ema() -> Check:
    """5. EMA arithmetic and its maturity accounting."""
    model, _small = _tiny()
    ema = EMAState(model, 0.9)
    reference = {
        name: value.detach().clone().float()
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    name = next(iter(reference))
    expected = reference[name] * 0.9 + model.state_dict()[name].float() * 0.1
    arithmetic = torch.allclose(ema.shadow[name], expected, atol=1e-6)
    weight = abs(ema.initialization_weight() - 0.9) < 1e-12
    round_trip = EMAState(model, 0.9)
    round_trip.load_recovery_state(ema.recovery_state())
    restored = torch.allclose(round_trip.shadow[name], ema.shadow[name])
    frozen = profile("capability").train
    mature = frozen.ema_mature_at()
    ok = arithmetic and weight and restored and mature < frozen.updates
    return (
        "EMA arithmetic and maturity",
        ok,
        (
            f"half-life {frozen.ema_half_life:.0f} updates, mature by {mature}, "
            f"residual initialization weight at {frozen.updates} = "
            f"{frozen.ema_decay**frozen.updates:.2e}"
        ),
        {
            "half_life": frozen.ema_half_life,
            "mature_at": mature,
            "residual_initialization_weight": frozen.ema_decay**frozen.updates,
        },
    )


def check_haar_and_rank_rule() -> Check:
    """6. Haar orthonormality and the corrected rank rule."""
    images = torch.randn(5, 3, 32, 32, dtype=torch.float64)
    coefficients = haar_transform(images)
    conserved = (
        abs(float(coefficients.square().sum()) - float(images.square().sum())) < 1e-8
    )
    # S3R's EMF arm: best ratio 4.056, final 1.661.  final/max = 0.410 failed
    # it; the corrected rule accepts it because approaching the target is not
    # collapse.  pMF's 0.349 must still fail.
    accepts_emf = rank_noncollapse(1.661, 4.056, 0.8)
    rejects_pmf = not rank_noncollapse(0.349, 4.056, 0.8)
    ok = conserved and accepts_emf and rejects_pmf
    return (
        "Haar energy and corrected rank rule",
        ok,
        (
            f"energy conserved={conserved}, accepts S3R EMF={accepts_emf}, "
            f"rejects pMF={rejects_pmf}"
        ),
        {"accepts_emf_1661": accepts_emf, "rejects_pmf_0349": rejects_pmf},
    )


def check_feature_parity() -> Check:
    """7. Taps are read-only, and the grid is 16x16 with no slice."""
    frozen = profile("capability")
    model = CAPPixelTransformer(frozen.model, 5).eval()
    images = torch.randn(2, 3, 32, 32)
    t = torch.full((2,), 0.6)
    h = torch.full((2,), 0.25)
    with torch.no_grad():
        plain = model(images, t, h)
        tapped, features = model.forward_with_features(images, t, h)
    identical = bool(torch.equal(plain, tapped))
    names = [name for name, _, _ in FEATURE_LEVELS]
    present = sorted(features) == sorted(names)
    grids = {}
    reshapes = True
    for name, value in features.items():
        reshapes = reshapes and value.shape[1] == 256
        grids[name] = tuple(value.shape)
        # 256 tokens must be a 16x16 grid; a conditioning-token slice would
        # leave 254 or 258 and fail here rather than silently mis-registering.
        value.reshape(len(value), 16, 16, frozen.model.width)
    ok = identical and present and reshapes
    return (
        "feature-tap parity and 16x16 grid",
        ok,
        f"bit-identical={identical}, levels={sorted(features)}",
        {"bit_identical": identical, "grids": grids},
    )


def check_parameter_count() -> Check:
    """8. The frozen model fits under the declared ceiling."""
    frozen = profile("capability")
    model = CAPPixelTransformer(frozen.model, 7)
    count = model.parameter_count()
    ok = count <= PARAMETER_CEILING
    return (
        "parameter count",
        ok,
        (
            f"{count:,} parameters against a {PARAMETER_CEILING:,} ceiling "
            f"({100 * count / PARAMETER_CEILING:.1f}%)"
        ),
        {"parameter_count": count, "ceiling": PARAMETER_CEILING},
    )


def _probe_once(
    device: torch.device, micro_batch: int, steps: int, warmup: int
) -> dict:
    frozen = profile("capability")
    model = CAPPixelTransformer(frozen.model, 9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=frozen.train.learning_rate)
    generator = torch.Generator().manual_seed(17)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = None
    for index in range(warmup + steps):
        if index == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.time()
        optimizer.zero_grad(set_to_none=True)
        clean = torch.randn(micro_batch, 3, 32, 32, device=device)
        noise = torch.randn(micro_batch, 3, 32, 32, device=device)
        triangle = sample_time_triangle(
            micro_batch, frozen.objective, generator, device
        )
        emf_loss(model, clean, noise, triangle, frozen.objective).loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), frozen.train.gradient_clip)
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    per_microbatch = (time.time() - started) / steps
    accumulation = frozen.train.effective_batch / micro_batch
    per_update = per_microbatch * accumulation
    return {
        "micro_batch": micro_batch,
        "steps": steps,
        "warmup": warmup,
        "seconds_per_microbatch": per_microbatch,
        "seconds_per_update": per_update,
        "projected_hours_this_device": per_update * frozen.train.updates / 3_600,
        "peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }


def check_throughput(device: torch.device, micro_batch: int, steps: int) -> Check:
    """9. Throughput and memory at the production microbatch where it fits.

    Warmup steps are excluded: the first kernels include autotuning, and
    including them inflated an earlier probe by about 30%.  The probe also
    starts at the production microbatch and only steps down on OOM, because a
    reduced-microbatch measurement scaled up by accumulation is latency-bound
    and systematically pessimistic — the first draft of this check reported
    152 h where the production shape measures 99 h on the same GPU.
    """
    frozen = profile("capability")
    warmup = max(2, steps // 5)
    ladder = [micro_batch]
    while ladder[-1] > 1:
        ladder.append(ladder[-1] // 2)
    attempts, chosen, failures = [], None, []
    for candidate in ladder:
        if frozen.train.effective_batch % candidate:
            continue
        try:
            record = _probe_once(device, candidate, steps, warmup)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            failures.append({"micro_batch": candidate, "error": str(error)[:160]})
            continue
        attempts.append(record)
        if chosen is None:
            chosen = record
        break
    if chosen is None:
        return (
            "throughput probe",
            False,
            f"no microbatch fit on this device: {failures}",
            {"failures": failures},
        )
    fits_production = chosen["micro_batch"] == frozen.train.micro_batch
    return (
        "throughput probe",
        True,
        (
            f"micro_batch {chosen['micro_batch']}"
            f"{'' if fits_production else ' (production shape did not fit)'}: "
            f"{chosen['seconds_per_update']:.3f} s/update, "
            f"{chosen['peak_memory_reserved_bytes'] / 2**30:.2f} GiB, "
            f"{chosen['projected_hours_this_device']:.1f} h on this device"
        ),
        {
            "chosen": chosen,
            "attempts": attempts,
            "failures": failures,
            "production_micro_batch_fits": fits_production,
            "caveat": (
                "development hardware. The cloud benchmark replaces this with "
                "a measurement on the rented GPU and selects by dollars per "
                "update; this number only bounds the projection."
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-EMF-1 local port audit")
    parser.add_argument("--out", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--probe-micro-batch",
        type=int,
        default=profile("capability").train.micro_batch,
        help="starting microbatch; the probe steps down only if it does not fit",
    )
    parser.add_argument("--probe-steps", type=int, default=20)
    parser.add_argument("--scratch", type=Path, default=None)
    args = parser.parse_args()

    assert_result_path_unused(args.out)
    device = resolve_device(args.device)
    settings = configure(device)
    # The throughput probe must measure the precision the run will use.
    precision = enable_tf32()
    scratch = args.scratch or (args.out.parent / "_preflight_scratch")
    scratch.mkdir(parents=True, exist_ok=True)

    checks: list[Check] = [
        check_shapes(),
        check_emf_against_jvp(),
        check_one_call(),
        check_restart_determinism(scratch),
        check_ema(),
        check_haar_and_rank_rule(),
        check_feature_parity(),
        check_parameter_count(),
        check_throughput(device, args.probe_micro_batch, args.probe_steps),
    ]

    width = max(len(name) for name, _, _, _ in checks)
    print("=== CAP-EMF-1 local port audit ===")
    for name, ok, detail, _ in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _, _ in checks if not ok]

    payload = {
        "status": "cap-emf1-preflight",
        "scope": "local mechanics only; no quality run and no test access",
        "device": settings,
        "precision": precision,
        "examples_seen_target": examples_seen(profile("capability").train),
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "profile": profile_payload(profile("capability")),
        "checks": [
            {"name": name, "passed": ok, "detail": detail, "data": data}
            for name, ok, detail, data in checks
        ],
        "parameter_ceiling_note": PARAMETER_CEILING_NOTE,
        "verdict": {
            "decision": "GO" if not failed else "BLOCKED",
            "failed": failed,
            "reading": (
                "GO: mechanics, EMF derivative, one-call inference, restart "
                "determinism, EMA, rank rule and feature taps all verified. "
                "This authorizes the cloud benchmark, not the training run."
                if not failed
                else f"BLOCKED: {', '.join(failed)}"
            ),
        },
    }
    digest = write_json(args.out, payload)
    print()
    print(f"wrote {args.out} sha256={digest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
