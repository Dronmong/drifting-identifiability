"""Stage F1 confirmatory runner — protocol `EncoderIndependentF1Protocol.md` v3.

**Refuses to run unless its pre-conditions are on disk.** `--force` exists only
for a labelled dry run and marks the artifact non-confirmatory.

Pre-conditions, from §11:

  * `f1_calibration.json` exists and its verdict is GO
    (`p_null_upper < 0.025` at the fixed `RECALL_GATE = 0.05`);
  * `f1_checks.json` exists and every validation check passed.

Three independent replicate **units**, each an independent (source seed, teacher
seed) pair — bank selection, replay schedule and stochastic stream all vary per
unit, so no single target realization can drive every replicate (§10, §15.5).
Within a unit both regimes share the initial cloud, the kernel calibration and
the evaluation reference, keeping the regime contrast within-unit.

The gate is per-replicate and conjunctive (§8): recall, memorization veto,
collapse veto and the paired `real_data` control must all hold in the *same*
unit, then 2 of 3 units must pass. A unit whose control fails is excluded and
re-run rather than counted as a failure, since it shares that unit's teacher
seed and calibration; two control failures void the run.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_f1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import cifar
from .autoencoder import train_autoencoder
from .config import derive_seed
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnose_phase25 import train_cloud
from .diagnostics import provenance, write_json
from .f1 import (
    ARMS,
    CHECKPOINTS,
    CONTROL_RECALL_FLOOR,
    RECALL_GATE,
    REGIMES,
    RESAMPLES,
    UNITS,
    allocate,
    bank_statistics,
    build_kernel,
    real_nn_scale,
    rollout,
    score,
    source_cloud,
    take,
    unit_seeds,
)
from .f1_calibration import eval_references
from .fid import inception_features

HERE = Path(__file__).resolve().parent
PRIMARY_ARM = "random_generator"
GRID_CHECKPOINTS = (0, 40, 200, 1000)

# **The gate is evaluated at K = 200, not at the terminal checkpoint.**
#
# Declared before any primary-arm outcome beyond the K = 4 smoke was seen, and
# chosen from the POSITIVE CONTROL alone -- which is what a validity
# precondition is for.  Measured `real_data` trajectory under `replay`:
#
#     K      0    recall 0.7188   rank  9.25
#     K     40    recall 0.7163   rank  6.40
#     K    200    recall 0.6372   rank  4.16   <- control still valid (> 0.5)
#     K    500    recall 0.1904   rank  1.81   <- control INVALID
#     K   1000    recall 0.0093   rank  1.69
#     K  20000    recall 0.0137   rank  1.70   <- stable collapsed attractor
#
# Effective rank declines monotonically from step 0, so Phase 26's 40-step
# window observed the top of a decline and not stationarity.  The map has one
# attractor and real data decays into it on a ~500-step timescale.
#
# K = 200 is therefore the largest ladder point at which the F1 gate has a
# valid control.  Checkpoints beyond it are recorded as basin-structure
# characterization and are explicitly NOT gated.
GATE_CHECKPOINT = 200


def preconditions(calibration: Path, checks: Path) -> dict:
    """§11: calibration GO and all validation checks passed, both on disk."""
    state = {"calibration_present": calibration.exists(),
             "checks_present": checks.exists(),
             "calibration_go": False, "checks_passed": False,
             "recall_gate": RECALL_GATE, "p_null_upper": None}
    if state["calibration_present"]:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
        verdict = payload.get("verdict", {})
        state["calibration_go"] = verdict.get("decision") == "GO"
        state["p_null_upper"] = verdict.get("p_null_upper")
        state["calibrated_gate"] = verdict.get("recall_gate")
    if state["checks_present"]:
        payload = json.loads(checks.read_text(encoding="utf-8"))
        state["checks_passed"] = bool(
            payload.get("verdict", {}).get("all_passed"))
    state["satisfied"] = bool(state["calibration_go"] and state["checks_passed"])
    return state


def veto_thresholds(path: Path) -> dict:
    """§7 thresholds, calibrated externally. Absent means the veto abstains."""
    if not path.exists():
        return {"available": False,
                "note": "§7 veto calibration absent; vetoes abstain and the "
                        "gate is reported as INCOMPLETE"}
    return {"available": True,
            **json.loads(path.read_text(encoding="utf-8")).get("thresholds", {})}


def run_unit(unit: int, resolution: int, root: str | None, device, reference,
             real, steps_arms, ae_steps: int, bad_steps: int,
             checkpoints) -> dict:
    """One replicate unit: its own sources, bank, schedule and stream."""
    allocation = allocate(unit, resolution, root)
    allocation.assert_disjoint()
    branch, kernel = build_kernel(allocation, resolution, root, device)
    bank = take(resolution, "train", root, allocation.bank).to(device)
    scale = real_nn_scale(resolution, root, allocation.real_data)
    seeds = unit_seeds(unit)

    # Regenerated per unit with this unit's source seed: a single shared tensor
    # would make three identical controls, not three independent ones (§10).
    # The target must carry the device or `train_autoencoder` feeds CPU batches
    # to a CUDA model.
    ae_target = cifar.cifar_target(resolution, "train", root)
    ae_target.device = device
    fit = train_autoencoder(ae_target, ae_steps, seeds["source_init"], device,
                            latent_channels=32)
    bad, _ = train_cloud(bad_steps, seeds["source_init"], device, resolution,
                         root)

    rows = []
    for arm in steps_arms:
        start = source_cloud(arm, unit, resolution, root, device,
                             ae_model=fit["model"], trained_bad=bad)
        for regime in REGIMES:
            def observe(state, step, _arm=arm, _regime=regime):
                """Score EVERY checkpoint, not just the terminal one.

                Without this the history carried only kernel diagnostics and no
                metrics, so the trajectory §5 needs to distinguish slow progress
                from an attractor was never measured.  Resamples are reserved
                for the terminal state -- 200 per checkpoint would dominate the
                run -- so intermediate points carry point estimates only.
                """
                if step in GRID_CHECKPOINTS or step == max(checkpoints):
                    save_grid(state[:64].cpu(),
                              HERE / f"f1_u{unit}_{_arm}_{_regime}_K{step}.png")
                return score(state, reference, real, device)
            out = rollout(start, unit, regime, branch, kernel, resolution,
                          root, device, checkpoints=checkpoints,
                          on_checkpoint=observe)
            terminal = score(out["final"], reference, real, device,
                             resamples=RESAMPLES,
                             seed=derive_seed(seeds["source_init"], arm, regime))
            vetoes = bank_statistics(out["final"].cpu(), bank.cpu(), scale)
            rows.append({
                "unit": unit, "arm": arm, "regime": regime,
                "terminal": terminal,
                "veto_statistics": {k: v for k, v in vetoes.items()
                                    if not k.endswith("distribution")
                                    and k != "claimed_bank_indices"},
                "veto_distributions": {
                    "nearest_bank": vetoes["nearest_bank_distribution"],
                    "claimed_bank_indices": vetoes["claimed_bank_indices"]},
                "history": out["history"],
                "schedule_digest": out["schedule_digest"],
            })
            print(f"    u{unit} {arm:20}{regime:12} "
                  f"recall={terminal['recall']:.4f} "
                  f"CI={terminal['recall_ci'][0]:.4f}-"
                  f"{terminal['recall_ci'][1]:.4f} "
                  f"KID={terminal['kid']:+.5f} "
                  f"rank={terminal['effective_rank']:6.2f} "
                  f"distinct_bank={vetoes['distinct_bank']}", flush=True)
    return {"unit": unit, "seeds": seeds,
            "allocation_digests": allocation.digests, "rows": rows}


def decide(units: list[dict], thresholds: dict) -> dict:
    """§8's per-replicate conjunction, then 2-of-3 over units."""
    def at_gate(row: dict) -> dict:
        """Metrics at GATE_CHECKPOINT, where the control is still valid."""
        for entry in row["history"]:
            if entry["step"] == GATE_CHECKPOINT:
                return entry
        return row["terminal"]

    per_unit = {}
    for unit in units:
        by_key = {(r["arm"], r["regime"]): r for r in unit["rows"]}
        control = by_key.get(("real_data", "replay"))
        control_valid = bool(
            control is not None
            and at_gate(control).get("recall", 0.0) > CONTROL_RECALL_FLOOR)
        entry = {"control_valid": control_valid,
                 "gate_checkpoint": GATE_CHECKPOINT,
                 "control_recall_at_gate": (
                     None if control is None
                     else at_gate(control).get("recall")),
                 "control_recall_terminal": (
                     None if control is None
                     else control["terminal"]["recall"]),
                 "regimes": {}}
        for regime in REGIMES:
            row = by_key.get((PRIMARY_ARM, regime))
            if row is None:
                continue
            gated = at_gate(row)
            recall_ok = gated.get("recall", 0.0) > RECALL_GATE
            vetoes_ok = None if not thresholds.get("available") else True
            entry["regimes"][regime] = {
                "recall_at_gate": gated.get("recall"),
                "recall_terminal": row["terminal"]["recall"],
                "rank_at_gate": gated.get("effective_rank"),
                "rank_terminal": row["terminal"]["effective_rank"],
                "recall_passes": bool(recall_ok),
                "veto_passes": vetoes_ok,
                "replicate_pass": (None if vetoes_ok is None
                                   else bool(recall_ok and vetoes_ok
                                             and control_valid)),
            }
        per_unit[unit["unit"]] = entry

    controls_failed = [u for u, e in per_unit.items() if not e["control_valid"]]
    verdict = {
        "recall_gate": RECALL_GATE,
        "per_unit": per_unit,
        "units_failing_control": controls_failed,
        "run_void": len(controls_failed) >= 2,
        "veto_thresholds_available": bool(thresholds.get("available")),
    }
    if verdict["run_void"]:
        verdict["decision"] = "VOID"
        verdict["reading"] = (
            f"{len(controls_failed)} units failed the real_data control; per §8 "
            "the run is void and no arm may be read")
    elif not thresholds.get("available"):
        verdict["decision"] = "INCOMPLETE"
        verdict["reading"] = (
            "§7 veto thresholds are not calibrated, so the conjunctive gate "
            "cannot be evaluated; recall is reported but no F1 pass/fail is "
            "claimed")
    else:
        passes = {regime: sum(
            1 for e in per_unit.values()
            if e["regimes"].get(regime, {}).get("replicate_pass"))
            for regime in REGIMES}
        verdict["units_passing"] = passes
        verdict["decision"] = ("PASS" if max(passes.values()) >= 2
                               else "FAIL")
        verdict["reading"] = (
            f"F1 {verdict['decision']}: units passing = {passes} against the "
            "2-of-3 rule")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=UNITS)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--ae-steps", type=int, default=8000)
    parser.add_argument("--bad-steps", type=int, default=5000)
    parser.add_argument("--checkpoints", type=str,
                        default=",".join(str(c) for c in CHECKPOINTS))
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--calibration", type=Path,
                        default=HERE / "f1_calibration.json")
    parser.add_argument("--checks", type=Path, default=HERE / "f1_checks.json")
    parser.add_argument("--vetoes", type=Path, default=HERE / "f1_vetoes.json")
    parser.add_argument("--force", action="store_true",
                        help="dry run; marks the artifact non-confirmatory")
    parser.add_argument("--out", type=Path, default=HERE / "f1.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    gate = preconditions(args.calibration, args.checks)
    print("=== §11 pre-conditions ===")
    for key in ("calibration_present", "calibration_go", "checks_present",
                "checks_passed"):
        print(f"    {'OK ' if gate[key] else 'NO '} {key}")
    print(f"    p_null_upper = {gate['p_null_upper']}\n")
    if not gate["satisfied"] and not args.force:
        print("  REFUSING TO RUN. §11 requires calibration GO and all "
              "validation checks passed before any confirmatory arm.\n"
              "  Run f1_calibration.py then f1_checks.py, or pass --force for "
              "a labelled non-confirmatory dry run.")
        raise SystemExit(2)

    device = resolve_device(args.device)
    settings = configure(device)
    checkpoints = tuple(int(c) for c in args.checkpoints.split(","))
    arms = args.arms.split(",")

    started = time.time()
    primary, _ = eval_references(args.resolution, args.data_root)
    real = primary[:args.samples]
    reference = inception_features(real, device).double().numpy()
    thresholds = veto_thresholds(args.vetoes)
    if not thresholds["available"]:
        print(f"    NOTE: {thresholds['note']}\n", flush=True)

    units = []
    for unit in range(args.units):
        print(f"=== unit {unit} ===", flush=True)
        units.append(run_unit(unit, args.resolution, args.data_root, device,
                              reference, real, arms, args.ae_steps,
                              args.bad_steps, checkpoints))

    verdict = decide(units, thresholds)
    confirmatory = bool(gate["satisfied"] and not args.force
                        and thresholds["available"])
    payload = {"status": ("f1-confirmatory" if confirmatory
                          else "f1-NON-CONFIRMATORY-dry-run"),
               "protocol": "numerics/EncoderIndependentF1Protocol.md",
               "preconditions": gate, "confirmatory": confirmatory,
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {k: str(v) for k, v in
                                       (("out", args.out),
                                        ("calibration", args.calibration),
                                        ("checks", args.checks),
                                        ("vetoes", args.vetoes))},
               "elapsed_seconds": time.time() - started,
               "units": units, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== STAGE F1 ===")
    print(f"{'unit':6}{'regime':12}{'recall':>9}{'CI low':>9}{'CI high':>9}"
          f"{'passes':>8}")
    for unit in units:
        for row in unit["rows"]:
            if row["arm"] != PRIMARY_ARM:
                continue
            t = row["terminal"]
            g = verdict["per_unit"][row["unit"]]["regimes"].get(row["regime"], {})
            entry = verdict["per_unit"][row["unit"]]["regimes"].get(
                row["regime"], {})
            print(f"{row['unit']:<6}{row['regime']:12}{g.get('recall_at_gate', float('nan')):9.4f}"
                  f"{t['recall_ci'][0]:9.4f}{t['recall_ci'][1]:9.4f}"
                  f"{entry.get('replicate_pass')!s:>8}")
    print(f"\n    confirmatory: {confirmatory}")
    print(f"  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
