"""Post-hoc EMA synthesis: arithmetic, integrity, and selection discipline."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from ..artifacts import file_sha256, profile_payload, save_snapshot
from ..config import screen_profile
from ..posthoc_ema import DECLARED_WINDOWS, synthesize

ARM = "ordered_uniform"
HASH = "a" * 64


def _unit_with_snapshots(root: Path, steps: tuple[int, ...]) -> tuple[dict, Path]:
    """Write real snapshot files and the ledger that binds them."""
    frozen = screen_profile(ARM, "local_1000_d0002_fp32", smoke=True)
    declared = profile_payload(frozen)
    snapshot_root = root / "raw_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    ledger = []
    for index, step in enumerate(steps):
        # Distinct constant weights per snapshot, so the mean is exactly known.
        state = {"w": torch.full((4,), float(index + 1))}
        path = snapshot_root / f"cap2_{ARM}_snapshot_step{step}.pt"
        digest = save_snapshot(
            path,
            state,
            step=step,
            arm=ARM,
            declared_profile=declared,
            realized_profile=declared,
            preflight_sha256=HASH,
            run_identity_sha256=HASH,
            unit_seed=0,
        )
        ledger.append(
            {"step": step, "path": f"raw_snapshots/{path.name}", "sha256": digest}
        )
    unit = {
        "arm": ARM,
        "declared_profile": declared,
        "preflight_sha256": HASH,
        "run_identity_sha256": HASH,
        "unit_seed": 0,
        "raw_snapshots": ledger,
    }
    unit_path = root / "unit.json"
    unit_path.write_text(json.dumps(unit), encoding="utf-8")
    return unit, unit_path


def test_average_is_exact_and_uses_the_trailing_window():
    steps = tuple(range(25_000, 25_000 * 9, 25_000))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        unit, unit_path = _unit_with_snapshots(root, steps)
        result = synthesize(unit, unit_path, 4, root / "out")
        payload = torch.load(result.path, map_location="cpu", weights_only=False)
        # Snapshots carry 1..8; the trailing four are 5,6,7,8 -> mean 6.5.
        assert torch.allclose(payload["state_dict"]["w"], torch.full((4,), 6.5))
        assert result.first_step == steps[-4]
        assert result.last_step == steps[-1]
        assert result.update_span == 75_000


def test_a_tampered_snapshot_is_refused():
    steps = tuple(range(25_000, 25_000 * 5, 25_000))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        unit, unit_path = _unit_with_snapshots(root, steps)
        victim = root / "raw_snapshots" / f"cap2_{ARM}_snapshot_step100000.pt"
        payload = torch.load(victim, map_location="cpu", weights_only=False)
        payload["state_dict"]["w"] = torch.zeros(4)
        torch.save(payload, victim)
        with pytest.raises(RuntimeError, match="changed after the unit was sealed"):
            synthesize(unit, unit_path, 2, root / "out")


def test_synthesis_refuses_to_overwrite_and_to_exceed_its_material():
    steps = tuple(range(25_000, 25_000 * 5, 25_000))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        unit, unit_path = _unit_with_snapshots(root, steps)
        synthesize(unit, unit_path, 2, root / "out")
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            synthesize(unit, unit_path, 2, root / "out")
        with pytest.raises(ValueError, match="exceeds"):
            synthesize(unit, unit_path, 99, root / "out2")
        with pytest.raises(ValueError, match="at least two"):
            synthesize(unit, unit_path, 1, root / "out3")


def test_windows_are_predeclared_and_bracket_the_cap1_result():
    """A sweep that could pick its own window would be post-hoc selection.

    8 snapshots at the 25,000 cadence is the 200,000-update window that moved
    CAP-EMF-1 from FID 112.94 to 83.65; the neighbours bracket it so the report
    shows whether that is a plateau or a peak.
    """
    assert DECLARED_WINDOWS == (4, 8, 16)
    assert 8 in DECLARED_WINDOWS
    assert min(DECLARED_WINDOWS) >= 2


def test_synthesized_weights_are_marked_ineligible_for_selection():
    steps = tuple(range(25_000, 25_000 * 5, 25_000))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        unit, unit_path = _unit_with_snapshots(root, steps)
        result = synthesize(unit, unit_path, 2, root / "out")
        payload = torch.load(result.path, map_location="cpu", weights_only=False)
        assert payload["secondary_exploratory"] is True
        assert payload["eligible_for_selection"] is False
        assert payload["stage"] == "cap-emf-2-posthoc-ema"
        # Must load through the same path as a real checkpoint.
        assert "profile" in payload and "model" in payload["profile"]
        assert file_sha256(result.path) == result.sha256


def test_module_cannot_open_the_sealed_test_split():
    """Structural: no code path here reaches the sealed split."""
    source = (Path(__file__).resolve().parents[1] / "posthoc_ema.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("sealed_test_pool", "acknowledge_sealed", "train=False"):
        assert forbidden not in source, forbidden


def test_a_truncated_run_skips_wide_windows_rather_than_aborting():
    """A budget stop truncates the snapshot ledger; CAP-EMF-1 ended on one.

    The widest declared window not fitting must not prevent the narrower ones
    from being reported.
    """
    available = 6
    usable = [w for w in DECLARED_WINDOWS if w <= available]
    skipped = [w for w in DECLARED_WINDOWS if w > available]
    assert usable == [4], usable
    assert skipped == [8, 16], skipped
    # And the synthesis itself still refuses an impossible window explicitly.
    with TemporaryDirectory() as directory:
        root = Path(directory)
        unit, unit_path = _unit_with_snapshots(
            root, tuple(range(25_000, 25_000 * (available + 1), 25_000))
        )
        with pytest.raises(ValueError, match="exceeds"):
            synthesize(unit, unit_path, 8, root / "out")
        result = synthesize(unit, unit_path, 4, root / "out")
        assert result.window == 4
