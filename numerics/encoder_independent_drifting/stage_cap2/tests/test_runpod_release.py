from pathlib import Path

from ..artifacts import PROTOCOL, source_manifest


def _package_relative(name: str) -> str:
    return name.split("encoder_independent_drifting/")[-1]


def test_runpod_protocol_is_the_source_bound_paid_run_card() -> None:
    assert PROTOCOL.name == "EncoderIndependentCAPEMF2ASFDRunPodProtocol.md"
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "RunPod Secure Cloud" in text
    assert "one 750k CAP foundation" in text
    assert "There is deliberately no `all` command" in text
    assert "foundation-admit-50k" in text
    assert "CAP_ASFD_FOUNDATION_DECISION" in text
    assert "CAP_ASFD_FINAL_DECISION" in text


def test_runpod_launchers_are_source_bound() -> None:
    manifest = {_package_relative(name) for name in source_manifest()}
    assert "stage_cap2/runpod_bootstrap.sh" in manifest
    assert "stage_cap2/runpod_pipeline.sh" in manifest


def test_runpod_pipeline_has_every_gate_and_no_all_command() -> None:
    pipeline = Path(__file__).resolve().parents[1] / "runpod_pipeline.sh"
    text = pipeline.read_text(encoding="utf-8")
    required = {
        "prepare",
        "evidence",
        "admission",
        "foundation-phase-a",
        "foundation-admit-50k",
        "foundation-phase-b",
        "foundation-evaluate",
        "foundation-review",
        "asfd-prepare",
        "asfd-run",
        "final-evaluate",
        "final-review",
        "restore-foundation",
        "restore-asfd",
    }
    for command in required:
        assert f"  {command})" in text
    assert "  all)" not in text
    assert "--pause-for-early-admission" in text
    assert "--i-have-authorized-the-screen-run" in text
    assert "--i-have-authorized-asfd-continuation" in text
    assert "runpod_operator.env.sha256" in text
    assert "RUNPOD_RELEASE_COMMIT" in text
    assert "RUNPOD_NETWORK_VOLUME_ID" in text
    assert "RUNPOD_POD_HOURLY_RATE" in text


def test_runpod_bootstrap_pins_the_release_environment() -> None:
    bootstrap = Path(__file__).resolve().parents[1] / "runpod_bootstrap.sh"
    text = bootstrap.read_text(encoding="utf-8")
    for token in (
        'PYTHON_VERSION="3.11.15"',
        'UV_VERSION="0.8.14"',
        '"torch": "2.7.1+cu126"',
        '"torchvision": "0.22.1+cu126"',
        '"numpy": "1.26.4"',
        '"pillow": "12.2.0"',
        "torch.cuda.is_available()",
        "CUBLAS_WORKSPACE_CONFIG",
        "tmux",
    ):
        assert token in text
