from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.modern_vla_probe import modern_vla_availability_markdown, run_modern_vla_availability_probe


def test_modern_vla_probe_records_local_matches_without_hf(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "openvla_policy").mkdir()
    results = tmp_path / "results"
    monkeypatch.setenv("HF_TOKEN", "redacted-test-token")

    payload = run_modern_vla_availability_probe(root, output_results_dir=results, probe_hf=False, scan_user_roots=False)

    assert payload["verified"] is True
    assert payload["probe_hf"] is False
    assert payload["local_vla_like_count"] >= 1
    assert payload["secret_status"]["env_present"]["HF_TOKEN"] is True
    assert payload["secret_status"]["tokens_redacted"] is True
    assert "LIBERO-compatible sparse-success VLA evaluation artifact" in payload["missing_for_ideal_claim"]
    assert (results / "modern_vla_availability_probe.json").exists()


def test_modern_vla_probe_counts_joint_runtime_as_eval_prerequisite(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    (results / "external_benchmark_runtime_probe.json").write_text(
        json.dumps(
            {
                "verified": True,
                "vla_libero_joint_runtime_available": True,
                "vla_runtime_success": {"name": "vla_robocasa_python_with_libero_source"},
            }
        ),
        encoding="utf-8",
    )

    payload = run_modern_vla_availability_probe(root, output_results_dir=results, probe_hf=False, scan_user_roots=False)

    assert payload["joint_runtime_probe_present"] is True
    assert payload["vla_libero_joint_runtime_available"] is True
    assert payload["ready_for_policy_eval"] is True
    assert "runnable modern VLA policy package" not in payload["missing_for_ideal_claim"]
    assert "pretrained VLA weights loaded" in payload["missing_for_ideal_claim"]
    assert "LIBERO-compatible sparse-success VLA evaluation artifact" in payload["missing_for_ideal_claim"]


def test_modern_vla_probe_counts_pretrained_load_but_not_eval(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    (results / "external_benchmark_runtime_probe.json").write_text(
        json.dumps({"verified": True, "vla_libero_joint_runtime_available": True}),
        encoding="utf-8",
    )
    (results / "modern_vla_pretrained_load_probe.json").write_text(
        json.dumps(
            {
                "verified": True,
                "pretrained_vla_loaded": True,
                "parameter_count": 450_046_212,
                "model_id": "lerobot/smolvla_base",
                "heldout_libero_policy_eval": False,
            }
        ),
        encoding="utf-8",
    )

    payload = run_modern_vla_availability_probe(root, output_results_dir=results, probe_hf=False, scan_user_roots=False)

    assert payload["pretrained_load_probe_present"] is True
    assert payload["pretrained_vla_loaded"] is True
    assert payload["pretrained_vla_parameter_count"] == 450_046_212
    assert "pretrained VLA weights loaded" not in payload["missing_for_ideal_claim"]
    assert payload["missing_for_ideal_claim"] == ["LIBERO-compatible sparse-success VLA evaluation artifact"]


def test_modern_vla_probe_records_failed_libero_execution_attempt(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    (results / "external_benchmark_runtime_probe.json").write_text(
        json.dumps({"verified": True, "vla_libero_joint_runtime_available": True}),
        encoding="utf-8",
    )
    (results / "modern_vla_pretrained_load_probe.json").write_text(
        json.dumps({"verified": True, "pretrained_vla_loaded": True, "parameter_count": 450_046_212}),
        encoding="utf-8",
    )
    (results / "modern_vla_libero_execution_probe.json").write_text(
        json.dumps(
            {
                "attempted": True,
                "verified": False,
                "policy_loaded": False,
                "action_selected": False,
                "libero_step_succeeded": False,
                "failure_stage": "policy_load",
                "error_type": "NotImplementedError",
                "heldout_libero_policy_eval": False,
            }
        ),
        encoding="utf-8",
    )

    payload = run_modern_vla_availability_probe(root, output_results_dir=results, probe_hf=False, scan_user_roots=False)

    assert payload["libero_execution_probe_present"] is True
    assert payload["libero_execution_attempted"] is True
    assert payload["libero_execution_verified"] is False
    assert payload["libero_execution_failure_stage"] == "policy_load"
    assert payload["runtime_ready_for_policy_eval_attempt"] is True
    assert payload["ready_for_policy_eval"] is False
    assert "pretrained VLA can select an action and step LIBERO in a compatible runtime" in payload["missing_for_ideal_claim"]
    assert "LIBERO-compatible sparse-success VLA evaluation artifact" in payload["missing_for_ideal_claim"]


def test_modern_vla_probe_markdown_does_not_dump_secret_values() -> None:
    payload = {
        "verified": True,
        "vla_package_importable": False,
        "local_vla_like_count": 0,
        "hf_reachable_count": 0,
        "vla_libero_joint_runtime_available": False,
        "pretrained_vla_loaded": False,
        "pretrained_vla_parameter_count": None,
        "libero_execution_probe_present": False,
        "libero_execution_verified": False,
        "libero_execution_failure_stage": None,
        "ready_for_policy_eval": False,
        "missing_for_ideal_claim": ["runnable modern VLA policy package"],
        "packages": [{"name": "openvla", "importable": False}],
        "hf_models": [{"repo_id": "openvla/openvla-7b", "reachable": False, "error_type": "HTTPError"}],
        "secret_status": {"env_present": {"HF_TOKEN": True}, "tokens_redacted": True},
    }

    text = modern_vla_availability_markdown(payload)

    assert "HF_TOKEN" not in text
    assert "runnable modern VLA policy package" in text
    assert "openvla/openvla-7b" in text
