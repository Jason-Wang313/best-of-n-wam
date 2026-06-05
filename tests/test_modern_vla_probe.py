from __future__ import annotations

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


def test_modern_vla_probe_markdown_does_not_dump_secret_values() -> None:
    payload = {
        "verified": True,
        "vla_package_importable": False,
        "local_vla_like_count": 0,
        "hf_reachable_count": 0,
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
