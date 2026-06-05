from __future__ import annotations

from pathlib import Path

from wam_inference_value.modern_vla_execution_probe import (
    modern_vla_libero_execution_markdown,
    run_modern_vla_libero_execution_probe,
)


def test_execution_probe_records_missing_python_without_promoting_eval(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"

    payload = run_modern_vla_libero_execution_probe(
        root,
        python_path=tmp_path / "missing" / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        timeout_s=1,
    )

    assert payload["attempted"] is True
    assert payload["verified"] is False
    assert payload["heldout_libero_policy_eval"] is False
    assert payload["failure_stage"] == "python_missing"
    assert payload["error_type"] == "PythonMissing"
    assert (results / "modern_vla_libero_execution_probe.json").exists()
    assert (root / "reports" / "modern_vla_libero_execution_probe_report.md").exists()


def test_execution_probe_markdown_states_not_performance_result() -> None:
    text = modern_vla_libero_execution_markdown(
        {
            "attempted": True,
            "verified": False,
            "heldout_libero_policy_eval": False,
            "model_id": "HuggingFaceVLA/smolvla_libero",
            "failure_stage": "policy_load",
            "error_type": "NotImplementedError",
            "parameter_count": 604_934_220,
            "processor_stats_loaded": {"preprocessor": True, "postprocessor": True},
            "input_feature_keys": ["observation.images.image", "observation.state"],
        }
    )

    assert "not a modern VLA LIBERO performance result" in text
    assert "policy_load" in text
    assert "604934220" in text
