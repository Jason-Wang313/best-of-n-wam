from __future__ import annotations

from pathlib import Path

from wam_inference_value.modern_vla_policy_eval import (
    modern_vla_libero_policy_eval_markdown,
    run_modern_vla_libero_policy_eval,
    wilson_ci,
)


def test_policy_eval_records_missing_python_without_promoting_eval(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"

    payload = run_modern_vla_libero_policy_eval(
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
    assert payload["success_ci"]["n"] == 0
    assert (results / "modern_vla_libero_policy_eval.json").exists()
    assert (root / "reports" / "modern_vla_libero_policy_eval_report.md").exists()


def test_policy_eval_markdown_does_not_promote_zero_success() -> None:
    text = modern_vla_libero_policy_eval_markdown(
        {
            "attempted": True,
            "verified": True,
            "heldout_libero_policy_eval": True,
            "model_id": "HuggingFaceVLA/smolvla_libero",
            "suite": "libero_object",
            "task_index": 0,
            "eval_episodes": 5,
            "eval_seeds": [300, 301, 302, 303, 304],
            "eval_successes": 0,
            "eval_success_rate": 0.0,
            "success_ci": wilson_ci(0, 5),
        }
    )

    assert "does not promote a positive modern VLA performance claim" in text
    assert "successes: `0`" in text


def test_wilson_ci_for_sparse_success_rate() -> None:
    ci = wilson_ci(1, 5)

    assert ci["n"] == 5
    assert ci["mean"] == 0.2
    assert 0.0 < ci["lo"] < ci["mean"] < ci["hi"] < 1.0
