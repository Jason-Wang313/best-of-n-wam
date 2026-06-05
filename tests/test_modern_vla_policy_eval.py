from __future__ import annotations

import json
from pathlib import Path

import wam_inference_value.modern_vla_policy_eval as policy_eval
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


def test_policy_eval_appends_only_compatible_episode_sets(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"

    def fake_child(seed: int, steps: int) -> dict:
        return {
            "returncode": 0,
            "stdout_tail": [],
            "stderr_tail": [],
            "child": {
                "verified": True,
                "heldout_libero_policy_eval": True,
                "policy_loaded": True,
                "parameter_count": 604_934_220,
                "processor_stats_loaded": {"preprocessor": True},
                "action_selected": True,
                "libero_steps_succeeded": True,
                "input_feature_keys": ["observation.images.image", "observation.state"],
                "episodes": [
                    {
                        "episode_id": 0,
                        "seed": seed,
                        "steps": steps,
                        "success": False,
                        "initial_distance": 1.0,
                        "final_distance": 0.9,
                        "progress": 0.1,
                        "total_reward": 0.0,
                        "energy": 1.0,
                        "done": False,
                        "truncated": False,
                        "step_error": None,
                    }
                ],
            },
        }

    calls = iter([fake_child(300, 1), fake_child(301, 1), fake_child(302, 5)])

    def fake_run_child(*args, **kwargs):
        return next(calls)

    monkeypatch.setattr(policy_eval, "_run_child", fake_run_child)

    first = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[300],
        horizon=2,
        max_steps=1,
        append_existing=True,
    )
    second = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[301],
        horizon=2,
        max_steps=1,
        append_existing=True,
    )
    incompatible = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[302],
        horizon=2,
        max_steps=5,
        append_existing=True,
    )

    assert first["eval_seeds"] == [300]
    assert second["eval_seeds"] == [300, 301]
    assert second["n_existing_compatible_episodes"] == 1
    assert incompatible["eval_seeds"] == [302]
    assert incompatible["n_existing_compatible_episodes"] == 0


def test_policy_eval_preserves_completed_eval_after_failed_append_chunk(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"

    success_child = {
        "returncode": 0,
        "stdout_tail": [],
        "stderr_tail": [],
        "child": {
            "verified": True,
            "heldout_libero_policy_eval": True,
            "policy_loaded": True,
            "parameter_count": 604_934_220,
            "processor_stats_loaded": {"preprocessor": True},
            "action_selected": True,
            "libero_steps_succeeded": True,
            "episodes": [
                {
                    "episode_id": 0,
                    "seed": 300,
                    "steps": 1,
                    "success": False,
                    "initial_distance": 1.0,
                    "final_distance": 0.9,
                    "progress": 0.1,
                    "total_reward": 0.0,
                    "energy": 1.0,
                    "done": False,
                    "truncated": False,
                    "step_error": None,
                }
            ],
        },
    }
    failed_child = {
        "returncode": 3221225477,
        "stdout_tail": ["Loading weights ..."],
        "stderr_tail": [],
        "child": {
            "verified": False,
            "heldout_libero_policy_eval": False,
            "failure_stage": "process_crash",
            "error_type": "WindowsAccessViolation",
            "episodes": [],
        },
    }
    timeout_child = {
        "returncode": None,
        "stdout_tail": ["Loading weights ..."],
        "stderr_tail": [],
        "child": {
            "verified": False,
            "heldout_libero_policy_eval": False,
            "failure_stage": "timeout",
            "error_type": "TimeoutExpired",
            "episodes": [],
        },
    }
    calls = iter([success_child, failed_child, timeout_child])
    monkeypatch.setattr(policy_eval, "_run_child", lambda *args, **kwargs: next(calls))

    completed = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[300],
        horizon=2,
        max_steps=1,
        append_existing=True,
    )
    preserved = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[301],
        horizon=5,
        max_steps=5,
        append_existing=True,
    )

    assert completed["eval_episodes"] == 1
    assert preserved["eval_episodes"] == 1
    assert preserved["max_steps"] == 1
    assert preserved["latest_attempt_preserved_previous"] is True
    last_attempt = results / "modern_vla_libero_policy_eval_last_attempt.json"
    assert last_attempt.exists()
    failed_payload = json.loads(last_attempt.read_text(encoding="utf-8"))
    assert failed_payload["failure_stage"] == "process_crash"
    assert failed_payload["max_steps"] == 5
    assert failed_payload["attempt_history"][0]["error_type"] == "WindowsAccessViolation"

    preserved_again = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[302],
        horizon=2,
        max_steps=2,
        append_existing=True,
    )

    failed_again_payload = json.loads(last_attempt.read_text(encoding="utf-8"))
    assert preserved_again["eval_episodes"] == 1
    assert preserved_again["attempt_history"][0]["error_type"] == "WindowsAccessViolation"
    assert preserved_again["attempt_history"][1]["error_type"] == "TimeoutExpired"
    assert failed_again_payload["previous_last_attempt_summary"]["error_type"] == "WindowsAccessViolation"
    assert len(failed_again_payload["attempt_history"]) == 2


def test_policy_eval_preserves_compatible_eval_after_failed_append_chunk(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"
    success_child = {
        "returncode": 0,
        "stdout_tail": [],
        "stderr_tail": [],
        "child": {
            "verified": True,
            "heldout_libero_policy_eval": True,
            "policy_loaded": True,
            "action_selected": True,
            "libero_steps_succeeded": True,
            "episodes": [{"episode_id": 0, "seed": 300, "success": False, "steps": 1}],
        },
    }
    failed_child = {
        "returncode": 3221225477,
        "stdout_tail": [],
        "stderr_tail": [],
        "child": {
            "verified": False,
            "heldout_libero_policy_eval": False,
            "failure_stage": "process_crash",
            "episodes": [],
        },
    }
    calls = iter([success_child, failed_child])
    monkeypatch.setattr(policy_eval, "_run_child", lambda *args, **kwargs: next(calls))

    completed = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[300],
        horizon=2,
        max_steps=1,
        append_existing=True,
    )
    preserved = run_modern_vla_libero_policy_eval(
        root,
        python_path=tmp_path / "python.exe",
        libero_source=tmp_path / "LIBERO",
        libero_config=tmp_path / ".libero",
        output_results_dir=results,
        seeds=[301],
        horizon=2,
        max_steps=1,
        append_existing=True,
    )

    current = json.loads((results / "modern_vla_libero_policy_eval.json").read_text(encoding="utf-8"))
    assert completed["verified"] is True
    assert preserved["verified"] is True
    assert preserved["eval_seeds"] == [300]
    assert preserved["latest_attempt_preserved_previous"] is True
    assert current["verified"] is True
    assert current["eval_seeds"] == [300]
    assert (results / "modern_vla_libero_policy_eval_last_attempt.json").exists()
