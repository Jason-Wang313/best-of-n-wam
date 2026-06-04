from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wam_inference_value.ideal_frontier_readiness import audit_ideal_frontier_readiness


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ideal_frontier_readiness_keeps_future_frontiers_unpromoted(tmp_path: Path) -> None:
    results = tmp_path / "results"
    model_path = tmp_path / "results" / "models" / "libero_vl.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, x=np.ones((2, 2)))

    write_json(
        results / "benchmark_libero_visual_language_bc_policy.json",
        {
            "verified": True,
            "eval_episodes": 10,
            "confidence_intervals": {"eval_success_rate": {"n": 10, "mean": 1.0, "lo": 1.0, "hi": 1.0}},
            "model_path": "results/models/libero_vl.npz",
            "policy": {
                "type": "rgb_proprio_language_knn_behavior_cloning",
                "uses_rgb": True,
                "uses_language": True,
                "uses_simulator_object_state": False,
                "uses_task_id": False,
                "uses_phase_index": False,
                "uses_target_point_command": False,
            },
        },
    )
    write_json(
        results / "benchmark_robocasa_catalog_probe.json",
        {
            "verified": True,
            "registry_count": 4,
            "verified_artifact_task_count": 2,
            "any_artifact_task_count": 3,
            "category_counts": [{"category": "pick_place", "registered": 4, "any_artifact_covered": 3}],
        },
    )
    write_json(
        results / "benchmark_maniskill_visual_probe.json",
        {
            "attempted": True,
            "any_visual_success": False,
            "visual_attempt_count": 10,
            "any_ee_control_success": False,
            "ee_control_attempt_count": 2,
        },
    )
    write_json(results / "benchmark_maniskill_dependency_probe.json", {"pinocchio_import_available": False})
    write_json(results / "publication_scope.json", {"verified": True})
    write_json(results / "external_benchmark_runtime_probe.json", {"verified": True, "libero_import_available": True})

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    assert payload["verified"] is True
    assert payload["n_frontiers"] == 5
    assert payload["n_ready_to_promote"] == 0
    by_id = {row["frontier_id"]: row for row in payload["rows"]}
    modern = by_id["modern_vla_libero"]
    assert "neural_visual_language_model_class" in modern["missing_signals"]
    assert "modern_vla_scale_or_pretrained_model" in modern["missing_signals"]
    assert "current_runtime_can_rerun_libero" not in modern["missing_signals"]
    assert modern["n_met_signals"] >= 5
    assert "full_registry_rollout_pool_coverage" in by_id["full_robocasa_wide"]["missing_signals"]


def test_ideal_frontier_readiness_accepts_neural_libero_policy_class(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    model = tmp_path / "results" / "models" / "libero_neural.npz"
    model.parent.mkdir()
    model.write_bytes(b"model")
    write_json(
        results / "benchmark_libero_visual_language_bc_policy.json",
        {
            "verified": True,
            "eval_episodes": 3,
            "model_path": "results/models/libero_neural.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 3}},
            "policy": {
                "type": "tiny_neural_vla_behavior_cloning",
                "is_neural": True,
                "uses_rgb": True,
                "uses_language": True,
                "uses_robot_proprio": True,
                "uses_simulator_object_state": False,
                "uses_task_id": False,
                "uses_phase_index": False,
                "uses_target_point_command": False,
            },
        },
    )
    write_json(results / "external_benchmark_runtime_probe.json", {"verified": True, "libero_import_available": True})

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    assert modern["ready_to_promote"] is False
    assert "neural_visual_language_model_class" not in modern["missing_signals"]
    assert modern["missing_signals"] == ["modern_vla_scale_or_pretrained_model"]


def test_ideal_frontier_readiness_accepts_auxiliary_neural_smoke(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    canonical_model = tmp_path / "results" / "models" / "libero_knn.npz"
    neural_model = tmp_path / "results" / "models" / "libero_neural_smoke.npz"
    canonical_model.parent.mkdir()
    canonical_model.write_bytes(b"knn")
    neural_model.write_bytes(b"neural")
    write_json(
        results / "benchmark_libero_visual_language_bc_policy.json",
        {
            "verified": True,
            "eval_episodes": 30,
            "model_path": "results/models/libero_knn.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 30, "mean": 1.0, "lo": 1.0}},
            "policy": {
                "type": "rgb_proprio_language_knn_behavior_cloning",
                "is_neural": False,
                "uses_rgb": True,
                "uses_language": True,
                "uses_robot_proprio": True,
                "uses_simulator_object_state": False,
                "uses_task_id": False,
                "uses_phase_index": False,
                "uses_target_point_command": False,
            },
        },
    )
    write_json(
        results / "benchmark_libero_tiny_neural_vla_policy.json",
        {
            "verified": True,
            "eval_episodes": 1,
            "model_path": "results/models/libero_neural_smoke.npz",
            "policy": {
                "type": "tiny_neural_vla_behavior_cloning",
                "is_neural": True,
                "pretrained_vla": False,
                "vla_scale_parameters": 0,
                "uses_rgb": True,
                "uses_language": True,
                "uses_robot_proprio": True,
                "uses_simulator_object_state": False,
                "uses_task_id": False,
                "uses_phase_index": False,
                "uses_target_point_command": False,
            },
        },
    )
    write_json(results / "external_benchmark_runtime_probe.json", {"verified": True, "libero_import_available": True})

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    assert modern["ready_to_promote"] is False
    assert "neural_visual_language_model_class" not in modern["missing_signals"]
    assert modern["missing_signals"] == ["modern_vla_scale_or_pretrained_model"]
