from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wam_inference_value.ideal_frontier_readiness import MODERN_VLA_MIN_PARAMETERS, audit_ideal_frontier_readiness


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
    write_json(
        results / "benchmark_maniskill_dependency_probe.json",
        {
            "pinocchio_import_available": True,
            "pinocchio_api_available": False,
            "pinocchio_missing_symbols": ["Model", "GeometryModel", "buildModelFromUrdf"],
        },
    )
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


def test_ideal_frontier_readiness_reports_robocasa_residual_attempts(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "benchmark_robocasa_catalog_probe.json",
        {
            "verified": True,
            "registry_count": 4,
            "verified_artifact_task_count": 1,
            "any_artifact_task_count": 2,
            "category_counts": [
                {"category": "close", "registered": 2, "any_artifact_covered": 1},
                {"category": "long_horizon_or_compositional", "registered": 2, "any_artifact_covered": 1},
            ],
        },
    )
    write_json(
        results / "benchmark_robocasa_residual_frontier_sweep_quick_close.json",
        {
            "attempted": True,
            "candidate_task_count": 1,
            "completed_chunk_count": 0,
            "timed_out_chunk_count": 1,
            "nondegenerate_task_count": 0,
            "categories": ["close"],
        },
    )
    write_json(
        results / "benchmark_robocasa_residual_frontier_sweep_quick_gap.json",
        {
            "attempted": True,
            "candidate_task_count": 1,
            "completed_chunk_count": 0,
            "timed_out_chunk_count": 1,
            "nondegenerate_task_count": 0,
            "categories": ["long_horizon_or_compositional"],
        },
    )

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    robocasa = {row["frontier_id"]: row for row in payload["rows"]}["full_robocasa_wide"]
    any_signal = next(signal for signal in robocasa["signals"] if signal["name"] == "full_registry_any_artifact_coverage")
    assert any_signal["ok"] is False
    assert "residual_attempts=2" in any_signal["detail"]
    assert "residual_timeouts=2" in any_signal["detail"]
    assert "quick_close" in any_signal["detail"]
    assert "quick_gap" in any_signal["detail"]


def test_ideal_frontier_readiness_reports_modern_vla_availability_probe(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "modern_vla_availability_probe.json",
        {
            "verified": True,
            "vla_package_importable": False,
            "local_vla_like_count": 0,
            "hf_reachable_count": 3,
            "ready_for_policy_eval": False,
        },
    )

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    scale_signal = next(signal for signal in modern["signals"] if signal["name"] == "modern_vla_scale_or_pretrained_model")
    assert scale_signal["ok"] is False
    assert "availability_probe_verified=True" in scale_signal["detail"]
    assert "ready_for_policy_eval=False" in scale_signal["detail"]


def test_ideal_frontier_readiness_counts_verified_smolvla_execution_as_scale_prerequisite(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "benchmark_libero_visual_language_bc_policy.json",
        {
            "verified": True,
            "eval_episodes": 3,
            "confidence_intervals": {"eval_success_rate": {"n": 3, "mean": 1.0}},
            "model_path": "results/models/libero_vl.npz",
            "policy": {
                "type": "rgb_proprio_language_knn_behavior_cloning",
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
    model_path = tmp_path / "results" / "models" / "libero_vl.npz"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    write_json(
        results / "benchmark_libero_tiny_neural_vla_policy.json",
        {
            "verified": True,
            "eval_episodes": 1,
            "eval_successes": 1,
            "train_seeds": [1],
            "eval_seeds": [2],
            "model_path": "results/models/libero_vl.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 1.0}},
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
    write_json(
        results / "modern_vla_libero_execution_probe.json",
        {
            "attempted": True,
            "verified": True,
            "policy_loaded": True,
            "action_selected": True,
            "libero_step_succeeded": True,
            "model_id": "HuggingFaceVLA/smolvla_libero",
            "parameter_count": MODERN_VLA_MIN_PARAMETERS,
            "heldout_libero_policy_eval": False,
        },
    )

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    scale_signal = next(signal for signal in modern["signals"] if signal["name"] == "modern_vla_scale_or_pretrained_model")
    heldout_signal = next(signal for signal in modern["signals"] if signal["name"] == "heldout_sparse_success_modern_vla_eval")
    assert scale_signal["ok"] is True
    assert "execution_model=HuggingFaceVLA/smolvla_libero" in scale_signal["detail"]
    assert heldout_signal["ok"] is False
    assert modern["missing_signals"] == ["heldout_sparse_success_modern_vla_eval"]


def test_ideal_frontier_readiness_does_not_count_real_robot_probe_as_trial(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "real_robot_hil_probe.json",
        {
            "verified": True,
            "possible_hardware_device_count": 2,
            "trial_metric_artifact_count": 0,
            "real_robot_or_hil_claim_ready": False,
        },
    )

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    real_robot = {row["frontier_id"]: row for row in payload["rows"]}["real_robot_hil"]
    artifact_signal = next(signal for signal in real_robot["signals"] if signal["name"] == "real_robot_or_hil_artifact_present")
    metric_signal = next(signal for signal in real_robot["signals"] if signal["name"] == "real_world_success_metrics_present")
    assert artifact_signal["ok"] is False
    assert metric_signal["ok"] is False
    assert "availability_probe_verified=True" in artifact_signal["detail"]


def test_ideal_frontier_readiness_counts_explicit_real_robot_trial_metrics(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_json(results / "real_robot_trial_metrics.json", {"success": [1, 1]})

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    real_robot = {row["frontier_id"]: row for row in payload["rows"]}["real_robot_hil"]
    assert "real_robot_or_hil_artifact_present" not in real_robot["missing_signals"]
    assert "real_world_success_metrics_present" not in real_robot["missing_signals"]


def test_ideal_frontier_readiness_reports_universal_boundary(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "universal_recipe_boundary.json",
        {
            "verified": True,
            "result_type": "no_free_lunch_boundary",
            "randomized_worst_case_regret_lower_bound": 0.5,
        },
    )

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    universal = {row["frontier_id"]: row for row in payload["rows"]}["universal_wam_training_recipe"]
    signal = next(signal for signal in universal["signals"] if signal["name"] == "universal_generalization_proof_present")
    assert signal["ok"] is False
    assert "boundary_verified=True" in signal["detail"]
    assert "boundary_regret_lb=0.5" in signal["detail"]


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
            "eval_successes": 3,
            "train_seeds": [1, 2, 3],
            "eval_seeds": [4, 5, 6],
            "model_path": "results/models/libero_neural.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 3, "mean": 1.0}},
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
    assert modern["missing_signals"] == [
        "modern_vla_scale_or_pretrained_model",
        "heldout_sparse_success_modern_vla_eval",
    ]


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
            "eval_successes": 1,
            "train_seeds": [1],
            "eval_seeds": [2],
            "model_path": "results/models/libero_neural_smoke.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 1.0}},
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
    assert modern["missing_signals"] == [
        "modern_vla_scale_or_pretrained_model",
        "heldout_sparse_success_modern_vla_eval",
    ]


def test_ideal_frontier_readiness_scans_tagged_auxiliary_neural_smokes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    model_dir = tmp_path / "results" / "models"
    model_dir.mkdir()
    (model_dir / "libero_knn.npz").write_bytes(b"knn")
    (model_dir / "failed_neural.npz").write_bytes(b"failed")
    (model_dir / "successful_neural.npz").write_bytes(b"success")
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
    base_neural_policy = {
        "type": "distilled_tiny_neural_vla_behavior_cloning",
        "is_neural": True,
        "pretrained_vla": False,
        "vla_scale_parameters": 607751,
        "uses_rgb": True,
        "uses_language": True,
        "uses_robot_proprio": True,
        "uses_simulator_object_state": False,
        "uses_task_id": False,
        "uses_phase_index": False,
        "uses_target_point_command": False,
    }
    write_json(
        results / "benchmark_libero_tiny_neural_failed_probe.json",
        {
            "verified": False,
            "eval_episodes": 1,
            "eval_successes": 0,
            "train_seeds": [1],
            "eval_seeds": [2],
            "model_path": "results/models/failed_neural.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 0.0}},
            "policy": base_neural_policy,
        },
    )
    write_json(
        results / "benchmark_libero_tiny_neural_success_probe.json",
        {
            "verified": True,
            "eval_episodes": 1,
            "eval_successes": 1,
            "train_seeds": [1],
            "eval_seeds": [3],
            "model_path": "results/models/successful_neural.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 1.0}},
            "policy": base_neural_policy,
        },
    )
    write_json(results / "external_benchmark_runtime_probe.json", {"verified": True, "libero_import_available": True})

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    neural_signal = next(signal for signal in modern["signals"] if signal["name"] == "neural_visual_language_model_class")
    assert neural_signal["ok"] is True
    assert "aux_candidates=2" in neural_signal["detail"]
    assert "benchmark_libero_tiny_neural_success_probe.json" in neural_signal["detail"]
    assert modern["missing_signals"] == [
        "modern_vla_scale_or_pretrained_model",
        "heldout_sparse_success_modern_vla_eval",
    ]


def test_ideal_frontier_readiness_rejects_subscale_neural_smoke_as_modern_vla(tmp_path: Path) -> None:
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
            "eval_successes": 1,
            "train_seeds": [1],
            "eval_seeds": [2],
            "model_path": "results/models/libero_neural_smoke.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 1.0}},
            "policy": {
                "type": "tiny_neural_vla_behavior_cloning",
                "is_neural": True,
                "pretrained_vla": False,
                "vla_scale_parameters": MODERN_VLA_MIN_PARAMETERS - 1,
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
    assert "neural_visual_language_model_class" not in modern["missing_signals"]
    assert "modern_vla_scale_or_pretrained_model" in modern["missing_signals"]


def test_ideal_frontier_readiness_rejects_zero_success_neural_smoke(tmp_path: Path) -> None:
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
            "eval_successes": 30,
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
            "eval_successes": 0,
            "eval_success_rate": 0.0,
            "train_seeds": [1],
            "eval_seeds": [2],
            "model_path": "results/models/libero_neural_smoke.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 0.0}},
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
    assert "neural_visual_language_model_class" in modern["missing_signals"]


def test_ideal_frontier_readiness_rejects_nonheldout_neural_smoke(tmp_path: Path) -> None:
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
            "eval_successes": 30,
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
            "eval_successes": 1,
            "eval_success_rate": 1.0,
            "train_seeds": [1],
            "eval_seeds": [1],
            "model_path": "results/models/libero_neural_smoke.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 1.0}},
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
    assert "neural_visual_language_model_class" in modern["missing_signals"]


def test_ideal_frontier_readiness_accepts_auxiliary_rbf_neural_smoke(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    canonical_model = tmp_path / "results" / "models" / "libero_knn.npz"
    neural_model = tmp_path / "results" / "models" / "libero_rbf_neural.npz"
    canonical_model.parent.mkdir()
    canonical_model.write_bytes(b"knn")
    neural_model.write_bytes(b"rbf-neural")
    write_json(
        results / "benchmark_libero_visual_language_bc_policy.json",
        {
            "verified": True,
            "eval_episodes": 30,
            "eval_successes": 30,
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
        results / "benchmark_libero_tiny_neural_rbf_vla_policy.json",
        {
            "verified": True,
            "eval_episodes": 1,
            "eval_successes": 1,
            "eval_success_rate": 1.0,
            "train_seeds": [100, 101],
            "eval_seeds": [200],
            "model_path": "results/models/libero_rbf_neural.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 1.0}},
            "policy": {
                "type": "rbf_neural_vla_behavior_cloning",
                "is_neural": True,
                "pretrained_vla": False,
                "vla_scale_parameters": 846286,
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
    assert "neural_visual_language_model_class" not in modern["missing_signals"]
    assert "modern_vla_scale_or_pretrained_model" in modern["missing_signals"]


def test_ideal_frontier_readiness_requires_separate_positive_modern_vla_policy_eval(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    model = tmp_path / "results" / "models" / "libero_knn.npz"
    model.parent.mkdir()
    model.write_bytes(b"model")
    write_json(
        results / "benchmark_libero_visual_language_bc_policy.json",
        {
            "verified": True,
            "eval_episodes": 30,
            "eval_successes": 30,
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
            "eval_successes": 1,
            "train_seeds": [1],
            "eval_seeds": [2],
            "model_path": "results/models/libero_knn.npz",
            "confidence_intervals": {"eval_success_rate": {"n": 1, "mean": 1.0}},
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
    write_json(
        results / "modern_vla_libero_execution_probe.json",
        {
            "attempted": True,
            "verified": True,
            "policy_loaded": True,
            "action_selected": True,
            "libero_step_succeeded": True,
            "model_id": "HuggingFaceVLA/smolvla_libero",
            "parameter_count": MODERN_VLA_MIN_PARAMETERS,
        },
    )
    write_json(
        results / "modern_vla_libero_policy_eval.json",
        {
            "verified": True,
            "heldout_libero_policy_eval": True,
            "eval_episodes": 5,
            "n_eval_seeds": 5,
            "max_steps": 5,
            "eval_successes": 1,
            "eval_success_rate": 0.2,
            "success_ci": {"n": 5, "mean": 0.2, "lo": 0.04, "hi": 0.62, "method": "wilson"},
        },
    )
    write_json(results / "external_benchmark_runtime_probe.json", {"verified": True, "libero_import_available": True})

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    assert modern["ready_to_promote"] is True
    assert modern["missing_signals"] == []


def test_ideal_frontier_readiness_does_not_promote_zero_success_modern_vla_eval(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    model = tmp_path / "results" / "models" / "libero_knn.npz"
    model.parent.mkdir()
    model.write_bytes(b"model")
    write_json(
        results / "benchmark_libero_visual_language_bc_policy.json",
        {
            "verified": True,
            "eval_episodes": 30,
            "eval_successes": 30,
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
        results / "modern_vla_libero_execution_probe.json",
        {
            "attempted": True,
            "verified": True,
            "policy_loaded": True,
            "action_selected": True,
            "libero_step_succeeded": True,
            "model_id": "HuggingFaceVLA/smolvla_libero",
            "parameter_count": MODERN_VLA_MIN_PARAMETERS,
        },
    )
    write_json(
        results / "modern_vla_libero_policy_eval.json",
        {
            "verified": True,
            "heldout_libero_policy_eval": True,
            "eval_episodes": 5,
            "n_eval_seeds": 5,
            "max_steps": 5,
            "eval_successes": 0,
            "eval_success_rate": 0.0,
            "success_ci": {"n": 5, "mean": 0.0, "lo": 0.0, "hi": 0.43, "method": "wilson"},
        },
    )
    write_json(results / "external_benchmark_runtime_probe.json", {"verified": True, "libero_import_available": True})

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    heldout_signal = next(signal for signal in modern["signals"] if signal["name"] == "heldout_sparse_success_modern_vla_eval")
    assert heldout_signal["ok"] is False
    assert "eval_successes=0" in heldout_signal["detail"]


def test_ideal_frontier_readiness_reports_failed_modern_vla_last_attempt(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "modern_vla_libero_policy_eval.json",
        {
            "verified": True,
            "heldout_libero_policy_eval": True,
            "eval_episodes": 1,
            "n_eval_seeds": 1,
            "max_steps": 1,
            "eval_successes": 0,
            "eval_success_rate": 0.0,
            "success_ci": {"n": 1, "mean": 0.0, "lo": 0.0, "hi": 0.79, "method": "wilson"},
        },
    )
    write_json(
        results / "modern_vla_libero_policy_eval_last_attempt.json",
        {
            "verified": False,
            "horizon": 5,
            "max_steps": 5,
            "requested_eval_seeds": [300],
            "failure_stage": "process_crash",
            "error_type": "WindowsAccessViolation",
            "child_returncode": 3221225477,
            "attempt_history": [
                {"failure_stage": "process_crash", "error_type": "WindowsAccessViolation"},
                {"failure_stage": "timeout", "error_type": "TimeoutExpired"},
            ],
        },
    )

    payload = audit_ideal_frontier_readiness(tmp_path, results)

    modern = {row["frontier_id"]: row for row in payload["rows"]}["modern_vla_libero"]
    heldout_signal = next(signal for signal in modern["signals"] if signal["name"] == "heldout_sparse_success_modern_vla_eval")
    assert heldout_signal["ok"] is False
    assert "last_attempt_present=True" in heldout_signal["detail"]
    assert "last_attempt_max_steps=5" in heldout_signal["detail"]
    assert "last_attempt_failure_stage=process_crash" in heldout_signal["detail"]
    assert "last_attempt_error_type=WindowsAccessViolation" in heldout_signal["detail"]
    assert "attempt_history_count=2" in heldout_signal["detail"]
    assert "TimeoutExpired" in heldout_signal["detail"]
