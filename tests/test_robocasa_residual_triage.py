from __future__ import annotations

import csv
import json
from pathlib import Path

from wam_inference_value.robocasa_residual_triage import build_robocasa_residual_triage


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_robocasa_residual_triage_separates_failure_modes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_json(
        results / "benchmark_robocasa_catalog_probe.json",
        {
            "verified_artifact_env_ids": ["robocasa/CoveredPool"],
            "micro_rollout_env_ids": ["robocasa/CoveredMicro"],
        },
    )
    write_csv(
        results / "tables" / "benchmark_robocasa_catalog_registry.csv",
        [
            {"env_id": "robocasa/CoveredPool", "category": "pick_place"},
            {"env_id": "robocasa/CoveredMicro", "category": "open"},
            {"env_id": "robocasa/OpenDoor", "category": "open"},
            {"env_id": "robocasa/OpenFixture", "category": "open"},
            {"env_id": "robocasa/SlowTask", "category": "long_horizon_or_compositional"},
            {"env_id": "robocasa/UntouchedTask", "category": "cooking"},
        ],
    )
    write_csv(
        results / "tables" / "benchmark_robocasa_micro_rollout_probe.csv",
        [
            {
                "env_id": "robocasa/OpenDoor",
                "reset_ok": False,
                "rollout_ok": False,
                "nondegenerate": False,
                "seconds": 0.01,
                "error": "TypeError: ManipulateDoor.__init__() missing 1 required positional argument: 'fixture_id'",
            }
        ],
    )
    write_csv(
        results / "tables" / "benchmark_robocasa_micro_rollout_zero_progress_probe.csv",
        [
            {
                "env_id": "robocasa/OpenFixture",
                "reset_ok": True,
                "rollout_ok": True,
                "nondegenerate": False,
                "initial_distance": 0.0,
                "mean_progress": 0.0,
                "utility_std": 0.2,
                "utility_min": -0.3,
                "utility_max": -0.1,
                "seconds": 0.02,
                "error": "",
            }
        ],
    )
    write_json(
        results / "benchmark_robocasa_micro_rollout_timeout_probe.json",
        {
            "timed_out": True,
            "env_ids": ["robocasa/SlowTask"],
            "wall_clock_seconds": 120.0,
        },
    )

    payload = build_robocasa_residual_triage(tmp_path, results)

    assert payload["verified"] is True
    assert payload["status_counts"]["rollout_pool_covered"] == 1
    assert payload["status_counts"]["micro_nondegenerate_covered"] == 1
    assert payload["status_counts"]["constructor_signature_failure"] == 1
    assert payload["status_counts"]["zero_distance_no_progress"] == 1
    assert payload["status_counts"]["timed_out"] == 1
    assert payload["status_counts"]["unattempted"] == 1
    assert payload["unattempted_by_category"] == {"cooking": 1}
