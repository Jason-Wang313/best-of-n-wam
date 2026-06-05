from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.real_robot_probe import (
    physical_trial_metric_artifacts,
    real_robot_hil_probe_markdown,
    run_real_robot_hil_probe,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_physical_trial_metric_artifacts_exclude_blocker_probes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_json(results / "real_robot_hil_probe.json", {"verified": True})
    write_json(results / "hardware_blocker_report.json", {"verified": True})
    write_json(results / "real_robot_trial_notes.json", {"note": "calibration only"})
    write_json(results / "real_robot_trial_metrics.json", {"success": [1, 0, 1]})

    artifacts = physical_trial_metric_artifacts(results)

    assert len(artifacts) == 1
    assert artifacts[0].endswith("real_robot_trial_metrics.json")


def test_real_robot_probe_redacts_env_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REAL_ROBOT_SECRET_TOKEN", "do-not-write-this")
    results = tmp_path / "results"

    payload = run_real_robot_hil_probe(tmp_path, output_results_dir=results, inspect_hardware=False)
    text = (results / "real_robot_hil_probe.json").read_text(encoding="utf-8")

    assert payload["verified"] is True
    assert payload["env_values_redacted"] is True
    assert payload["robot_env_names_present"]["REAL_ROBOT_SECRET_TOKEN"] is True
    assert "do-not-write-this" not in text
    assert payload["real_robot_or_hil_claim_ready"] is False


def test_real_robot_probe_markdown_is_claim_guarded() -> None:
    text = real_robot_hil_probe_markdown(
        {
            "verified": True,
            "possible_hardware_device_count": 2,
            "trial_metric_artifact_count": 0,
            "real_robot_or_hil_claim_ready": False,
            "packages": [{"name": "rclpy", "importable": False}],
            "commands": [{"name": "ros2", "available": False}],
        }
    )

    assert "does not support a real-robot/HIL claim" in text
    assert "available=`False`" in text
