from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.ideal_frontier_blocker_audit import (
    BLOCKER_REQUIREMENTS,
    build_ideal_frontier_blocker_audit,
    ideal_frontier_blocker_markdown,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_text(path: Path, text: str = "# Report\n\nAttempted blocker evidence.\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def readiness_payload(*, detail: str = "concrete missing evidence detail") -> dict:
    rows = []
    for frontier_id in BLOCKER_REQUIREMENTS:
        rows.append(
            {
                "frontier_id": frontier_id,
                "ready_to_promote": False,
                "n_signals": 2,
                "n_met_signals": 1,
                "n_missing_signals": 1,
                "missing_signals": ["missing_evidence"],
                "signals": [
                    {"name": "present_evidence", "ok": True, "detail": "ok"},
                    {"name": "missing_evidence", "ok": False, "detail": detail},
                ],
                "next_action": f"Collect the missing ideal-frontier evidence for {frontier_id}.",
            }
        )
    return {
        "experiment": "ideal_frontier_readiness",
        "verified": True,
        "n_frontiers": len(rows),
        "n_ready_to_promote": 0,
        "rows": rows,
    }


def write_required_artifacts(root: Path) -> None:
    for spec in BLOCKER_REQUIREMENTS.values():
        for relative in spec["required_artifacts"]:
            path = root / relative
            if path.suffix == ".json":
                write_json(path, {"verified": True, "attempted": True})
            else:
                write_text(path)


def test_ideal_frontier_blocker_audit_accepts_documented_unpromoted_frontiers(tmp_path: Path) -> None:
    write_json(tmp_path / "results" / "ideal_frontier_readiness.json", readiness_payload())
    write_required_artifacts(tmp_path)

    payload = build_ideal_frontier_blocker_audit(tmp_path, tmp_path / "results")

    assert payload["verified"] is True
    assert payload["n_frontiers"] == len(BLOCKER_REQUIREMENTS)
    assert payload["n_ready_to_promote"] == 0
    assert {row["frontier_id"] for row in payload["blocker_rows"]} == set(BLOCKER_REQUIREMENTS)


def test_ideal_frontier_blocker_audit_rejects_missing_detail(tmp_path: Path) -> None:
    write_json(tmp_path / "results" / "ideal_frontier_readiness.json", readiness_payload(detail="short"))
    write_required_artifacts(tmp_path)

    payload = build_ideal_frontier_blocker_audit(tmp_path, tmp_path / "results")

    assert payload["verified"] is False
    assert any(issue["name"].endswith("_missing_signal_details_present") for issue in payload["issues"])


def test_ideal_frontier_blocker_audit_rejects_missing_artifact(tmp_path: Path) -> None:
    write_json(tmp_path / "results" / "ideal_frontier_readiness.json", readiness_payload())
    write_required_artifacts(tmp_path)
    (tmp_path / "results" / "universal_recipe_boundary.json").unlink()

    payload = build_ideal_frontier_blocker_audit(tmp_path, tmp_path / "results")

    assert payload["verified"] is False
    assert "universal_wam_training_recipe_blocker_artifacts_present" in {issue["name"] for issue in payload["issues"]}


def test_ideal_frontier_blocker_markdown_marks_scope_as_blocker_only(tmp_path: Path) -> None:
    write_json(tmp_path / "results" / "ideal_frontier_readiness.json", readiness_payload())
    write_required_artifacts(tmp_path)
    payload = build_ideal_frontier_blocker_audit(tmp_path, tmp_path / "results")

    text = ideal_frontier_blocker_markdown(payload)

    assert "blocker evidence, not validation evidence" in text
    assert "real_robot_hil" in text
