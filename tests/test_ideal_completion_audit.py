from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.ideal_completion_audit import audit_ideal_completion


def write_boundary(results: Path, *, false_supported_future: bool = False) -> None:
    rows = [
        {
            "id": "supported_math",
            "ideal_claim": "Supported theorem result.",
            "endpoint_supported": True,
            "promotable": True,
            "future_only": False,
            "paper_status": "promotable_result",
        },
        {
            "id": "real_robot_hil",
            "ideal_claim": "Real robot validation.",
            "endpoint_supported": false_supported_future,
            "promotable": false_supported_future,
            "future_only": True,
            "paper_status": "future_only_not_promotable",
            "limitation": "No real robot artifacts exist.",
            "promotion_requirements": ["real logs", "claim evidence"],
            "missing_evidence_classes": ["hardware trials", "real success metrics"],
            "gap_evidence_files": [{"path": "reports/final_decision_report.md", "exists": True, "bytes": 1}],
            "missing_gap_evidence_files": [],
        },
    ]
    results.mkdir(parents=True, exist_ok=True)
    (results / "ideal_claim_boundary.json").write_text(
        json.dumps(
            {
                "verified": True,
                "goal_completion_status": "incomplete_future_only_gaps_remain",
                "all_ideal_claims_promotable": False,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_ideal_completion_audit_reports_incomplete_future_blockers(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_boundary(results)

    payload = audit_ideal_completion(tmp_path, results)

    assert payload["verified"] is True
    assert payload["completion_verdict"] == "not_complete"
    assert payload["all_ideal_endpoints_supported"] is False
    assert payload["n_supported_endpoints"] == 1
    assert payload["n_future_blockers"] == 1
    assert payload["future_blocker_ids"] == ["real_robot_hil"]


def test_ideal_completion_audit_rejects_future_row_counted_as_supported(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_boundary(results, false_supported_future=True)

    payload = audit_ideal_completion(tmp_path, results)

    assert payload["verified"] is False
    assert "future_only_rows_not_counted_as_supported" in {issue["name"] for issue in payload["issues"]}


def test_ideal_completion_audit_accepts_complete_supported_rows(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "ideal_claim_boundary.json").write_text(
        json.dumps(
            {
                "verified": True,
                "goal_completion_status": "complete_all_ideal_endpoints_supported",
                "all_ideal_claims_promotable": True,
                "rows": [
                    {
                        "id": "supported_math",
                        "ideal_claim": "Supported theorem result.",
                        "endpoint_supported": True,
                        "promotable": True,
                        "future_only": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = audit_ideal_completion(tmp_path, results)

    assert payload["verified"] is True
    assert payload["completion_verdict"] == "complete"
    assert payload["all_ideal_endpoints_supported"] is True
