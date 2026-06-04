import json
from pathlib import Path

from wam_inference_value.tracked_artifact_provenance import audit_tracked_artifact_provenance


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_fixture(root: Path) -> None:
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "results" / "source.json").write_text('{"metric": 1}', encoding="utf-8")
    (root / "results" / "tables" / "table.csv").write_text("seed,value\n1,2\n", encoding="utf-8")
    (root / "reports" / "source_report.md").write_text("# Report\n", encoding="utf-8")
    write_json(
        root / "results" / "claim_evidence_quality.json",
        {
            "source_records": [
                {"claim_id": 1, "raw_path": "results/source.json", "resolved_path": str(root / "results" / "source.json")},
                {"claim_id": 2, "raw_path": "reports/source_report.md", "resolved_path": str(root / "reports" / "source_report.md")},
            ]
        },
    )
    write_json(
        root / "results" / "artifact_integrity.json",
        {
            "references": [
                {
                    "source_json": "summary.json",
                    "json_path": "table_path",
                    "raw_path": "results/tables/table.csv",
                    "resolved_path": str(root / "results" / "tables" / "table.csv"),
                    "status": "ok",
                }
            ]
        },
    )


def test_tracked_artifact_provenance_accepts_tracked_sources(tmp_path: Path):
    write_fixture(tmp_path)
    tracked = {
        "results/source.json",
        "reports/source_report.md",
        "results/tables/table.csv",
    }

    payload = audit_tracked_artifact_provenance(tmp_path, tmp_path / "results", tracked_paths=tracked)

    assert payload["verified"] is True
    assert payload["n_issues"] == 0
    assert payload["n_claim_sources"] == 2
    assert payload["n_artifact_references"] == 1


def test_tracked_artifact_provenance_flags_untracked_sources(tmp_path: Path):
    write_fixture(tmp_path)
    tracked = {"results/source.json"}

    payload = audit_tracked_artifact_provenance(tmp_path, tmp_path / "results", tracked_paths=tracked)

    failures = {check["name"] for check in payload["issues"]}
    assert "claim_sources_tracked" in failures
    assert "artifact_references_tracked" in failures
    assert payload["n_untracked_claim_sources"] == 1
    assert payload["n_untracked_artifact_references"] == 1
