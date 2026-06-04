import json
from pathlib import Path

from wam_inference_value.evidence_hash_coverage import audit_evidence_hash_coverage


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_evidence_hash_coverage_hashes_claim_sources_and_artifact_refs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    report = tmp_path / "reports" / "claim.md"
    table = results / "tables" / "table.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# claim\n", encoding="utf-8")
    table.write_text("x\n1\n", encoding="utf-8")
    write_json(
        results / "claim_evidence_quality.json",
        {
            "source_records": [
                {"claim_id": 1, "raw_path": "reports/claim.md", "resolved_path": str(report)}
            ]
        },
    )
    write_json(
        results / "artifact_integrity.json",
        {
            "references": [
                {"source_json": "summary.json", "json_path": "table_path", "raw_path": "results/tables/table.csv", "resolved_path": str(table)}
            ]
        },
    )

    payload = audit_evidence_hash_coverage(tmp_path, results)

    assert payload["n_claim_sources"] == 1
    assert payload["n_artifact_references"] == 1
    hashes = payload["claim_source_hashes"] + payload["artifact_reference_hashes"]
    assert all(len(record["sha256"]) == 64 for record in hashes)


def test_evidence_hash_coverage_flags_missing_nonself_source(tmp_path: Path) -> None:
    results = tmp_path / "results"
    table = results / "tables" / "table.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("x\n1\n", encoding="utf-8")
    write_json(
        results / "claim_evidence_quality.json",
        {
            "source_records": [
                {"claim_id": 1, "raw_path": "reports/missing.md", "resolved_path": str(tmp_path / "reports" / "missing.md")}
            ]
        },
    )
    write_json(
        results / "artifact_integrity.json",
        {
            "references": [
                {"source_json": "summary.json", "json_path": "table_path", "raw_path": "results/tables/table.csv", "resolved_path": str(table)}
            ]
        },
    )

    payload = audit_evidence_hash_coverage(tmp_path, results)

    failures = {check["name"] for check in payload["issues"]}
    assert "all_nonself_records_exist" in failures
    assert "all_nonself_records_hashed" in failures


def test_evidence_hash_coverage_excludes_self_outputs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    table = results / "tables" / "table.csv"
    self_json = results / "evidence_hash_coverage.json"
    self_report = tmp_path / "reports" / "evidence_hash_coverage_report.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    self_report.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("x\n1\n", encoding="utf-8")
    self_json.write_text("{}", encoding="utf-8")
    self_report.write_text("# self\n", encoding="utf-8")
    write_json(
        results / "claim_evidence_quality.json",
        {
            "source_records": [
                {"claim_id": 120, "raw_path": "results/evidence_hash_coverage.json", "resolved_path": str(self_json)},
                {"claim_id": 120, "raw_path": "reports/evidence_hash_coverage_report.md", "resolved_path": str(self_report)},
            ]
        },
    )
    write_json(
        results / "artifact_integrity.json",
        {
            "references": [
                {"source_json": "summary.json", "json_path": "table_path", "raw_path": "results/tables/table.csv", "resolved_path": str(table)}
            ]
        },
    )

    payload = audit_evidence_hash_coverage(tmp_path, results)

    assert payload["n_claim_sources"] == 0
    assert payload["n_self_outputs_excluded"] == 2
    assert payload["artifact_reference_hashes"][0]["sha256"]
