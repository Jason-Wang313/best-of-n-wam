from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.artifact_integrity import audit_result_artifacts, collect_artifact_references


def test_collect_artifact_references_ignores_free_text_evidence(tmp_path: Path) -> None:
    payload = {
        "evidence": "model=C:\\Users\\wangz\\best-of-n-wam\\results\\models\\missing.npz",
        "model_path": "results/models/model.npz",
        "artifacts": {"table": "results/tables/table.csv"},
    }

    refs = collect_artifact_references(tmp_path / "summary.json", payload)

    assert {ref.json_path for ref in refs} == {"model_path", "artifacts.table"}
    assert all("evidence" not in ref.json_path for ref in refs)


def test_audit_result_artifacts_validates_csv_rows(tmp_path: Path) -> None:
    results = tmp_path / "results"
    table_dir = results / "tables"
    model_dir = results / "models"
    table_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (table_dir / "ok.csv").write_text("seed,value\n1,2\n", encoding="utf-8")
    (model_dir / "model.npz").write_bytes(b"model")
    (results / "summary.json").write_text(
        json.dumps(
            {
                "model_path": "results/models/model.npz",
                "artifacts": {"table": "results/tables/ok.csv"},
            }
        ),
        encoding="utf-8",
    )

    payload = audit_result_artifacts(tmp_path, results)

    assert payload["verified"] is True
    assert payload["n_references"] == 2
    assert payload["n_issues"] == 0
    row_counts = {record["json_path"]: record["rows"] for record in payload["references"]}
    assert row_counts["artifacts.table"] == 1


def test_audit_result_artifacts_flags_missing_and_zero_row_csv(tmp_path: Path) -> None:
    results = tmp_path / "results"
    table_dir = results / "tables"
    table_dir.mkdir(parents=True)
    (table_dir / "empty_rows.csv").write_text("seed,value\n", encoding="utf-8")
    (results / "summary.json").write_text(
        json.dumps(
            {
                "model_path": "results/models/missing.npz",
                "artifacts": {"table": "results/tables/empty_rows.csv"},
            }
        ),
        encoding="utf-8",
    )

    payload = audit_result_artifacts(tmp_path, results)

    assert payload["verified"] is False
    assert payload["status_counts"]["missing"] == 1
    assert payload["status_counts"]["zero_row_csv"] == 1


def test_audit_result_artifacts_skips_own_output(tmp_path: Path) -> None:
    results = tmp_path / "results"
    table_dir = results / "tables"
    table_dir.mkdir(parents=True)
    (table_dir / "ok.csv").write_text("seed,value\n1,2\n", encoding="utf-8")
    (results / "summary.json").write_text(
        json.dumps({"artifact": "results/tables/ok.csv"}),
        encoding="utf-8",
    )
    (results / "artifact_integrity.json").write_text(
        json.dumps({"raw_path": "results/tables/does_not_exist.csv"}),
        encoding="utf-8",
    )

    payload = audit_result_artifacts(tmp_path, results)

    assert payload["verified"] is True
    assert payload["n_references"] == 1
