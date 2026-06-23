import json
from pathlib import Path

from wam_inference_value.repo_bound_artifact_audit import audit_repo_bound_artifacts


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_repo_bound_artifact_audit_classifies_inside_repo_records(tmp_path: Path) -> None:
    results = tmp_path / "results"
    report = tmp_path / "reports" / "claim.md"
    table = results / "tables" / "table.csv"
    figure = results / "figures" / "plot.png"
    model = results / "models" / "model.npz"
    for path, text in [(report, "# claim\n"), (table, "x\n1\n"), (figure, "png"), (model, "model")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    source_json = results / "summary.json"
    write_json(source_json, {"metric": 1})
    write_json(
        results / "claim_evidence_quality.json",
        {"source_records": [{"claim_id": 1, "raw_path": "reports/claim.md", "resolved_path": str(report)}]},
    )
    write_json(
        results / "artifact_integrity.json",
        {
            "references": [
                {"source_json": "summary.json", "json_path": "table_path", "raw_path": "results/tables/table.csv", "resolved_path": str(table)},
                {"source_json": "summary.json", "json_path": "figure_path", "raw_path": "results/figures/plot.png", "resolved_path": str(figure)},
                {"source_json": "summary.json", "json_path": "model_path", "raw_path": "results/models/model.npz", "resolved_path": str(model)},
                {"source_json": "summary.json", "json_path": "artifact_path", "raw_path": "results/summary.json", "resolved_path": str(source_json)},
            ]
        },
    )

    payload = audit_repo_bound_artifacts(tmp_path, results)

    assert payload["n_outside_repo"] == 0
    assert payload["n_missing"] == 0
    assert payload["artifact_categories"]["report"] == 1
    assert payload["artifact_categories"]["result_table"] == 1
    assert payload["artifact_categories"]["figure"] == 1
    assert payload["artifact_categories"]["model"] == 1
    assert payload["artifact_categories"]["result_json"] == 1


def test_repo_bound_artifact_audit_flags_outside_repo_record(tmp_path: Path) -> None:
    results = tmp_path / "results"
    external = tmp_path.parent / "external.csv"
    external.write_text("x\n1\n", encoding="utf-8")
    write_json(
        results / "claim_evidence_quality.json",
        {"source_records": [{"claim_id": 1, "raw_path": str(external), "resolved_path": str(external)}]},
    )
    write_json(results / "artifact_integrity.json", {"references": []})

    payload = audit_repo_bound_artifacts(tmp_path, results)

    failures = {check["name"] for check in payload["issues"]}
    assert "all_records_inside_repo" in failures
    assert payload["n_outside_repo"] == 1


def test_repo_bound_artifact_audit_reanchors_relocated_worktree_paths(tmp_path: Path) -> None:
    active = tmp_path / "active"
    results = active / "results"
    active_table = results / "tables" / "table.csv"
    active_table.parent.mkdir(parents=True, exist_ok=True)
    active_table.write_text("x\n1\n", encoding="utf-8")
    other_checkout = tmp_path / "other_checkout"
    stale_absolute = other_checkout / "results" / "tables" / "table.csv"
    write_json(results / "claim_evidence_quality.json", {"source_records": []})
    write_json(
        results / "artifact_integrity.json",
        {
            "references": [
                {
                    "source_json": "summary.json",
                    "json_path": "table_path",
                    "raw_path": str(stale_absolute),
                    "resolved_path": str(stale_absolute),
                }
            ]
        },
    )

    payload = audit_repo_bound_artifacts(active, results)

    assert payload["n_outside_repo"] == 0
    assert payload["records"][0]["repo_relative"] == "results/tables/table.csv"
    assert payload["records"][0]["artifact_category"] == "result_table"


def test_repo_bound_artifact_audit_flags_parent_traversal(tmp_path: Path) -> None:
    results = tmp_path / "results"
    inside = results / "source.json"
    write_json(inside, {"ok": True})
    write_json(
        results / "claim_evidence_quality.json",
        {"source_records": [{"claim_id": 1, "raw_path": "results/../results/source.json", "resolved_path": str(inside)}]},
    )
    write_json(results / "artifact_integrity.json", {"references": []})

    payload = audit_repo_bound_artifacts(tmp_path, results)

    failures = {check["name"] for check in payload["issues"]}
    assert "no_parent_traversal_refs" in failures
    assert payload["n_parent_traversal"] == 1
