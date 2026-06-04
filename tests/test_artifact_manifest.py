from pathlib import Path

from wam_inference_value.artifact_manifest import audit_artifact_manifest_payload, build_artifact_manifest


def test_artifact_manifest_hashes_scientific_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    (results / "tables").mkdir(parents=True)
    (results / "tables" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (results / "summary.json").write_text('{"metric": 1}', encoding="utf-8")
    (results / "artifact_integrity.json").write_text("{}", encoding="utf-8")
    (results / "smoke").mkdir()
    (results / "smoke" / "temporary.json").write_text("{}", encoding="utf-8")

    manifest = build_artifact_manifest(tmp_path, results)
    paths = {record["path"] for record in manifest["records"]}

    assert paths == {"results/summary.json", "results/tables/table.csv"}
    assert manifest["counts_by_suffix"] == {".json": 1, ".csv": 1}


def test_artifact_manifest_detects_tampered_file(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    target = results / "summary.json"
    target.write_text('{"metric": 1}', encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, results)
    target.write_text('{"metric": 2}', encoding="utf-8")

    audit = audit_artifact_manifest_payload(manifest, root=tmp_path, results_dir=results)

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "manifest_hashes_match_current_files" in issue_names
    assert audit["hash_mismatches"] == ["results/summary.json"]


def test_artifact_manifest_detects_stale_record(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    target = results / "summary.json"
    target.write_text('{"metric": 1}', encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, results)
    target.unlink()

    audit = audit_artifact_manifest_payload(manifest, root=tmp_path, results_dir=results)

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "manifest_has_no_stale_artifacts" in issue_names
