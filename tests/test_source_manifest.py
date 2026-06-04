from pathlib import Path

from wam_inference_value.source_manifest import audit_source_manifest_payload, build_source_manifest


def test_source_manifest_hashes_source_files(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "generated.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")

    manifest = build_source_manifest(tmp_path)
    paths = {record["path"] for record in manifest["records"]}

    assert "src/pkg/module.py" in paths
    assert "scripts/run.sh" in paths
    assert "README.md" in paths
    assert "results/generated.json" not in paths


def test_source_manifest_detects_tampered_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    manifest = build_source_manifest(tmp_path)
    target.write_text("x = 2\n", encoding="utf-8")

    audit = audit_source_manifest_payload(manifest, root=tmp_path)

    assert "source_hashes_match_current_files" in {issue["name"] for issue in audit["issues"]}
    assert audit["hash_mismatches"] == ["src/module.py"]


def test_source_manifest_detects_stale_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "module.py"
    target.write_text("x = 1\n", encoding="utf-8")
    manifest = build_source_manifest(tmp_path)
    target.unlink()

    audit = audit_source_manifest_payload(manifest, root=tmp_path)

    assert "source_manifest_has_no_stale_files" in {issue["name"] for issue in audit["issues"]}
