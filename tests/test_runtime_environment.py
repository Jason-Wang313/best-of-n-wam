from pathlib import Path

from wam_inference_value.runtime_environment import audit_runtime_environment_payload, build_runtime_environment


def write_minimal_runtime_files(root: Path, requirement: str = "pytest>=1\n") -> None:
    (root / "requirements.txt").write_text(requirement, encoding="utf-8")
    (root / "requirements-benchmark.txt").write_text("definitely-missing-optional-runtime-gate-package>=1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        """
[project]
dependencies = ["pytest>=1"]

[project.optional-dependencies]
test = ["pytest>=1"]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_runtime_environment_records_core_and_optional_requirements(tmp_path: Path) -> None:
    write_minimal_runtime_files(tmp_path)

    payload = build_runtime_environment(tmp_path)

    assert payload["n_core_requirements"] >= 1
    assert payload["n_optional_requirements"] == 1
    assert payload["module_probes"]["pytest"]["available"] is True
    assert payload["python"]["version_info"][0] >= 3


def test_runtime_environment_detects_requirement_hash_drift(tmp_path: Path) -> None:
    write_minimal_runtime_files(tmp_path)
    payload = build_runtime_environment(tmp_path)
    (tmp_path / "requirements.txt").write_text("pytest>=9999\n", encoding="utf-8")

    audit = audit_runtime_environment_payload(payload, root=tmp_path)

    assert "runtime_requirement_hashes_match" in {issue["name"] for issue in audit["issues"]}
    assert audit["requirement_hash_mismatches"] == ["requirements.txt"]


def test_runtime_environment_detects_python_version_mismatch(tmp_path: Path) -> None:
    write_minimal_runtime_files(tmp_path)
    payload = build_runtime_environment(tmp_path)
    payload["python"]["version_info"] = [0, 0, 0]

    audit = audit_runtime_environment_payload(payload, root=tmp_path)

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "runtime_python_version_supported" in issue_names
    assert "runtime_python_matches_current" in issue_names


def test_runtime_environment_detects_missing_core_package(tmp_path: Path) -> None:
    write_minimal_runtime_files(tmp_path, "definitely-missing-core-runtime-gate-package>=1\n")
    payload = build_runtime_environment(tmp_path)

    audit = audit_runtime_environment_payload(payload, root=tmp_path)

    assert "runtime_core_packages_installed" in {issue["name"] for issue in audit["issues"]}
    assert audit["core_missing"] == ["definitely-missing-core-runtime-gate-package"]
