from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_DIRS = ("src", "scripts", "experiments", "tests")
SOURCE_SUFFIXES = {".py", ".sh"}
ROOT_SOURCE_FILES = (
    "README.md",
    "paper_outline.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-benchmark.txt",
    "docs/theory.md",
)


@dataclass(frozen=True)
class SourceManifestCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[SourceManifestCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(SourceManifestCheck(name=name, ok=bool(ok), detail=detail))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_source_file(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts:
        return False
    if relative.as_posix() in ROOT_SOURCE_FILES:
        return True
    return bool(relative.parts) and relative.parts[0] in SOURCE_DIRS and path.suffix.lower() in SOURCE_SUFFIXES


def scan_source_files(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    records: list[dict[str, Any]] = []
    candidates: list[Path] = []
    for directory in SOURCE_DIRS:
        base = root / directory
        if base.exists():
            candidates.extend(path for path in base.rglob("*") if path.is_file())
    candidates.extend(root / path for path in ROOT_SOURCE_FILES)
    for path in sorted(set(candidates)):
        if not path.exists() or not is_source_file(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def build_source_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    records = scan_source_files(root)
    counts_by_suffix: dict[str, int] = {}
    counts_by_dir: dict[str, int] = {}
    for record in records:
        suffix = str(record["suffix"])
        counts_by_suffix[suffix] = counts_by_suffix.get(suffix, 0) + 1
        directory = str(record["path"]).split("/", 1)[0]
        counts_by_dir[directory] = counts_by_dir.get(directory, 0) + 1
    return {
        "experiment": "source_manifest",
        "schema_version": 1,
        "root": str(root),
        "n_files": len(records),
        "total_bytes": int(sum(int(record["bytes"]) for record in records)),
        "counts_by_suffix": counts_by_suffix,
        "counts_by_dir": counts_by_dir,
        "records": records,
    }


def audit_source_manifest_payload(payload: dict[str, Any], *, root: Path) -> dict[str, Any]:
    root = root.resolve()
    checks: list[SourceManifestCheck] = []
    records = payload.get("records") or []
    current_records = scan_source_files(root)
    by_path = {str(record.get("path")): record for record in records if isinstance(record, dict)}
    current_by_path = {str(record.get("path")): record for record in current_records}
    missing = sorted(set(current_by_path) - set(by_path))
    stale = sorted(set(by_path) - set(current_by_path))
    hash_mismatches = []
    byte_mismatches = []
    malformed_records = []

    for record in records:
        if not isinstance(record, dict):
            malformed_records.append(str(record))
            continue
        path = str(record.get("path") or "")
        current = current_by_path.get(path)
        if current is None:
            continue
        if record.get("sha256") != current.get("sha256"):
            hash_mismatches.append(path)
        if int(record.get("bytes") or -1) != int(current.get("bytes") or -2):
            byte_mismatches.append(path)
        if not isinstance(record.get("sha256"), str) or len(str(record.get("sha256"))) != 64:
            malformed_records.append(path)

    current_counts_by_suffix: dict[str, int] = {}
    current_counts_by_dir: dict[str, int] = {}
    for record in current_records:
        suffix = str(record["suffix"])
        current_counts_by_suffix[suffix] = current_counts_by_suffix.get(suffix, 0) + 1
        directory = str(record["path"]).split("/", 1)[0]
        current_counts_by_dir[directory] = current_counts_by_dir.get(directory, 0) + 1

    add(checks, "source_manifest_records_present", len(records) > 0, f"records={len(records)}")
    add(checks, "source_file_count_matches_scan", int(payload.get("n_files") or -1) == len(current_records), f"payload={payload.get('n_files')}, scan={len(current_records)}")
    add(checks, "source_total_bytes_matches_scan", int(payload.get("total_bytes") or -1) == sum(int(record["bytes"]) for record in current_records), f"payload={payload.get('total_bytes')}, scan={sum(int(record['bytes']) for record in current_records)}")
    add(checks, "source_suffix_counts_match_scan", (payload.get("counts_by_suffix") or {}) == current_counts_by_suffix, f"payload={payload.get('counts_by_suffix')}, scan={current_counts_by_suffix}")
    add(checks, "source_dir_counts_match_scan", (payload.get("counts_by_dir") or {}) == current_counts_by_dir, f"payload={payload.get('counts_by_dir')}, scan={current_counts_by_dir}")
    add(checks, "source_manifest_has_no_missing_current_files", not missing, f"missing={missing[:10]}, count={len(missing)}")
    add(checks, "source_manifest_has_no_stale_files", not stale, f"stale={stale[:10]}, count={len(stale)}")
    add(checks, "source_hashes_match_current_files", not hash_mismatches, f"mismatches={hash_mismatches[:10]}, count={len(hash_mismatches)}")
    add(checks, "source_bytes_match_current_files", not byte_mismatches, f"mismatches={byte_mismatches[:10]}, count={len(byte_mismatches)}")
    add(checks, "source_records_well_formed", not malformed_records, f"malformed={malformed_records[:10]}, count={len(malformed_records)}")
    add(checks, "source_manifest_file_count", len(current_records) >= 150, f"files={len(current_records)}")
    add(checks, "source_manifest_total_bytes", sum(int(record["bytes"]) for record in current_records) >= 1_000_000, f"bytes={sum(int(record['bytes']) for record in current_records)}")
    add(checks, "source_manifest_dir_coverage", all(current_counts_by_dir.get(directory, 0) > 0 for directory in SOURCE_DIRS), f"dirs={current_counts_by_dir}")
    add(checks, "source_manifest_root_file_coverage", all((root / path).exists() and path in current_by_path for path in ROOT_SOURCE_FILES), f"root_files={ROOT_SOURCE_FILES}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "source_manifest",
        "verified": len(issues) == 0,
        "root": str(root),
        "n_files": len(current_records),
        "total_bytes": int(sum(int(record["bytes"]) for record in current_records)),
        "counts_by_suffix": current_counts_by_suffix,
        "counts_by_dir": current_counts_by_dir,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "missing_current_files": missing,
        "stale_manifest_files": stale,
        "hash_mismatches": hash_mismatches,
        "byte_mismatches": byte_mismatches,
        "malformed_records": malformed_records,
    }


def build_and_audit_source_manifest(root: Path) -> dict[str, Any]:
    manifest = build_source_manifest(root)
    audit = audit_source_manifest_payload(manifest, root=root)
    return {**manifest, **{key: value for key, value in audit.items() if key not in {"counts_by_suffix", "counts_by_dir"}}}


def source_manifest_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Source Manifest Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Files hashed: {payload.get('n_files')}",
        f"- Total bytes: {payload.get('total_bytes')}",
        f"- Counts by directory: {payload.get('counts_by_dir')}",
        f"- Counts by suffix: {payload.get('counts_by_suffix')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Source, experiment, script, test, README, paper-outline, requirement, and theory files have a deterministic SHA-256 manifest.")
    lines.append("")
    return "\n".join(lines)
