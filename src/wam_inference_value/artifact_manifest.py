from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INCLUDE_SUFFIXES = {".csv", ".json", ".npz", ".pdf", ".png"}
EXCLUDED_RESULT_NAMES = {
    "artifact_integrity.json",
    "artifact_manifest.json",
    "claim_evidence_quality.json",
    "claim_generation_consistency.json",
    "claim_ledger_integrity.json",
    "claim_semantics.json",
    "claims_status.json",
    "claims_status.md",
    "command_result_consistency.json",
    "experiment_registry.json",
    "figure_quality.json",
    "model_artifact_integrity.json",
    "narrative_consistency.json",
    "raw_result_recompute.json",
    "report_generation_consistency.json",
    "result_consistency.json",
    "runtime_environment.json",
    "script_contracts.json",
    "source_manifest.json",
    "table_schema.json",
    "test_inventory.json",
}


@dataclass(frozen=True)
class ArtifactManifestCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ArtifactManifestCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ArtifactManifestCheck(name=name, ok=bool(ok), detail=detail))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_manifested_artifact(path: Path, results_dir: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(results_dir)
    if relative.parts and relative.parts[0] == "smoke":
        return False
    if path.name in EXCLUDED_RESULT_NAMES:
        return False
    return path.suffix.lower() in INCLUDE_SUFFIXES


def scan_scientific_artifacts(root: Path, results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*")):
        if not is_manifested_artifact(path, results_dir):
            continue
        rel_path = path.relative_to(root).as_posix()
        records.append(
            {
                "path": rel_path,
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def build_artifact_manifest(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    records = scan_scientific_artifacts(root, results_dir)
    counts_by_suffix: dict[str, int] = {}
    for record in records:
        suffix = str(record["suffix"])
        counts_by_suffix[suffix] = counts_by_suffix.get(suffix, 0) + 1
    return {
        "experiment": "artifact_manifest",
        "schema_version": 1,
        "results_dir": str(results_dir),
        "n_files": len(records),
        "total_bytes": int(sum(int(record["bytes"]) for record in records)),
        "counts_by_suffix": counts_by_suffix,
        "records": records,
    }


def audit_artifact_manifest_payload(payload: dict[str, Any], *, root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    checks: list[ArtifactManifestCheck] = []
    records = payload.get("records") or []
    current_records = scan_scientific_artifacts(root, results_dir)
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

    counts = payload.get("counts_by_suffix") or {}
    current_counts: dict[str, int] = {}
    for record in current_records:
        suffix = str(record.get("suffix"))
        current_counts[suffix] = current_counts.get(suffix, 0) + 1

    add(checks, "manifest_records_present", len(records) > 0, f"records={len(records)}")
    add(checks, "manifest_file_count_matches_scan", int(payload.get("n_files") or -1) == len(current_records), f"payload={payload.get('n_files')}, scan={len(current_records)}")
    add(checks, "manifest_total_bytes_matches_scan", int(payload.get("total_bytes") or -1) == sum(int(record["bytes"]) for record in current_records), f"payload={payload.get('total_bytes')}, scan={sum(int(record['bytes']) for record in current_records)}")
    add(checks, "manifest_counts_by_suffix_match_scan", counts == current_counts, f"payload={counts}, scan={current_counts}")
    add(checks, "manifest_has_no_missing_current_artifacts", not missing, f"missing={missing[:10]}, count={len(missing)}")
    add(checks, "manifest_has_no_stale_artifacts", not stale, f"stale={stale[:10]}, count={len(stale)}")
    add(checks, "manifest_hashes_match_current_files", not hash_mismatches, f"mismatches={hash_mismatches[:10]}, count={len(hash_mismatches)}")
    add(checks, "manifest_bytes_match_current_files", not byte_mismatches, f"mismatches={byte_mismatches[:10]}, count={len(byte_mismatches)}")
    add(checks, "manifest_records_well_formed", not malformed_records, f"malformed={malformed_records[:10]}, count={len(malformed_records)}")
    add(checks, "manifest_scientific_artifact_file_count", len(current_records) >= 350, f"files={len(current_records)}")
    add(checks, "manifest_table_count", current_counts.get(".csv", 0) >= 150, f"csv={current_counts.get('.csv', 0)}")
    add(checks, "manifest_json_count", current_counts.get(".json", 0) >= 80, f"json={current_counts.get('.json', 0)}")
    add(checks, "manifest_model_count", current_counts.get(".npz", 0) >= 20, f"npz={current_counts.get('.npz', 0)}")
    add(checks, "manifest_figure_count", current_counts.get(".png", 0) >= 20, f"png={current_counts.get('.png', 0)}")
    add(checks, "manifest_total_bytes_large_enough", int(payload.get("total_bytes") or 0) >= 10_000_000, f"bytes={payload.get('total_bytes')}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "artifact_manifest",
        "verified": len(issues) == 0,
        "results_dir": str(results_dir),
        "n_files": len(current_records),
        "total_bytes": int(sum(int(record["bytes"]) for record in current_records)),
        "counts_by_suffix": current_counts,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "missing_current_artifacts": missing,
        "stale_manifest_artifacts": stale,
        "hash_mismatches": hash_mismatches,
        "byte_mismatches": byte_mismatches,
        "malformed_records": malformed_records,
    }


def build_and_audit_artifact_manifest(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    manifest = build_artifact_manifest(root, results_dir)
    audit = audit_artifact_manifest_payload(manifest, root=root, results_dir=results_dir)
    return {**manifest, **{key: value for key, value in audit.items() if key not in {"counts_by_suffix"}}}


def artifact_manifest_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Artifact Manifest Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Files hashed: {payload.get('n_files')}",
        f"- Total bytes: {payload.get('total_bytes')}",
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
        lines.append("Canonical scientific result artifacts have a deterministic SHA-256 manifest covering result JSONs, CSV tables, figures, and model files, excluding temporary smoke outputs and self-referential meta-gate outputs.")
    lines.append("")
    return "\n".join(lines)
