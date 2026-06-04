from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wam_inference_value.tracked_artifact_provenance import (
    collect_artifact_reference_records,
    collect_claim_source_records,
)


SELF_OUTPUTS = {
    "reports/evidence_hash_coverage_report.md",
    "results/evidence_hash_coverage.json",
}


@dataclass(frozen=True)
class EvidenceHashCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[EvidenceHashCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(EvidenceHashCheck(name=name, ok=bool(ok), detail=detail))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix()
        data_hash = sha256_file(child)
        size = child.stat().st_size
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def hash_record(root: Path, record: dict[str, Any], category: str) -> dict[str, Any]:
    relative = str(record.get("repo_relative") or "")
    path = root / relative
    exists = path.exists()
    kind = "directory" if exists and path.is_dir() else "file"
    hashed: dict[str, Any] = {
        "category": category,
        "source": record.get("source"),
        "claim_id": record.get("claim_id"),
        "repo_relative": relative,
        "exists": exists,
        "kind": kind,
        "bytes": None,
        "tree_files": None,
        "sha256": None,
        "self_output": relative in SELF_OUTPUTS,
    }
    if not exists or relative in SELF_OUTPUTS:
        return hashed
    if path.is_dir():
        digest, total_bytes, file_count = sha256_tree(path)
        hashed.update({"sha256": digest, "bytes": total_bytes, "tree_files": file_count})
        return hashed
    hashed.update({"sha256": sha256_file(path), "bytes": path.stat().st_size, "tree_files": 1})
    return hashed


def build_evidence_hash_records(root: Path, results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claim_sources = [
        hash_record(root, record, "claim_source")
        for record in collect_claim_source_records(results_dir, root)
    ]
    artifact_references = [
        hash_record(root, record, "artifact_reference")
        for record in collect_artifact_reference_records(results_dir, root)
    ]
    return claim_sources, artifact_references


def audit_evidence_hash_coverage(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claim_sources, artifact_references = build_evidence_hash_records(root, results_dir)
    all_records = claim_sources + artifact_references
    nonself_claim_sources = [record for record in claim_sources if not record.get("self_output")]
    nonself_artifact_references = [record for record in artifact_references if not record.get("self_output")]
    nonself_records = [record for record in all_records if not record.get("self_output")]
    self_output_records = [record for record in all_records if record.get("self_output")]
    missing_records = [record for record in nonself_records if not record.get("exists")]
    unhashed_records = [record for record in nonself_records if not record.get("sha256")]
    malformed_hashes = [
        record
        for record in nonself_records
        if not isinstance(record.get("sha256"), str) or len(str(record.get("sha256"))) != 64
    ]
    empty_file_records = [
        record
        for record in nonself_records
        if record.get("kind") == "file" and int(record.get("bytes") or 0) <= 0
    ]
    conflicting_duplicates = []
    by_relative: dict[str, set[str]] = {}
    for record in nonself_records:
        relative = str(record.get("repo_relative") or "")
        digest = str(record.get("sha256") or "")
        by_relative.setdefault(relative, set()).add(digest)
    for relative, digests in by_relative.items():
        if len(digests) > 1:
            conflicting_duplicates.append(relative)
    checks: list[EvidenceHashCheck] = []
    add(checks, "claim_sources_present", len(nonself_claim_sources) > 0, f"claim_sources={len(nonself_claim_sources)}")
    add(checks, "artifact_references_present", len(nonself_artifact_references) > 0, f"artifact_references={len(nonself_artifact_references)}")
    add(checks, "claim_source_hash_count", len(nonself_claim_sources) >= 100, f"claim_sources={len(nonself_claim_sources)}")
    add(checks, "artifact_reference_hash_count", len(nonself_artifact_references) >= 400, f"artifact_references={len(nonself_artifact_references)}")
    add(checks, "all_nonself_records_exist", not missing_records, f"missing={len(missing_records)}")
    add(checks, "all_nonself_records_hashed", not unhashed_records, f"unhashed={len(unhashed_records)}")
    add(checks, "hashes_well_formed", not malformed_hashes, f"malformed={len(malformed_hashes)}")
    add(checks, "hashed_files_nonempty", not empty_file_records, f"empty={len(empty_file_records)}")
    add(checks, "duplicate_hashes_consistent", not conflicting_duplicates, f"conflicting={conflicting_duplicates[:10]}, count={len(conflicting_duplicates)}")
    add(checks, "self_outputs_excluded", len(self_output_records) <= len(SELF_OUTPUTS) * 2, f"self_outputs={len(self_output_records)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "evidence_hash_coverage",
        "verified": len(issues) == 0,
        "n_claim_sources": len(nonself_claim_sources),
        "n_artifact_references": len(nonself_artifact_references),
        "n_hashed_records": len([record for record in nonself_records if record.get("sha256")]),
        "n_self_outputs_excluded": len(self_output_records),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "missing_records": missing_records,
        "unhashed_records": unhashed_records,
        "malformed_hashes": malformed_hashes,
        "empty_file_records": empty_file_records,
        "conflicting_duplicate_hashes": conflicting_duplicates,
        "claim_source_hashes": nonself_claim_sources,
        "artifact_reference_hashes": nonself_artifact_references,
        "self_output_records": self_output_records,
    }


def evidence_hash_coverage_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence Hash Coverage Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Non-self claim source artifacts hashed: {payload.get('n_claim_sources')}",
        f"- Non-self published artifact references hashed: {payload.get('n_artifact_references')}",
        f"- Total non-self records hashed: {payload.get('n_hashed_records')}",
        f"- Self outputs excluded: {payload.get('n_self_outputs_excluded')}",
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
        lines.append("Every non-self current claim source and every non-self published artifact reference has a deterministic SHA-256 hash recorded by this gate.")
    lines.append("")
    return "\n".join(lines)
