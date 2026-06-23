from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REANCHOR_TOP_LEVELS = {
    "docs",
    "experiments",
    "reports",
    "results",
    "scripts",
    "src",
    "tests",
    "README.md",
    "paper_outline.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-benchmark.txt",
}


@dataclass(frozen=True)
class RepoBoundCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[RepoBoundCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(RepoBoundCheck(name=name, ok=bool(ok), detail=detail))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def resolve_record(root: Path, record: dict[str, Any]) -> Path | None:
    candidate = record.get("resolved_path") or record.get("raw_path")
    if not isinstance(candidate, str) or not candidate:
        return None
    path = Path(candidate).expanduser()
    path = path if path.is_absolute() else root / path
    return reanchor_repo_path(root, path)


def reanchor_repo_path(root: Path, path: Path) -> Path:
    try:
        path.resolve().relative_to(root)
        return path
    except ValueError:
        pass

    parts = path.parts
    for index, part in enumerate(parts):
        if part not in REANCHOR_TOP_LEVELS:
            continue
        candidate = root / Path(*parts[index:])
        if candidate.exists():
            return candidate
    return path


def repo_relative(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def category_for(relative: str | None, suffix: str, kind: str) -> str:
    if kind == "directory":
        return "directory"
    if not relative:
        return "outside_or_unresolved"
    if relative == "README.md" or relative == "paper_outline.md" or relative.startswith("docs/"):
        return "documentation"
    if relative.startswith("reports/"):
        return "report"
    if relative.startswith(("src/", "scripts/", "experiments/", "tests/")):
        return "source_or_test"
    if relative.startswith("results/tables/") or suffix == ".csv":
        return "result_table"
    if relative.startswith("results/figures/") or suffix == ".png":
        return "figure"
    if relative.startswith("results/models/") or suffix in {".npz", ".joblib"}:
        return "model"
    if relative.startswith("results/") and suffix == ".json":
        return "result_json"
    return "other_repo_artifact"


def normalized_record(root: Path, category: str, record: dict[str, Any]) -> dict[str, Any]:
    path = resolve_record(root, record)
    relative = repo_relative(root, path)
    exists = bool(path and path.exists())
    kind = "directory" if exists and path and path.is_dir() else "file"
    suffix = path.suffix.lower() if path else ""
    raw = record.get("raw_path")
    raw_text = raw if isinstance(raw, str) else ""
    return {
        "record_category": category,
        "source": record.get("source"),
        "source_json": record.get("source_json"),
        "json_ref": record.get("json_path"),
        "claim_id": record.get("claim_id"),
        "raw": raw_text,
        "resolved": str(path) if path else "",
        "repo_relative": relative,
        "inside_repo": relative is not None,
        "exists": exists,
        "kind": kind,
        "suffix": suffix,
        "artifact_category": category_for(relative, suffix, kind),
        "raw_has_parent_traversal": any(part == ".." for part in Path(raw_text).parts) if raw_text else False,
    }


def collect_records(root: Path, results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    claim_payload = load_json(results_dir / "claim_evidence_quality.json")
    for record in claim_payload.get("source_records") or []:
        if isinstance(record, dict):
            records.append(normalized_record(root, "claim_source", {"source": "claim_evidence_quality", **record}))
    artifact_payload = load_json(results_dir / "artifact_integrity.json")
    for record in artifact_payload.get("references") or []:
        if isinstance(record, dict):
            records.append(normalized_record(root, "artifact_reference", {"source": "artifact_integrity", **record}))
    records.sort(key=lambda item: (str(item.get("repo_relative")), str(item.get("record_category")), str(item.get("source_json"))))
    return records


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def audit_repo_bound_artifacts(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    records = collect_records(root, results_dir)
    outside = [record for record in records if not record.get("inside_repo")]
    missing = [record for record in records if not record.get("exists")]
    traversal = [record for record in records if record.get("raw_has_parent_traversal")]
    claim_sources = [record for record in records if record.get("record_category") == "claim_source"]
    artifact_references = [record for record in records if record.get("record_category") == "artifact_reference"]
    artifact_categories = count_by(records, "artifact_category")

    checks: list[RepoBoundCheck] = []
    add(checks, "records_present", len(records) > 0, f"records={len(records)}")
    add(checks, "claim_sources_present", len(claim_sources) >= 100, f"claim_sources={len(claim_sources)}")
    add(checks, "artifact_references_present", len(artifact_references) >= 400, f"artifact_references={len(artifact_references)}")
    add(checks, "all_records_inside_repo", not outside, f"outside={len(outside)}")
    add(checks, "all_records_exist", not missing, f"missing={len(missing)}")
    add(checks, "no_parent_traversal_refs", not traversal, f"traversal={len(traversal)}")
    add(checks, "result_json_coverage", artifact_categories.get("result_json", 0) >= 100, f"count={artifact_categories.get('result_json', 0)}")
    add(checks, "result_table_coverage", artifact_categories.get("result_table", 0) >= 200, f"count={artifact_categories.get('result_table', 0)}")
    add(checks, "report_coverage", artifact_categories.get("report", 0) >= 20, f"count={artifact_categories.get('report', 0)}")
    add(checks, "figure_coverage", artifact_categories.get("figure", 0) >= 30, f"count={artifact_categories.get('figure', 0)}")
    add(checks, "model_coverage", artifact_categories.get("model", 0) >= 30, f"count={artifact_categories.get('model', 0)}")
    add(checks, "source_or_doc_coverage", artifact_categories.get("source_or_test", 0) + artifact_categories.get("documentation", 0) >= 5, f"source_or_test={artifact_categories.get('source_or_test', 0)}, documentation={artifact_categories.get('documentation', 0)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "repo_bound_artifact_audit",
        "verified": len(issues) == 0,
        "results_dir": str(results_dir),
        "n_records": len(records),
        "n_claim_sources": len(claim_sources),
        "n_artifact_references": len(artifact_references),
        "n_outside_repo": len(outside),
        "n_missing": len(missing),
        "n_parent_traversal": len(traversal),
        "artifact_categories": artifact_categories,
        "suffix_counts": count_by(records, "suffix"),
        "record_category_counts": count_by(records, "record_category"),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "outside_repo_records": outside,
        "missing_records": missing,
        "parent_traversal_records": traversal,
        "records": records,
    }


def repo_bound_artifact_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Repo-Bound Artifact Audit Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Records checked: {payload.get('n_records')}",
        f"- Claim sources: {payload.get('n_claim_sources')}",
        f"- Artifact references: {payload.get('n_artifact_references')}",
        f"- Outside-repo records: {payload.get('n_outside_repo')}",
        f"- Missing records: {payload.get('n_missing')}",
        f"- Parent traversal refs: {payload.get('n_parent_traversal')}",
        f"- Artifact categories: {payload.get('artifact_categories')}",
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
        lines.append("Every current claim source and published artifact reference resolves inside the repository and the evidence surface spans result JSON, CSV table, report, figure, model, source, and documentation artifact classes.")
    lines.append("")
    return "\n".join(lines)
