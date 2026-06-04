from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PATH_KEY_EXACT = {
    "artifact",
    "artifacts",
    "artifact_coverage_path",
    "curves_path",
    "data_path",
    "eval_path",
    "exact_path",
    "model_path",
    "registry_path",
    "report_path",
    "seed_metrics_path",
    "table_path",
}
PATH_KEY_SUFFIXES = ("_path",)
DIAGNOSTIC_JSONS_WITH_EXTERNAL_PATHS = {
    "external_benchmark_runtime_probe.json",
}


@dataclass(frozen=True)
class ArtifactReference:
    source_json: str
    json_path: str
    raw_path: str


def is_path_key(key: str) -> bool:
    lower = key.lower()
    return lower in PATH_KEY_EXACT or lower.endswith(PATH_KEY_SUFFIXES)


def resolve_artifact_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else root / path


def _iter_string_leaves(value: Any, trail: tuple[str, ...]) -> list[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        return [(trail, value)]
    if isinstance(value, dict):
        leaves: list[tuple[tuple[str, ...], str]] = []
        for key, nested in value.items():
            leaves.extend(_iter_string_leaves(nested, trail + (str(key),)))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, nested in enumerate(value):
            leaves.extend(_iter_string_leaves(nested, trail + (str(index),)))
        return leaves
    return []


def collect_artifact_references(json_path: Path, payload: Any) -> list[ArtifactReference]:
    if json_path.name in DIAGNOSTIC_JSONS_WITH_EXTERNAL_PATHS:
        return []

    refs: list[ArtifactReference] = []

    def walk(value: Any, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                next_trail = trail + (str(key),)
                if is_path_key(str(key)):
                    for leaf_trail, raw_path in _iter_string_leaves(nested, next_trail):
                        refs.append(
                            ArtifactReference(
                                source_json=json_path.name,
                                json_path=".".join(leaf_trail),
                                raw_path=raw_path,
                            )
                        )
                else:
                    walk(nested, next_trail)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, trail + (str(index),))

    walk(payload)
    return refs


def validate_reference(ref: ArtifactReference, root: Path) -> dict[str, Any]:
    path = resolve_artifact_path(ref.raw_path, root)
    record: dict[str, Any] = {
        "source_json": ref.source_json,
        "json_path": ref.json_path,
        "raw_path": ref.raw_path,
        "resolved_path": str(path),
        "exists": path.exists(),
        "bytes": None,
        "rows": None,
        "status": "ok",
        "reason": "",
    }
    if not path.exists():
        record["status"] = "missing"
        record["reason"] = "referenced artifact does not exist"
        return record
    if path.is_dir():
        record["status"] = "ok"
        record["reason"] = "directory exists"
        return record
    size = path.stat().st_size
    record["bytes"] = size
    if size <= 0:
        record["status"] = "empty"
        record["reason"] = "referenced artifact is empty"
        return record
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                row_count = sum(1 for _ in reader)
                record["rows"] = row_count
                if not reader.fieldnames:
                    record["status"] = "invalid_csv"
                    record["reason"] = "CSV has no header"
                elif row_count <= 0:
                    record["status"] = "zero_row_csv"
                    record["reason"] = "CSV has zero rows"
        except Exception as exc:  # pragma: no cover - exercised by corrupt external files.
            record["status"] = "invalid_csv"
            record["reason"] = f"CSV parse failed: {exc}"
    elif suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - exercised by corrupt external files.
            record["status"] = "invalid_json"
            record["reason"] = f"JSON parse failed: {exc}"
    return record


def audit_result_artifacts(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    references: list[ArtifactReference] = []
    json_files = [path for path in sorted(results_dir.glob("*.json")) if path.name != "artifact_integrity.json"]
    for json_file in json_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            references.append(
                ArtifactReference(
                    source_json=json_file.name,
                    json_path="<json>",
                    raw_path=str(json_file),
                )
            )
            continue
        references.extend(collect_artifact_references(json_file, payload))

    records = [validate_reference(ref, root) for ref in references]
    issue_statuses = {"missing", "empty", "invalid_csv", "zero_row_csv", "invalid_json"}
    issues = [record for record in records if record["status"] in issue_statuses]
    by_status: dict[str, int] = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
    return {
        "experiment": "artifact_integrity",
        "verified": len(issues) == 0,
        "results_dir": str(results_dir),
        "n_result_json": len(json_files),
        "n_references": len(records),
        "status_counts": by_status,
        "n_issues": len(issues),
        "issues": issues,
        "references": records,
    }


def artifact_integrity_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Artifact Integrity Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Result JSON files: {payload.get('n_result_json')}",
        f"- Artifact references checked: {payload.get('n_references')}",
        f"- Issues: {payload.get('n_issues')}",
        f"- Status counts: {payload.get('status_counts')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        for issue in issues:
            lines.append(
                "- "
                f"{issue.get('source_json')}:{issue.get('json_path')} -> "
                f"{issue.get('raw_path')} [{issue.get('status')}: {issue.get('reason')}]"
            )
    else:
        lines.append("No missing, empty, invalid, or zero-row referenced artifacts were found.")
    return "\n".join(lines) + "\n"
