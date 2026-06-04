from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TrackedArtifactCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[TrackedArtifactCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(TrackedArtifactCheck(name=name, ok=bool(ok), detail=detail))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def git_tracked_paths(root: Path) -> tuple[set[str], int, str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    paths = {line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()}
    return paths, proc.returncode, proc.stdout


def resolve_candidate(root: Path, record: dict[str, Any]) -> Path | None:
    candidate = record.get("resolved_path") or record.get("raw_path")
    if not isinstance(candidate, str) or not candidate:
        return None
    path = Path(candidate).expanduser()
    return path if path.is_absolute() else root / path


def relative_to_root(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def is_tracked_or_tracked_dir(relative: str, tracked_paths: set[str]) -> bool:
    if relative in tracked_paths:
        return True
    prefix = relative.rstrip("/") + "/"
    return any(path.startswith(prefix) for path in tracked_paths)


def unique_records(records: Iterable[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        path = resolve_candidate(root, record)
        relative = relative_to_root(root, path)
        if relative is None or relative in seen:
            continue
        seen.add(relative)
        unique.append(
            {
                "source": record.get("source"),
                "claim_id": record.get("claim_id"),
                "repo_relative": relative,
                "exists": bool(path and path.exists()),
                "kind": "directory" if path and path.is_dir() else "file",
            }
        )
    return sorted(unique, key=lambda item: item["repo_relative"])


def collect_claim_source_records(results_dir: Path, root: Path) -> list[dict[str, Any]]:
    payload = load_json(results_dir / "claim_evidence_quality.json")
    records = []
    for record in payload.get("source_records") or []:
        if isinstance(record, dict):
            records.append({"source": "claim_evidence_quality", **record})
    return unique_records(records, root)


def collect_artifact_reference_records(results_dir: Path, root: Path) -> list[dict[str, Any]]:
    payload = load_json(results_dir / "artifact_integrity.json")
    records = []
    for record in payload.get("references") or []:
        if isinstance(record, dict):
            records.append({"source": "artifact_integrity", **record})
    return unique_records(records, root)


def mark_tracking(records: list[dict[str, Any]], tracked_paths: set[str]) -> list[dict[str, Any]]:
    marked = []
    for record in records:
        relative = str(record.get("repo_relative") or "")
        marked.append({**record, "tracked": is_tracked_or_tracked_dir(relative, tracked_paths)})
    return marked


def audit_tracked_artifact_provenance(
    root: Path,
    results_dir: Path | None = None,
    *,
    tracked_paths: set[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    tracked, git_returncode, git_output = git_tracked_paths(root) if tracked_paths is None else (tracked_paths, 0, "")
    claim_sources = mark_tracking(collect_claim_source_records(results_dir, root), tracked)
    artifact_references = mark_tracking(collect_artifact_reference_records(results_dir, root), tracked)

    untracked_claim_sources = [record for record in claim_sources if not record.get("tracked")]
    untracked_artifact_references = [record for record in artifact_references if not record.get("tracked")]
    missing_claim_sources = [record for record in claim_sources if not record.get("exists")]
    missing_artifact_references = [record for record in artifact_references if not record.get("exists")]

    checks: list[TrackedArtifactCheck] = []
    add(checks, "git_ls_files_exit_zero", git_returncode == 0, f"returncode={git_returncode}")
    add(checks, "tracked_index_nonempty", len(tracked) > 0, f"tracked={len(tracked)}")
    add(checks, "claim_sources_present", len(claim_sources) > 0, f"claim_sources={len(claim_sources)}")
    add(checks, "artifact_references_present", len(artifact_references) > 0, f"artifact_references={len(artifact_references)}")
    add(checks, "claim_sources_exist", not missing_claim_sources, f"missing={len(missing_claim_sources)}")
    add(checks, "artifact_references_exist", not missing_artifact_references, f"missing={len(missing_artifact_references)}")
    add(checks, "claim_sources_tracked", not untracked_claim_sources, f"untracked={len(untracked_claim_sources)}")
    add(checks, "artifact_references_tracked", not untracked_artifact_references, f"untracked={len(untracked_artifact_references)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "tracked_artifact_provenance",
        "verified": len(issues) == 0,
        "n_tracked_paths": len(tracked),
        "n_claim_sources": len(claim_sources),
        "n_artifact_references": len(artifact_references),
        "n_untracked_claim_sources": len(untracked_claim_sources),
        "n_untracked_artifact_references": len(untracked_artifact_references),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "git_returncode": git_returncode,
        "git_output_excerpt": git_output[:500],
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "untracked_claim_sources": untracked_claim_sources,
        "untracked_artifact_references": untracked_artifact_references,
    }


def tracked_artifact_provenance_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Tracked Artifact Provenance Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Git-tracked paths: {payload.get('n_tracked_paths')}",
        f"- Claim source artifacts checked: {payload.get('n_claim_sources')}",
        f"- Published artifact references checked: {payload.get('n_artifact_references')}",
        f"- Untracked claim sources: {payload.get('n_untracked_claim_sources')}",
        f"- Untracked artifact references: {payload.get('n_untracked_artifact_references')}",
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
        lines.append("Every current claim source and published artifact reference resolves to a file or directory represented in the git index.")
    lines.append("")
    return "\n".join(lines)
