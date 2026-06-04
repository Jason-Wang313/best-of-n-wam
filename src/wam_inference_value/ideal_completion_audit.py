from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wam_inference_value.ideal_claim_boundary import audit_ideal_claim_boundary


@dataclass(frozen=True)
class IdealCompletionCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[IdealCompletionCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(IdealCompletionCheck(name=name, ok=bool(ok), detail=detail))


def load_boundary(root: Path, results_dir: Path) -> dict[str, Any]:
    path = results_dir / "ideal_claim_boundary.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return audit_ideal_claim_boundary(root, results_dir)


def audit_ideal_completion(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    boundary = load_boundary(root, results_dir)
    rows = [row for row in boundary.get("rows") or [] if isinstance(row, dict)]
    supported_rows = [row for row in rows if row.get("endpoint_supported") is True]
    unsupported_rows = [row for row in rows if row.get("endpoint_supported") is not True]
    future_blockers = [row for row in unsupported_rows if row.get("future_only") is True]
    false_supported_future = [
        row for row in rows if row.get("future_only") is True and row.get("endpoint_supported") is True
    ]
    blockers_missing_requirements = [
        row for row in future_blockers if len(row.get("promotion_requirements") or []) < 2
    ]
    blockers_missing_evidence_classes = [
        row for row in future_blockers if len(row.get("missing_evidence_classes") or []) < 2
    ]
    blockers_missing_gap_files = [
        row for row in future_blockers if row.get("missing_gap_evidence_files")
    ]
    all_supported = bool(rows) and not unsupported_rows
    completion_verdict = "complete" if all_supported else "not_complete"

    checks: list[IdealCompletionCheck] = []
    add(checks, "ideal_boundary_verified", boundary.get("verified") is True, f"verified={boundary.get('verified')}")
    add(checks, "ideal_rows_present", len(rows) >= 1, f"rows={len(rows)}")
    add(
        checks,
        "future_only_rows_not_counted_as_supported",
        not false_supported_future,
        f"false_supported={[row.get('id') for row in false_supported_future]}",
    )
    add(
        checks,
        "future_blockers_have_promotion_requirements",
        not blockers_missing_requirements,
        f"missing={[row.get('id') for row in blockers_missing_requirements]}",
    )
    add(
        checks,
        "future_blockers_have_missing_evidence_classes",
        not blockers_missing_evidence_classes,
        f"missing={[row.get('id') for row in blockers_missing_evidence_classes]}",
    )
    add(
        checks,
        "future_blockers_have_gap_evidence_files",
        not blockers_missing_gap_files,
        f"missing={[row.get('id') for row in blockers_missing_gap_files]}",
    )
    add(
        checks,
        "completion_verdict_matches_rows",
        (completion_verdict == "complete") == all_supported,
        f"verdict={completion_verdict}, unsupported={len(unsupported_rows)}",
    )

    blocker_rows = []
    for row in future_blockers:
        blocker_rows.append(
            {
                "id": row.get("id"),
                "ideal_claim": row.get("ideal_claim"),
                "limitation": row.get("limitation"),
                "promotion_requirements": row.get("promotion_requirements") or [],
                "missing_evidence_classes": row.get("missing_evidence_classes") or [],
                "gap_evidence_files": row.get("gap_evidence_files") or [],
            }
        )

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "ideal_completion_audit",
        "verified": len(issues) == 0,
        "completion_verdict": completion_verdict,
        "all_ideal_endpoints_supported": all_supported,
        "n_ideal_claims": len(rows),
        "n_supported_endpoints": len(supported_rows),
        "n_unsupported_endpoints": len(unsupported_rows),
        "n_future_blockers": len(future_blockers),
        "supported_endpoint_ids": [row.get("id") for row in supported_rows],
        "unsupported_endpoint_ids": [row.get("id") for row in unsupported_rows],
        "future_blocker_ids": [row.get("id") for row in future_blockers],
        "future_blockers": blocker_rows,
        "boundary_goal_completion_status": boundary.get("goal_completion_status"),
        "boundary_all_ideal_claims_promotable": boundary.get("all_ideal_claims_promotable"),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def ideal_completion_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ideal Completion Audit",
        "",
        f"- Verified audit: {payload.get('verified')}",
        f"- Completion verdict: {payload.get('completion_verdict')}",
        f"- All ideal endpoints supported: {payload.get('all_ideal_endpoints_supported')}",
        f"- Ideal claims audited: {payload.get('n_ideal_claims')}",
        f"- Supported endpoints: {payload.get('n_supported_endpoints')}",
        f"- Unsupported endpoints: {payload.get('n_unsupported_endpoints')}",
        f"- Future-only blockers: {payload.get('n_future_blockers')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    blockers = payload.get("future_blockers") or []
    if blockers:
        lines.extend(["## Future-Only Blockers", ""])
        for blocker in blockers:
            lines.append(f"- `{blocker.get('id')}`: {blocker.get('limitation')}")
            requirements = blocker.get("promotion_requirements") or []
            if requirements:
                lines.append("  Promotion requirements before this can become a result:")
                for requirement in requirements:
                    lines.append(f"  - {requirement}")
            missing = blocker.get("missing_evidence_classes") or []
            if missing:
                lines.append("  Missing evidence classes:")
                for item in missing:
                    lines.append(f"  - {item}")
    issues = payload.get("issues") or []
    if issues:
        lines.extend(["", "## Audit Issues", ""])
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.extend(
            [
                "",
                "The audit is internally consistent. It does not certify completion unless every ideal endpoint is supported by current artifacts.",
            ]
        )
    lines.append("")
    return "\n".join(lines)
