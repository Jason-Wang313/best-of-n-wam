from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRONTIER_SURFACES = [
    "README.md",
    "paper_outline.md",
    "reports/paper_result_summary.md",
    "reports/final_decision_report.md",
    "reports/reviewer_risk_assessment.md",
    "reports/maxout_initial_audit.md",
    "reports/maxout_completion_audit.md",
]
SAFE_CONTEXT_TOKENS = [
    "not ",
    "no ",
    "lacks",
    "missing",
    "future",
    "discussion",
    "do not claim",
    "without claiming",
    "not claimed",
    "not counted",
    "not evidence",
    "unsupported",
    "unresolved",
    "limitation",
    "blocker",
    "failed",
    "probe",
    "attempt",
    "remains",
    "weaker",
    "absence",
    "beyond",
    "next step",
    "still",
]
SAFE_HEADING_TOKENS = [
    "limitation",
    "future",
    "discussion",
    "do not claim",
    "unresolved",
    "risk",
    "weakest",
    "skeptical",
    "next step",
    "remaining",
]
FRONTIER_ITEMS = [
    {
        "id": "real_robot_hil",
        "label": "Real robot or hardware-in-the-loop evidence",
        "pattern": re.compile(r"\breal[- ]robot\b|\bhardware-in-the-loop\b", re.I),
        "minimum_guarded_mentions": 4,
        "required_artifacts": [],
    },
    {
        "id": "modern_vla_libero",
        "label": "Modern VLA-style LIBERO sparse-success policy evidence",
        "pattern": re.compile(r"\bmodern\s+VLA\b|\bVLA-style\b", re.I),
        "minimum_guarded_mentions": 3,
        "required_artifacts": [],
    },
    {
        "id": "full_robocasa_wide",
        "label": "Full RoboCasa-wide learned-WAM validation",
        "pattern": re.compile(r"\bfull\s+RoboCasa-wide\b", re.I),
        "minimum_guarded_mentions": 3,
        "required_artifacts": [],
    },
    {
        "id": "maniskill_visual_ee",
        "label": "ManiSkill visual/RGB-D or end-effector-control validation",
        "pattern": re.compile(r"\bManiSkill\b.*\b(?:RGB|RGB-D|RGB/RGB-D|visual|EE-control|end-effector)\b", re.I),
        "minimum_guarded_mentions": 3,
        "required_artifacts": [
            "results/benchmark_maniskill_visual_probe.json",
            "results/benchmark_maniskill_dependency_probe.json",
            "reports/maniskill_visual_blocker_report.md",
            "reports/maniskill_dependency_blocker_report.md",
        ],
    },
]


@dataclass(frozen=True)
class FrontierCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[FrontierCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(FrontierCheck(name=name, ok=bool(ok), detail=detail))


def current_heading(line: str, previous: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return stripped.lstrip("#").strip()
    return previous


def guarded_context(text: str, heading: str) -> bool:
    haystack = f"{heading}\n{text}".lower()
    return any(token in haystack for token in SAFE_CONTEXT_TOKENS) or any(token in heading.lower() for token in SAFE_HEADING_TOKENS)


def load_claims(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / "claims_status.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload.get("claims") if isinstance(payload, dict) else []
    return [claim for claim in claims or [] if isinstance(claim, dict)]


def scan_surfaces(root: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
    pattern = item["pattern"]
    records: list[dict[str, Any]] = []
    for relative in FRONTIER_SURFACES:
        path = root / relative
        if not path.exists():
            continue
        heading = ""
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            heading = current_heading(line, heading)
            stripped = line.strip()
            if not stripped or not pattern.search(stripped):
                continue
            records.append(
                {
                    "frontier_id": item["id"],
                    "surface": relative,
                    "line": line_no,
                    "heading": heading,
                    "guarded": guarded_context(stripped, heading),
                    "text": stripped,
                }
            )
    return records


def scan_claim_promotions(claims: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    pattern = item["pattern"]
    promoted: list[dict[str, Any]] = []
    for claim in claims:
        text = str(claim.get("claim") or "")
        evidence = str(claim.get("evidence") or "")
        combined = f"{text} {evidence}"
        if not pattern.search(combined):
            continue
        if guarded_context(combined, ""):
            continue
        promoted.append(
            {
                "frontier_id": item["id"],
                "claim_id": claim.get("id"),
                "status": claim.get("status"),
                "claim": text,
                "evidence": evidence,
            }
        )
    return promoted


def artifact_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def audit_frontier_integrity(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claims = load_claims(results_dir)
    checks: list[FrontierCheck] = []
    all_records: list[dict[str, Any]] = []
    all_promotions: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []

    missing_surfaces = [surface for surface in FRONTIER_SURFACES if not (root / surface).exists()]
    add(checks, "frontier_surfaces_exist", not missing_surfaces, f"missing_surfaces={missing_surfaces}")
    add(checks, "claim_ledger_loaded", len(claims) >= 120, f"claims={len(claims)}")

    for item in FRONTIER_ITEMS:
        records = scan_surfaces(root, item)
        promotions = scan_claim_promotions(claims, item)
        artifact_records = [artifact_record(root, relative) for relative in item.get("required_artifacts", [])]
        missing_artifacts = [record for record in artifact_records if not record["exists"] or record["bytes"] <= 0]
        guarded_records = [record for record in records if record.get("guarded")]
        unguarded_records = [record for record in records if not record.get("guarded")]
        all_records.extend(records)
        all_promotions.extend(promotions)
        all_records.extend({"frontier_id": item["id"], "artifact": record} for record in artifact_records)
        frontier_rows.append(
            {
                "frontier_id": item["id"],
                "label": item["label"],
                "status": "guarded_not_promoted",
                "n_mentions": len(records),
                "n_guarded_mentions": len(guarded_records),
                "n_unguarded_mentions": len(unguarded_records),
                "n_promoted_claims": len(promotions),
                "n_required_artifacts": len(artifact_records),
                "n_missing_required_artifacts": len(missing_artifacts),
                "required_artifacts": artifact_records,
            }
        )
        add(
            checks,
            f"{item['id']}_guarded_mentions_present",
            len(guarded_records) >= int(item["minimum_guarded_mentions"]),
            f"guarded={len(guarded_records)}, required={item['minimum_guarded_mentions']}",
        )
        add(checks, f"{item['id']}_no_unguarded_publication_mentions", not unguarded_records, f"unguarded={len(unguarded_records)}")
        add(checks, f"{item['id']}_not_promoted_as_verified_claim", not promotions, f"promotions={len(promotions)}")
        add(checks, f"{item['id']}_required_blocker_artifacts_exist", not missing_artifacts, f"missing_artifacts={missing_artifacts}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "frontier_integrity",
        "verified": len(issues) == 0,
        "n_frontier_items": len(FRONTIER_ITEMS),
        "n_claims_loaded": len(claims),
        "n_publication_surfaces": len(FRONTIER_SURFACES),
        "n_frontier_mentions": len([record for record in all_records if "surface" in record]),
        "n_guarded_frontier_mentions": len([record for record in all_records if record.get("guarded") is True]),
        "n_unguarded_frontier_mentions": len([record for record in all_records if record.get("guarded") is False]),
        "n_promoted_frontier_claims": len(all_promotions),
        "frontier_items": frontier_rows,
        "records": all_records,
        "promoted_claims": all_promotions,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def frontier_integrity_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Frontier Integrity Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Frontier items: {payload.get('n_frontier_items')}",
        f"- Claims loaded: {payload.get('n_claims_loaded')}",
        f"- Publication surfaces: {payload.get('n_publication_surfaces')}",
        f"- Frontier mentions: {payload.get('n_frontier_mentions')}",
        f"- Guarded frontier mentions: {payload.get('n_guarded_frontier_mentions')}",
        f"- Unguarded frontier mentions: {payload.get('n_unguarded_frontier_mentions')}",
        f"- Promoted frontier claims: {payload.get('n_promoted_frontier_claims')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
        "## Frontier Items",
        "",
    ]
    for item in payload.get("frontier_items") or []:
        lines.append(
            f"- `{item.get('frontier_id')}`: mentions={item.get('n_mentions')}, "
            f"guarded={item.get('n_guarded_mentions')}, promoted_claims={item.get('n_promoted_claims')}, "
            f"required_artifacts={item.get('n_required_artifacts')}, absent_required_artifacts={item.get('n_missing_required_artifacts')}"
        )
    issues = payload.get("issues") or []
    if issues:
        lines.extend(["", "## Issues", ""])
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.extend(
            [
                "",
                "The remaining ideal robotics endpoints are explicitly framed as limitations, blockers, or future work and are not promoted as verified results.",
            ]
        )
    lines.append("")
    return "\n".join(lines)
