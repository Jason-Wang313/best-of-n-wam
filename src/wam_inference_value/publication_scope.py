from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PUBLICATION_SURFACES = [
    "README.md",
    "paper_outline.md",
    "reports/paper_result_summary.md",
    "reports/final_decision_report.md",
    "reports/reviewer_risk_assessment.md",
]
RISK_PATTERNS = {
    "real_robot": re.compile(r"\breal[- ]robot\b|\bhardware-in-the-loop\b", re.I),
    "modern_vla": re.compile(r"\bmodern\s+VLA\b", re.I),
    "full_robocasa": re.compile(r"\bfull\s+RoboCasa-wide\b", re.I),
    "maniskill_rgb_or_ee": re.compile(r"\bManiSkill\b.*\b(?:RGB|RGB-D|RGB/RGB-D|visual|EE-control|end-effector)\b", re.I),
    "universal_wam": re.compile(r"\buniversal\s+WAM\b|\bRobot Chinchilla\b", re.I),
    "dreamzero_uwm": re.compile(r"\b(?:DreamZero|UWM)\b", re.I),
}
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
    "unresolved",
    "limitation",
    "blocker",
    "unavailable",
    "failed",
    "probe",
    "attempt",
    "remains",
    "weaker",
    "absence",
    "beyond",
    "next step",
    "separate",
    "guarded",
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
]


@dataclass(frozen=True)
class PublicationScopeCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[PublicationScopeCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(PublicationScopeCheck(name=name, ok=bool(ok), detail=detail))


def load_claims(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def verified_claim_count(claims_payload: dict[str, Any]) -> int:
    return sum(1 for claim in claims_payload.get("claims") or [] if isinstance(claim, dict) and claim.get("status") == "VERIFIED")


def current_heading(line: str, previous: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return stripped.lstrip("#").strip()
    return previous


def safe_context(text: str, heading: str) -> bool:
    haystack = f"{heading}\n{text}".lower()
    if any(token in haystack for token in SAFE_CONTEXT_TOKENS):
        return True
    heading_lower = heading.lower()
    return any(token in heading_lower for token in SAFE_HEADING_TOKENS)


def scan_surface(root: Path, relative: str) -> list[dict[str, Any]]:
    path = root / relative
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    heading = ""
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        heading = current_heading(line, heading)
        stripped = line.strip()
        if not stripped:
            continue
        for name, pattern in RISK_PATTERNS.items():
            if pattern.search(stripped):
                guarded = safe_context(stripped, heading)
                records.append(
                    {
                        "surface": relative,
                        "line": line_no,
                        "heading": heading,
                        "pattern": name,
                        "guarded": guarded,
                        "text": stripped,
                    }
                )
    return records


def audit_publication_scope(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    records: list[dict[str, Any]] = []
    missing_surfaces: list[str] = []
    for relative in PUBLICATION_SURFACES:
        if not (root / relative).exists():
            missing_surfaces.append(relative)
        records.extend(scan_surface(root, relative))
    unguarded = [record for record in records if not record.get("guarded")]
    by_surface = {relative: sum(1 for record in records if record.get("surface") == relative) for relative in PUBLICATION_SURFACES}
    by_pattern = {name: sum(1 for record in records if record.get("pattern") == name) for name in RISK_PATTERNS}
    claims_payload = load_claims(results_dir / "claims_status.json")
    n_verified_claims = verified_claim_count(claims_payload)

    checks: list[PublicationScopeCheck] = []
    add(checks, "publication_surfaces_exist", not missing_surfaces, f"missing={missing_surfaces}")
    add(checks, "risk_mentions_present", len(records) >= 20, f"mentions={len(records)}")
    add(checks, "all_risk_mentions_guarded", not unguarded, f"unguarded={len(unguarded)}")
    add(checks, "readme_scope_mentions_present", by_surface.get("README.md", 0) >= 5, f"mentions={by_surface.get('README.md', 0)}")
    add(checks, "paper_scope_mentions_present", by_surface.get("paper_outline.md", 0) >= 5, f"mentions={by_surface.get('paper_outline.md', 0)}")
    add(checks, "future_only_categories_covered", sum(1 for count in by_pattern.values() if count > 0) >= 5, f"patterns={by_pattern}")
    add(checks, "current_claim_ledger_loaded", n_verified_claims >= 120, f"verified_claims={n_verified_claims}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "publication_scope",
        "verified": len(issues) == 0,
        "n_publication_surfaces": len(PUBLICATION_SURFACES),
        "n_existing_surfaces": len(PUBLICATION_SURFACES) - len(missing_surfaces),
        "n_risk_mentions": len(records),
        "n_guarded_mentions": len(records) - len(unguarded),
        "n_unguarded_mentions": len(unguarded),
        "n_verified_claims": n_verified_claims,
        "mentions_by_surface": by_surface,
        "mentions_by_pattern": by_pattern,
        "records": records,
        "unguarded_records": unguarded,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "n_checks": len(checks),
        "n_issues": len(issues),
    }


def publication_scope_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Publication Scope Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Publication surfaces: {payload.get('n_publication_surfaces')}",
        f"- Existing surfaces: {payload.get('n_existing_surfaces')}",
        f"- Risk mentions: {payload.get('n_risk_mentions')}",
        f"- Guarded mentions: {payload.get('n_guarded_mentions')}",
        f"- Unguarded mentions: {payload.get('n_unguarded_mentions')}",
        f"- Verified claims loaded: {payload.get('n_verified_claims')}",
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
        unguarded = payload.get("unguarded_records") or []
        if unguarded:
            lines.append("")
            lines.append("## Unguarded Mentions")
            lines.append("")
            for record in unguarded[:20]:
                lines.append(f"- `{record.get('surface')}:{record.get('line')}` `{record.get('pattern')}`: {record.get('text')}")
    else:
        lines.append("All risky publication-surface phrases are guarded by limitation, blocker, discussion, future-work, or explicit non-claim context.")
    lines.append("")
    return "\n".join(lines)
