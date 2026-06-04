from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APPROVED_ABSTRACT_CLAIMS: dict[str, list[int]] = {
    "Exact best-of-N inference laws for rollout selection.": [1, 2, 3, 4],
    "The score/utility distribution determines the value of additional rollouts.": [1, 2, 7, 42],
    "Model/scorer mismatch can make best-of-N amplify imagined futures rather than real utility.": [10, 11, 12, 13, 43],
    "Learned and multi-env toy artifacts validate the theory and failure modes.": [22, 23, 24, 25, 26, 27, 28, 29, 30, 45],
}
DISCUSSION_ONLY_MARKERS = [
    "modern VLA",
    "full RoboCasa-wide",
    "ManiSkill RGB/RGB-D",
    "Universal WAM training",
]
UNRESOLVED_MARKERS = [
    "Real robot validation",
    "modern VLA",
    "full RoboCasa-wide",
    "ManiSkill RGB/RGB-D",
]
FORBIDDEN_HEADLINE_PATTERNS = {
    "real_robot": re.compile(r"\breal[- ]robot\b", re.I),
    "modern_vla": re.compile(r"\bmodern\s+VLA\b", re.I),
    "full_robocasa": re.compile(r"\bfull\s+RoboCasa-wide\b", re.I),
    "maniskill_visual_or_ee": re.compile(r"\bManiSkill\s+(?:RGB|RGB-D|RGB/RGB-D|visual|EE|end-effector)\b", re.I),
    "universal_training": re.compile(r"\buniversal\s+WAM\s+(?:training|train-inference|training recipe|training laws?)\b", re.I),
    "dreamzero_uwm": re.compile(r"\b(?:DreamZero|UWM)\b", re.I),
}


@dataclass(frozen=True)
class AbstractClaimCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[AbstractClaimCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(AbstractClaimCheck(name=name, ok=bool(ok), detail=detail))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def section_lines(text: str, heading: str) -> list[str]:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return []
    start = text.find("\n", start)
    if start < 0:
        return []
    next_heading = text.find("\n## ", start + 1)
    section = text[start + 1 :] if next_heading < 0 else text[start + 1 : next_heading]
    lines = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            lines.append(stripped[2:].strip())
    return lines


def section_text(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_heading = text.find("\n## ", start + len(marker))
    return text[start:] if next_heading < 0 else text[start:next_heading]


def verified_claim_ids(claims_payload: dict[str, Any]) -> set[int]:
    verified: set[int] = set()
    for claim in claims_payload.get("claims") or []:
        if isinstance(claim, dict) and claim.get("status") == "VERIFIED":
            try:
                verified.add(int(claim.get("id")))
            except (TypeError, ValueError):
                continue
    return verified


def forbidden_hits(lines: list[str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for line in lines:
        for name, pattern in FORBIDDEN_HEADLINE_PATTERNS.items():
            if pattern.search(line):
                hits.append({"pattern": name, "line": line})
    return hits


def audit_abstract_claim_support(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    final_report = root / "reports" / "final_decision_report.md"
    report_text = final_report.read_text(encoding="utf-8") if final_report.exists() else ""
    claims_payload = load_json(results_dir / "claims_status.json")
    verified_ids = verified_claim_ids(claims_payload)
    abstract_lines = section_lines(report_text, "4. Abstract Claims")
    discussion_lines = section_lines(report_text, "5. Discussion-Only Claims")
    unresolved_text = section_text(report_text, "8. Unresolved")
    reviewer_attack_text = section_text(report_text, "6. Skeptical Reviewer Attack")

    expected_lines = list(APPROVED_ABSTRACT_CLAIMS)
    unexpected = [line for line in abstract_lines if line not in APPROVED_ABSTRACT_CLAIMS]
    missing = [line for line in expected_lines if line not in abstract_lines]
    backing_records = []
    missing_backing: list[dict[str, Any]] = []
    for line in abstract_lines:
        required_ids = APPROVED_ABSTRACT_CLAIMS.get(line, [])
        absent_ids = [claim_id for claim_id in required_ids if claim_id not in verified_ids]
        record = {"claim": line, "required_claim_ids": required_ids, "missing_verified_claim_ids": absent_ids}
        backing_records.append(record)
        if absent_ids:
            missing_backing.append(record)

    discussion_text = "\n".join(discussion_lines)
    discussion_text_lower = discussion_text.lower()
    missing_discussion_markers = [marker for marker in DISCUSSION_ONLY_MARKERS if marker.lower() not in discussion_text_lower]
    limitation_text = reviewer_attack_text + "\n" + unresolved_text
    limitation_text_lower = limitation_text.lower()
    missing_unresolved_markers = [marker for marker in UNRESOLVED_MARKERS if marker.lower() not in limitation_text_lower]
    headline_forbidden_hits = forbidden_hits(abstract_lines)

    checks: list[AbstractClaimCheck] = []
    add(checks, "final_decision_report_exists", final_report.exists(), f"path={final_report}")
    add(checks, "abstract_claims_present", len(abstract_lines) == len(expected_lines), f"claims={len(abstract_lines)}")
    add(checks, "abstract_claims_exact_approved_set", not unexpected and not missing, f"unexpected={unexpected}, missing={missing}")
    add(checks, "abstract_claims_have_verified_backing", not missing_backing, f"missing_backing={missing_backing}")
    add(checks, "headline_has_no_future_only_phrases", not headline_forbidden_hits, f"hits={headline_forbidden_hits}")
    add(checks, "discussion_only_claims_present", len(discussion_lines) >= 4, f"claims={len(discussion_lines)}")
    add(checks, "future_only_claims_are_discussion_scoped", not missing_discussion_markers, f"missing={missing_discussion_markers}")
    add(checks, "unresolved_limitations_are_explicit", not missing_unresolved_markers, f"missing={missing_unresolved_markers}")
    add(checks, "all_backing_claim_ids_verified", len(verified_ids) >= 120, f"verified_claim_ids={len(verified_ids)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "abstract_claim_support",
        "verified": len(issues) == 0,
        "n_abstract_claims": len(abstract_lines),
        "n_approved_abstract_claims": len(expected_lines),
        "n_discussion_only_claims": len(discussion_lines),
        "n_backing_claim_links": sum(len(record["required_claim_ids"]) for record in backing_records),
        "n_forbidden_headline_hits": len(headline_forbidden_hits),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "abstract_claims": abstract_lines,
        "discussion_only_claims": discussion_lines,
        "backing_records": backing_records,
        "forbidden_headline_hits": headline_forbidden_hits,
        "missing_discussion_markers": missing_discussion_markers,
        "missing_unresolved_markers": missing_unresolved_markers,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def abstract_claim_support_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Abstract Claim Support Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Abstract claims: {payload.get('n_abstract_claims')}",
        f"- Approved abstract claims: {payload.get('n_approved_abstract_claims')}",
        f"- Discussion-only claims: {payload.get('n_discussion_only_claims')}",
        f"- Backing claim links: {payload.get('n_backing_claim_links')}",
        f"- Forbidden headline hits: {payload.get('n_forbidden_headline_hits')}",
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
        lines.append("All final-report abstract claims match the approved headline set, have verified backing claim IDs, and keep future-only robotics evidence in discussion or unresolved sections.")
    lines.append("")
    return "\n".join(lines)
