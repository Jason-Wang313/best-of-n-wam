from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REFERENCE_SURFACES = [
    "README.md",
    "paper_outline.md",
    "reports/final_decision_report.md",
    "reports/paper_result_summary.md",
    "reports/reviewer_risk_assessment.md",
]
VERIFIED_CLAIM_RE = re.compile(r"\bVERIFIED\s+CLAIM\s+(\d+)\b", re.I)


@dataclass(frozen=True)
class ClaimReferenceCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ClaimReferenceCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ClaimReferenceCheck(name=name, ok=bool(ok), detail=detail))


def load_claims(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload.get("claims") if isinstance(payload, dict) else []
    return {int(claim["id"]): claim for claim in claims or [] if isinstance(claim, dict) and "id" in claim}


def scan_surface(root: Path, relative: str) -> list[dict[str, Any]]:
    path = root / relative
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in VERIFIED_CLAIM_RE.finditer(line):
            records.append(
                {
                    "surface": relative,
                    "line": line_no,
                    "claim_id": int(match.group(1)),
                    "text": line.strip(),
                }
            )
    return records


def audit_claim_references(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claims = load_claims(results_dir / "claims_status.json")
    missing_surfaces = [surface for surface in REFERENCE_SURFACES if not (root / surface).exists()]
    records: list[dict[str, Any]] = []
    for surface in REFERENCE_SURFACES:
        records.extend(scan_surface(root, surface))

    missing_claims = [record for record in records if record["claim_id"] not in claims]
    nonverified_claims = [
        {**record, "status": claims.get(record["claim_id"], {}).get("status")}
        for record in records
        if record["claim_id"] in claims and claims[record["claim_id"]].get("status") != "VERIFIED"
    ]
    unique_ids = sorted({record["claim_id"] for record in records})
    by_surface = {surface: sum(1 for record in records if record["surface"] == surface) for surface in REFERENCE_SURFACES}

    checks: list[ClaimReferenceCheck] = []
    add(checks, "reference_surfaces_exist", not missing_surfaces, f"missing={missing_surfaces}")
    add(checks, "claim_ledger_loaded", len(claims) >= 120, f"claims={len(claims)}")
    add(checks, "verified_claim_references_present", len(records) >= 10, f"references={len(records)}")
    add(checks, "unique_claim_references_present", len(unique_ids) >= 10, f"unique={len(unique_ids)}")
    add(checks, "all_referenced_claims_exist", not missing_claims, f"missing={missing_claims[:10]}")
    add(checks, "all_referenced_claims_verified", not nonverified_claims, f"nonverified={nonverified_claims[:10]}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "claim_reference_integrity",
        "verified": len(issues) == 0,
        "n_claims_loaded": len(claims),
        "n_reference_surfaces": len(REFERENCE_SURFACES),
        "n_existing_surfaces": len(REFERENCE_SURFACES) - len(missing_surfaces),
        "n_references": len(records),
        "n_unique_referenced_claims": len(unique_ids),
        "references_by_surface": by_surface,
        "referenced_claim_ids": unique_ids,
        "records": records,
        "missing_claim_references": missing_claims,
        "nonverified_claim_references": nonverified_claims,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def claim_reference_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Reference Integrity Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Claims loaded: {payload.get('n_claims_loaded')}",
        f"- Reference surfaces: {payload.get('n_reference_surfaces')}",
        f"- Existing surfaces: {payload.get('n_existing_surfaces')}",
        f"- VERIFIED CLAIM references: {payload.get('n_references')}",
        f"- Unique referenced claims: {payload.get('n_unique_referenced_claims')}",
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
        lines.append("Every explicit `VERIFIED CLAIM N` narrative reference resolves to a current verified claim in `results/claims_status.json`.")
    lines.append("")
    return "\n".join(lines)
