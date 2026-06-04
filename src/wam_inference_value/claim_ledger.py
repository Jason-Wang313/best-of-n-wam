from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_STATUSES = ("VERIFIED", "PARTIAL", "UNSUPPORTED", "FAILED")
VALID_STATUS_SET = set(VALID_STATUSES)
COUNT_KEYS = {
    "VERIFIED": "num_verified",
    "PARTIAL": "num_partial",
    "UNSUPPORTED": "num_unsupported",
    "FAILED": "num_failed",
}


@dataclass(frozen=True)
class ClaimLedgerCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ClaimLedgerCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ClaimLedgerCheck(name=name, ok=bool(ok), detail=detail))


def _int_ids(claims: list[dict[str, Any]]) -> tuple[list[int], list[Any]]:
    ids: list[int] = []
    invalid: list[Any] = []
    for claim in claims:
        raw = claim.get("id")
        try:
            cid = int(raw)
        except (TypeError, ValueError):
            invalid.append(raw)
            continue
        if cid != raw and not (isinstance(raw, str) and raw.isdigit()):
            invalid.append(raw)
            continue
        ids.append(cid)
    return ids, invalid


def status_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    return {status: sum(claim.get("status") == status for claim in claims) for status in VALID_STATUSES}


def audit_claim_ledger_payload(
    payload: dict[str, Any],
    *,
    claims_status_md: str | None = None,
    claims_report_md: str | None = None,
) -> dict[str, Any]:
    checks: list[ClaimLedgerCheck] = []
    claims = payload.get("claims")
    if not isinstance(claims, list):
        claims = []
    claims = [claim for claim in claims if isinstance(claim, dict)]
    ids, invalid_ids = _int_ids(claims)
    unique_ids = sorted(set(ids))
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    max_id = max(unique_ids) if unique_ids else 0
    missing = [cid for cid in range(1, max_id + 1) if cid not in unique_ids]
    statuses = [str(claim.get("status")) for claim in claims]
    invalid_statuses = sorted({status for status in statuses if status not in VALID_STATUS_SET})
    empty_claims = [claim.get("id") for claim in claims if not str(claim.get("claim") or "").strip()]
    empty_evidence = [claim.get("id") for claim in claims if not str(claim.get("evidence") or "").strip()]
    sorted_ids = ids == sorted(ids)
    counts = status_counts(claims)

    add(checks, "claims_present", len(claims) > 0, f"claims={len(claims)}")
    add(checks, "claim_ids_parse", not invalid_ids and len(ids) == len(claims), f"invalid_ids={invalid_ids}")
    add(checks, "claim_ids_unique", not duplicates, f"duplicates={duplicates}")
    add(checks, "claim_ids_contiguous", not missing and unique_ids == list(range(1, max_id + 1)), f"missing={missing}, max_id={max_id}")
    add(checks, "claim_ids_sorted", sorted_ids, f"first_ids={ids[:10]}, last_ids={ids[-10:]}")
    add(checks, "claim_statuses_valid", not invalid_statuses, f"invalid_statuses={invalid_statuses}")
    add(checks, "claim_text_nonempty", not empty_claims, f"empty_claim_ids={empty_claims}")
    add(checks, "claim_evidence_nonempty", not empty_evidence, f"empty_evidence_ids={empty_evidence}")
    for status, key in COUNT_KEYS.items():
        add(checks, f"{key}_matches_claims", int(payload.get(key) or 0) == counts[status], f"json={payload.get(key)}, computed={counts[status]}")
    total_json = sum(int(payload.get(key) or 0) for key in COUNT_KEYS.values())
    add(checks, "status_counts_sum_to_claim_count", total_json == len(claims), f"status_total={total_json}, claims={len(claims)}")
    add(
        checks,
        "no_nonverified_claims",
        counts["PARTIAL"] == 0 and counts["UNSUPPORTED"] == 0 and counts["FAILED"] == 0,
        f"partial={counts['PARTIAL']}, unsupported={counts['UNSUPPORTED']}, failed={counts['FAILED']}",
    )
    for key in ["readme_overclaims", "paper_overclaims", "report_overclaims", "narrative_overclaims", "overclaims"]:
        values = payload.get(key) or []
        add(checks, f"{key}_empty", isinstance(values, list) and len(values) == 0, f"count={len(values) if isinstance(values, list) else 'n/a'}")

    if claims_status_md is not None:
        md_rows = re.findall(r"^- Claim (\d+): \*\*([A-Z]+)\*\* - ", claims_status_md, flags=re.MULTILINE)
        md_ids = sorted(int(cid) for cid, _ in md_rows)
        md_status_by_id = {int(cid): status for cid, status in md_rows}
        json_status_by_id = {int(claim["id"]): str(claim.get("status")) for claim in claims if "id" in claim}
        add(checks, "claims_status_md_claim_count", len(md_rows) == len(claims), f"md={len(md_rows)}, json={len(claims)}")
        add(checks, "claims_status_md_id_set", md_ids == unique_ids, f"md_missing={sorted(set(unique_ids) - set(md_ids))}")
        mismatched = [cid for cid, status in json_status_by_id.items() if md_status_by_id.get(cid) != status]
        add(checks, "claims_status_md_statuses", not mismatched, f"mismatched_ids={mismatched[:10]}")

    if claims_report_md is not None:
        for status, key in COUNT_KEYS.items():
            label = key.removeprefix("num_")
            expected = f"- {label}: `{counts[status]}`"
            add(checks, f"claims_report_{label}_count", expected in claims_report_md, f"expected={expected}")
        for key, label in [
            ("readme_overclaims", "README overclaims"),
            ("paper_overclaims", "paper_outline overclaims"),
            ("report_overclaims", "report overclaims"),
        ]:
            expected = f"- {label}: `{len(payload.get(key) or [])}`"
            add(checks, f"claims_report_{key}_count", expected in claims_report_md, f"expected={expected}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "claim_ledger_integrity",
        "verified": len(issues) == 0,
        "n_claims": len(claims),
        "max_claim_id": max_id,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "status_counts": counts,
        "missing_ids": missing,
        "duplicate_ids": duplicates,
        "invalid_statuses": invalid_statuses,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def audit_claim_ledger(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claims_json = results_dir / "claims_status.json"
    claims_md = results_dir / "claims_status.md"
    claims_report = root / "reports" / "claims_report.md"
    payload = json.loads(claims_json.read_text(encoding="utf-8")) if claims_json.exists() else {}
    return audit_claim_ledger_payload(
        payload,
        claims_status_md=claims_md.read_text(encoding="utf-8") if claims_md.exists() else None,
        claims_report_md=claims_report.read_text(encoding="utf-8") if claims_report.exists() else None,
    )


def claim_ledger_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Ledger Integrity Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Claims: {payload.get('n_claims')}",
        f"- Max claim ID: {payload.get('max_claim_id')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        f"- Status counts: {payload.get('status_counts')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Claim IDs, statuses, counts, nonempty evidence, overclaim arrays, and generated Markdown summaries are internally consistent.")
    lines.append("")
    return "\n".join(lines)
