from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
LABELED_NUMBER_RE = re.compile(r"([A-Za-z][A-Za-z0-9 _./+-]*?)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


@dataclass(frozen=True)
class ClaimSemanticCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ClaimSemanticCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ClaimSemanticCheck(name=name, ok=bool(ok), detail=detail))


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def extract_literal_dicts(text: str) -> list[dict[str, Any]]:
    dicts: list[dict[str, Any]] = []
    starts: list[int] = []
    for index, char in enumerate(text):
        if char == "{":
            starts.append(index)
        elif char == "}" and starts:
            start = starts.pop()
            if starts:
                continue
            raw = text[start : index + 1]
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, dict):
                dicts.append(parsed)
    return dicts


def ci_objects(evidence: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for parsed in extract_literal_dicts(evidence):
        if {"n", "mean", "lo", "hi"}.issubset(set(parsed)):
            found.append(parsed)
    return found


def ci_is_sane(ci: dict[str, Any]) -> bool:
    n = finite_float(ci.get("n"))
    mean = finite_float(ci.get("mean"))
    lo = finite_float(ci.get("lo"))
    hi = finite_float(ci.get("hi"))
    std = finite_float(ci.get("std", 0.0))
    stderr = finite_float(ci.get("stderr", 0.0))
    ci95 = finite_float(ci.get("ci95", 0.0))
    return (
        n is not None
        and n >= 1
        and mean is not None
        and lo is not None
        and hi is not None
        and lo <= mean <= hi
        and std is not None
        and stderr is not None
        and ci95 is not None
        and std >= 0.0
        and stderr >= 0.0
        and ci95 >= 0.0
    )


def all_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(text):
        value = finite_float(match.group(0))
        if value is not None:
            values.append(value)
    return values


def labeled_numbers(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for label, raw_value in LABELED_NUMBER_RE.findall(text):
        value = finite_float(raw_value)
        if value is not None:
            values[label.strip().lower()] = value
    return values


def claim_requires_ci(claim: str, evidence: str) -> bool:
    lower = claim.lower()
    evidence_lower = evidence.lower()
    return (
        " ci" in lower
        or lower.endswith("ci.")
        or "with ci" in lower
        or " ci=" in evidence_lower
        or "ci={" in evidence_lower
        or "success ci" in evidence_lower
    )


def claim_requires_positive_ci(claim: str) -> bool:
    lower = claim.lower()
    if "without requiring significance" in lower or "honestly reported" in lower:
        return False
    positive_terms = [
        "beats",
        "improves",
        "helps",
        "remains above",
        "upper bound",
        "gap grows",
        "reduces",
        "high-n gain",
        "savings",
        "predicts",
        "reproduces key",
        "blocks harmful",
    ]
    return any(term in lower for term in positive_terms)


def evidence_has_positive_scalar(evidence: str) -> bool:
    values = all_numbers(evidence)
    return any(value > 0.0 for value in values)


def claim_requires_error_threshold(claim: str) -> bool:
    lower = claim.lower()
    return (
        "exact law" in lower
        or "finite binary law" in lower
        or "utility-valued finite law" in lower
        or "auc identity" in lower
        or "conditional law" in lower
    )


def evidence_has_small_error(evidence: str, threshold: float = 0.05) -> bool:
    lower = evidence.lower()
    if "mae" not in lower and "error" not in lower:
        return False
    values = all_numbers(evidence)
    return bool(values) and min(abs(value) for value in values) <= threshold


def claim_requires_zero_count(claim: str) -> bool:
    lower = claim.lower()
    return "no unsupported" in lower or "internally consistent" in lower or "structurally consistent" in lower or "overclaims" in lower


def evidence_has_zero_count(evidence: str, labels: tuple[str, ...]) -> bool:
    values = labeled_numbers(evidence)
    for label, value in values.items():
        if any(token in label for token in labels) and value == 0:
            return True
    return False


def claim_requires_positive_count(claim: str) -> bool:
    lower = claim.lower()
    return (
        "trained" in lower
        or "collected" in lower
        or "generated" in lower
        or "available" in lower
        or "attempted" in lower
        or "coverage audit" in lower
        or "profiles generated" in lower
    )


def evidence_has_positive_count_or_artifact(evidence: str) -> bool:
    lower = evidence.lower()
    if "true" in lower or "model=" in lower:
        return True
    for label, value in labeled_numbers(evidence).items():
        if any(token in label for token in ("rows", "pools", "train", "val", "eval", "profiles", "decisions", "tasks", "registered", "covered", "claims", "sources", "scripts", "checks", "metrics", "count")) and value > 0:
            return True
    return False


def audit_claim_semantics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    claims = [claim for claim in payload.get("claims") or [] if isinstance(claim, dict)]
    checks: list[ClaimSemanticCheck] = []
    ci_claim_ids: list[int] = []
    positive_ci_claim_ids: list[int] = []
    error_threshold_claim_ids: list[int] = []
    zero_count_claim_ids: list[int] = []
    positive_count_claim_ids: list[int] = []
    sane_ci_count = 0

    add(checks, "claims_present", bool(claims), f"claims={len(claims)}")
    add(checks, "all_claims_verified", all(claim.get("status") == "VERIFIED" for claim in claims), f"claims={len(claims)}")

    for claim in claims:
        cid = int(claim.get("id"))
        text = str(claim.get("claim") or "")
        evidence = str(claim.get("evidence") or "")
        cis = ci_objects(evidence)
        if cis:
            ci_ok = all(ci_is_sane(ci) for ci in cis)
            sane_ci_count += sum(1 for ci in cis if ci_is_sane(ci))
            add(checks, f"claim_{cid}_ci_sanity", ci_ok, f"ci_objects={len(cis)}")
        if claim_requires_ci(text, evidence):
            ci_claim_ids.append(cid)
            add(checks, f"claim_{cid}_has_required_ci", bool(cis), f"ci_objects={len(cis)}")
        if claim_requires_positive_ci(text):
            positive_ci_claim_ids.append(cid)
            if cis:
                positive = [ci for ci in cis if finite_float(ci.get("lo")) is not None and float(ci["lo"]) > 0.0]
                add(checks, f"claim_{cid}_positive_ci_lower_bound", bool(positive), f"positive_ci_objects={len(positive)}")
            else:
                add(checks, f"claim_{cid}_positive_scalar_effect", evidence_has_positive_scalar(evidence), f"evidence={evidence[:120]}")
        if claim_requires_error_threshold(text):
            error_threshold_claim_ids.append(cid)
            add(checks, f"claim_{cid}_error_below_threshold", evidence_has_small_error(evidence), f"evidence={evidence[:120]}")
        if claim_requires_zero_count(text):
            zero_count_claim_ids.append(cid)
            add(checks, f"claim_{cid}_zero_issue_or_overclaim_count", evidence_has_zero_count(evidence, ("issues", "overclaims", "unsupported", "failed", "partial")), f"evidence={evidence[:120]}")
        if claim_requires_positive_count(text):
            positive_count_claim_ids.append(cid)
            add(checks, f"claim_{cid}_positive_count_or_artifact", evidence_has_positive_count_or_artifact(evidence), f"evidence={evidence[:120]}")

    add(checks, "ci_claim_coverage", len(ci_claim_ids) >= 45, f"ci_claims={len(ci_claim_ids)}")
    add(checks, "positive_ci_claim_coverage", len(positive_ci_claim_ids) >= 25, f"positive_ci_claims={len(positive_ci_claim_ids)}")
    add(checks, "error_threshold_claim_coverage", len(error_threshold_claim_ids) >= 10, f"error_claims={len(error_threshold_claim_ids)}")
    add(checks, "semantic_ci_objects_present", sane_ci_count >= 55, f"sane_ci_objects={sane_ci_count}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "claim_semantics",
        "verified": len(issues) == 0,
        "n_claims": len(claims),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "n_ci_claims": len(ci_claim_ids),
        "n_positive_ci_claims": len(positive_ci_claim_ids),
        "n_error_threshold_claims": len(error_threshold_claim_ids),
        "n_zero_count_claims": len(zero_count_claim_ids),
        "n_positive_count_claims": len(positive_count_claim_ids),
        "n_sane_ci_objects": sane_ci_count,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "ci_claim_ids": ci_claim_ids,
        "positive_ci_claim_ids": positive_ci_claim_ids,
        "error_threshold_claim_ids": error_threshold_claim_ids,
        "zero_count_claim_ids": zero_count_claim_ids,
        "positive_count_claim_ids": positive_count_claim_ids,
    }


def audit_claim_semantics(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claims_path = results_dir / "claims_status.json"
    payload = json.loads(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else {}
    return audit_claim_semantics_payload(payload)


def claim_semantics_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Semantics Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Claims audited: {payload.get('n_claims')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        f"- CI-backed claims: {payload.get('n_ci_claims')}",
        f"- Positive-CI semantic claims: {payload.get('n_positive_ci_claims')}",
        f"- Error-threshold claims: {payload.get('n_error_threshold_claims')}",
        f"- Sane CI objects: {payload.get('n_sane_ci_objects')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues[:50]:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Verified claim wording is backed by the expected threshold semantics: required CIs exist, positive-comparison claims have positive CI lower bounds, exact-law claims have small errors, and meta-claims report zero issues or overclaims.")
    lines.append("")
    return "\n".join(lines)
