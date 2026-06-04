from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VERIFICATION_RE = re.compile(r"\b(?:verified|validation|validated)\b", re.I)
BENCHMARK_RE = re.compile(
    r"\b(?:benchmark|ManiSkill|Gymnasium|Meta-World|RoboSuite|RoboCasa|LIBERO|Fetch|Reacher)\b",
    re.I,
)
VISUAL_RE = re.compile(r"\b(?:visual|RGB|RGB-D|frame)\b", re.I)
POLICY_SUCCESS_RE = re.compile(r"\b(?:policy|success|successes|sparse-success)\b", re.I)
SUITE_FAMILY_RE = re.compile(r"\b(?:suite|family|all-ten|\d+-task|tasks?)\b", re.I)
MODEL_RE = re.compile(r"\b(?:trained|evaluated|model|WAM-lite|learned WAM)\b", re.I)
RISKY_FULL_RE = re.compile(r"\b(?:full|real[- ]robot|hardware-in-the-loop|modern VLA|DreamZero|UWM|universal WAM)\b", re.I)

NUMERIC_SCOPE_RE = re.compile(
    r"\b(?:"
    r"MAE|RMSE|AUC|CI|corr|correlation|utility|success|gap|delta|"
    r"envs?|tasks?|rows?|pools?|rollouts?|train|val|validation|eval|episodes|successes|seeds?|"
    r"checks?|issues?|files?|bytes|tables?|figures?|models?|arrays?|predictors?|"
    r"control|registered|covered|candidates|nondegenerate|profiles|decisions|attempted|count|N\d+"
    r")\b",
    re.I,
)
CONCRETE_SCOPE_RE = re.compile(
    r"(?:"
    r"benchmark=|envs?=|tasks?=|rows?=|pools?=|rollouts?=|train=|val=|eval=|episodes=|successes=|"
    r"Reacher-v5|FetchReach|FetchPush|FetchPickAndPlace|PickCube|PushCube|PegInsertion|"
    r"Meta-World|reach-v3|push-v3|drawer-open-v3|RoboSuite|Lift|Stack|Door|"
    r"RoboCasa|robocasa/|LIBERO|libero_|ManiSkill|Gymnasium|Panda|Sawyer|"
    r"smoke|probe|open-loop|closed-loop|state-mode|state benchmark|rollout-pool|sparse-success|optional|"
    r"toy|RGB|RGB-D|frame_std|model=|artifact=|control="
    r")",
    re.I,
)
GUARDED_RISK_RE = re.compile(
    r"\b(?:not|no|without|future|discussion|limitation|blocker|probe|attempt|optional|not claimed)\b",
    re.I,
)


@dataclass(frozen=True)
class ClaimScopeCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ClaimScopeCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ClaimScopeCheck(name=name, ok=bool(ok), detail=detail))


def load_claims(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def has_numeric_scope(text: str) -> bool:
    return bool(NUMERIC_SCOPE_RE.search(text) and re.search(r"\d", text))


def has_concrete_scope(text: str) -> bool:
    return bool(CONCRETE_SCOPE_RE.search(text))


def has_guarded_risk_scope(text: str) -> bool:
    return bool(GUARDED_RISK_RE.search(text))


def claim_scope_records(claim: dict[str, Any]) -> list[str]:
    text = str(claim.get("claim") or "")
    categories: list[str] = []
    if VERIFICATION_RE.search(text):
        categories.append("verification")
    if BENCHMARK_RE.search(text):
        categories.append("benchmark")
    if VISUAL_RE.search(text):
        categories.append("visual")
    if POLICY_SUCCESS_RE.search(text):
        categories.append("policy_success")
    if SUITE_FAMILY_RE.search(text):
        categories.append("suite_family")
    if MODEL_RE.search(text):
        categories.append("model")
    if RISKY_FULL_RE.search(text):
        categories.append("risky_full")
    return categories


def audit_claim_scope_payload(payload: dict[str, Any]) -> dict[str, Any]:
    claims = [claim for claim in payload.get("claims") or [] if isinstance(claim, dict)]
    checks: list[ClaimScopeCheck] = []
    records: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}

    add(checks, "claims_present", bool(claims), f"claims={len(claims)}")

    for claim in claims:
        cid = int(claim.get("id"))
        claim_text = str(claim.get("claim") or "")
        evidence = str(claim.get("evidence") or "")
        combined = f"{claim_text} {evidence}"
        categories = claim_scope_records(claim)
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
            records.append(
                {
                    "claim_id": cid,
                    "category": category,
                    "claim": claim_text,
                    "evidence": evidence,
                    "has_numeric_scope": has_numeric_scope(combined),
                    "has_concrete_scope": has_concrete_scope(combined),
                }
            )

        if "verification" in categories:
            add(
                checks,
                f"claim_{cid}_verified_wording_has_metric_or_scope",
                has_numeric_scope(combined) or has_concrete_scope(combined),
                f"claim={claim_text}; evidence={evidence[:160]}",
            )
        if "benchmark" in categories:
            adapter_available = "adapter available" in claim_text.lower() and "attempted=" in evidence.lower() and "any_available=" in evidence.lower()
            visual_optional = "visual optional" in claim_text.lower() and ("optional" in combined.lower() or "frame_std" in combined.lower())
            blocker_probe = "probe" in claim_text.lower() and ("blocker=" in evidence.lower() or "pinocchio=" in evidence.lower())
            add(
                checks,
                f"claim_{cid}_benchmark_wording_has_benchmark_scope",
                adapter_available or visual_optional or blocker_probe or (has_concrete_scope(combined) and has_numeric_scope(combined)),
                f"claim={claim_text}; evidence={evidence[:180]}",
            )
        if "visual" in categories:
            add(
                checks,
                f"claim_{cid}_visual_wording_has_observation_scope",
                has_concrete_scope(combined) and ("visual" in combined.lower() or "rgb" in combined.lower() or "frame" in combined.lower()),
                f"claim={claim_text}; evidence={evidence[:180]}",
            )
        if "policy_success" in categories:
            policy_mode = re.search(r"\b(?:smoke|sparse-success|scripted|BC|behavior|open-loop|closed-loop)\b", combined, re.I)
            policy_counts = re.search(r"\b(?:success(?:es)?|episodes|eval|rows|CI|tasks?)\b", evidence, re.I) and re.search(r"\d", evidence)
            add(
                checks,
                f"claim_{cid}_policy_success_wording_has_smoke_or_eval_scope",
                bool(policy_mode and policy_counts),
                f"claim={claim_text}; evidence={evidence[:180]}",
            )
        if "suite_family" in categories:
            add(
                checks,
                f"claim_{cid}_suite_family_wording_has_task_or_env_scope",
                has_concrete_scope(combined) and has_numeric_scope(combined),
                f"claim={claim_text}; evidence={evidence[:180]}",
            )
        if "model" in categories:
            model_scope = re.search(
                r"\b(?:model=|models=|model_path|train=|val=|validation|eval=|rows|samples|corr|MAE|CI|metrics?|loadable|arrays?|predictors?)\b",
                combined,
                re.I,
            )
            add(
                checks,
                f"claim_{cid}_model_wording_has_training_or_eval_scope",
                bool(model_scope),
                f"claim={claim_text}; evidence={evidence[:180]}",
            )
        if "risky_full" in categories:
            add(
                checks,
                f"claim_{cid}_risky_full_wording_is_guarded",
                has_guarded_risk_scope(combined),
                f"claim={claim_text}; evidence={evidence[:180]}",
            )

    add(checks, "scope_mentions_present", len(records) >= 55, f"records={len(records)}")
    add(checks, "benchmark_scope_coverage", category_counts.get("benchmark", 0) >= 25, f"benchmark={category_counts.get('benchmark', 0)}")
    add(checks, "verification_scope_coverage", category_counts.get("verification", 0) >= 35, f"verification={category_counts.get('verification', 0)}")
    add(checks, "visual_scope_coverage", category_counts.get("visual", 0) >= 8, f"visual={category_counts.get('visual', 0)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "claim_scope_audit",
        "verified": len(issues) == 0,
        "n_claims": len(claims),
        "n_scope_mentions": len(records),
        "category_counts": category_counts,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "records": records,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def audit_claim_scope(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    return audit_claim_scope_payload(load_claims(results_dir / "claims_status.json"))


def claim_scope_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Scope Audit Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Claims audited: {payload.get('n_claims')}",
        f"- Scope mentions: {payload.get('n_scope_mentions')}",
        f"- Category counts: {payload.get('category_counts')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues[:80]:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Broad claim wording is scoped by concrete task/env names, sample counts, metrics, CIs, mode qualifiers, smoke/probe labels, or explicit risk guards.")
    lines.append("")
    return "\n".join(lines)
