from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.abstract_claim_support import APPROVED_ABSTRACT_CLAIMS, audit_abstract_claim_support


def write_claims(results: Path, verified_ids: set[int]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    claims = []
    for claim_id in range(1, 123):
        claims.append(
            {
                "id": claim_id,
                "claim": f"claim {claim_id}",
                "status": "VERIFIED" if claim_id in verified_ids else "UNSUPPORTED",
                "evidence": f"value={claim_id}",
            }
        )
    (results / "claims_status.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")


def write_report(root: Path, abstract_claims: list[str]) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Decision Report",
        "",
        "## 4. Abstract Claims",
        "",
        *[f"- {claim}" for claim in abstract_claims],
        "",
        "## 5. Discussion-Only Claims",
        "",
        "- Modern VLA-style sparse-success LIBERO policy performance and full RoboCasa-wide learned-WAM validation.",
        "- ManiSkill RGB/RGB-D or EE-control validation.",
        "- ManiSkill RGB/RGB-D benchmark WAM validation.",
        "- Universal WAM training and train-inference scaling.",
        "",
        "## 6. Skeptical Reviewer Attack",
        "",
        "The project still lacks real robot artifacts, modern VLA-style sparse-success LIBERO policy validation, full RoboCasa-wide learned-WAM validation, and ManiSkill visual or EE-control validation.",
        "",
        "## 8. Unresolved",
        "",
        "- Real robot validation.",
        "- Modern VLA-style sparse-success LIBERO policy evaluation.",
        "- Full RoboCasa-wide learned-WAM rollout collection.",
        "- ManiSkill RGB/RGB-D or end-effector-control validation.",
    ]
    (reports / "final_decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def required_ids() -> set[int]:
    ids: set[int] = set()
    for claim_ids in APPROVED_ABSTRACT_CLAIMS.values():
        ids.update(claim_ids)
    return ids


def test_abstract_claim_support_accepts_exact_supported_headlines(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results, set(range(1, 123)))
    write_report(tmp_path, list(APPROVED_ABSTRACT_CLAIMS))

    payload = audit_abstract_claim_support(tmp_path, results)

    assert payload["verified"] is True
    assert payload["n_abstract_claims"] == 4
    assert payload["n_forbidden_headline_hits"] == 0


def test_abstract_claim_support_rejects_unapproved_headline(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results, set(range(1, 123)))
    write_report(tmp_path, list(APPROVED_ABSTRACT_CLAIMS)[:-1] + ["Real-robot validation is complete."])

    payload = audit_abstract_claim_support(tmp_path, results)

    assert payload["verified"] is False
    assert "abstract_claims_exact_approved_set" in {issue["name"] for issue in payload["issues"]}
    assert payload["n_forbidden_headline_hits"] >= 1


def test_abstract_claim_support_requires_verified_backing_claim_ids(tmp_path: Path) -> None:
    results = tmp_path / "results"
    verified = required_ids()
    verified.discard(43)
    write_claims(results, verified)
    write_report(tmp_path, list(APPROVED_ABSTRACT_CLAIMS))

    payload = audit_abstract_claim_support(tmp_path, results)

    assert payload["verified"] is False
    assert "abstract_claims_have_verified_backing" in {issue["name"] for issue in payload["issues"]}
