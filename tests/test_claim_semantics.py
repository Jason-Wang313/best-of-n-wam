from wam_inference_value.claim_semantics import audit_claim_semantics_payload


def payload(claims):
    return {"claims": claims}


def test_claim_semantics_accepts_positive_ci_comparison() -> None:
    audit = audit_claim_semantics_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "Useful scorer beats random with CI.",
                    "status": "VERIFIED",
                    "evidence": "learned-random CI={'n': 5, 'mean': 1.0, 'std': 0.1, 'stderr': 0.04, 'ci95': 0.08, 'lo': 0.92, 'hi': 1.08}",
                }
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_has_required_ci" not in issue_names
    assert "claim_1_positive_ci_lower_bound" not in issue_names


def test_claim_semantics_flags_nonpositive_beats_ci() -> None:
    audit = audit_claim_semantics_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "Useful scorer beats random with CI.",
                    "status": "VERIFIED",
                    "evidence": "learned-random CI={'n': 5, 'mean': 0.1, 'std': 0.2, 'stderr': 0.1, 'ci95': 0.2, 'lo': -0.1, 'hi': 0.3}",
                }
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_positive_ci_lower_bound" in issue_names


def test_claim_semantics_requires_ci_for_positive_empirical_claims() -> None:
    audit = audit_claim_semantics_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "Pilot-to-heldout improves with K.",
                    "status": "VERIFIED",
                    "evidence": "relative MAE reduction=0.72",
                }
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_positive_ci_required" in issue_names


def test_claim_semantics_flags_exact_law_without_small_error() -> None:
    audit = audit_claim_semantics_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "Exact law verified.",
                    "status": "VERIFIED",
                    "evidence": "utility MAE=0.5",
                }
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_error_below_threshold" in issue_names


def test_claim_semantics_flags_nonzero_overclaims() -> None:
    audit = audit_claim_semantics_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "README has no unsupported claims.",
                    "status": "VERIFIED",
                    "evidence": "README overclaims=2",
                }
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_zero_issue_or_overclaim_count" in issue_names
