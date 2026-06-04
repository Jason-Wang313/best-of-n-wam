from wam_inference_value.claim_scope_audit import audit_claim_scope_payload


def payload(claims):
    return {"claims": claims}


def test_claim_scope_audit_accepts_scoped_benchmark_and_smoke_claims() -> None:
    audit = audit_claim_scope_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "ManiSkill state benchmark suite verified.",
                    "status": "VERIFIED",
                    "evidence": "envs=['PickCube-v1', 'PushCube-v1', 'PegInsertionSide-v1'], pools=30, rows=900, control=pd_joint_delta_pos",
                },
                {
                    "id": 2,
                    "claim": "LIBERO sparse-success scripted policy smoke verified.",
                    "status": "VERIFIED",
                    "evidence": "tasks=['libero_object/0'], episodes=50, successes=50, rows=50, success CI={'n': 50, 'mean': 1.0, 'lo': 1.0, 'hi': 1.0}",
                },
                {
                    "id": 3,
                    "claim": "Benchmark RGB visual WAM exact law verified.",
                    "status": "VERIFIED",
                    "evidence": "benchmark=Reacher-v5, RGB frame rows=750, utility MAE=0.015",
                },
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_benchmark_wording_has_benchmark_scope" not in issue_names
    assert "claim_2_policy_success_wording_has_smoke_or_eval_scope" not in issue_names
    assert "claim_3_visual_wording_has_observation_scope" not in issue_names


def test_claim_scope_audit_rejects_vague_benchmark_validation() -> None:
    audit = audit_claim_scope_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "Benchmark validation verified.",
                    "status": "VERIFIED",
                    "evidence": "verified=True",
                }
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_benchmark_wording_has_benchmark_scope" in issue_names


def test_claim_scope_audit_rejects_unscoped_policy_success() -> None:
    audit = audit_claim_scope_payload(
        payload(
            [
                {
                    "id": 1,
                    "claim": "LIBERO policy success verified.",
                    "status": "VERIFIED",
                    "evidence": "success=1",
                }
            ]
        )
    )

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "claim_1_policy_success_wording_has_smoke_or_eval_scope" in issue_names
