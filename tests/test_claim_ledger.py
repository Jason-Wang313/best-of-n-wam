from pathlib import Path

from wam_inference_value.claim_ledger import audit_claim_ledger_payload


def payload_with_claims(claims):
    return {
        "claims": claims,
        "readme_overclaims": [],
        "paper_overclaims": [],
        "report_overclaims": [],
        "narrative_overclaims": [],
        "overclaims": [],
        "num_verified": sum(claim["status"] == "VERIFIED" for claim in claims),
        "num_partial": sum(claim["status"] == "PARTIAL" for claim in claims),
        "num_unsupported": sum(claim["status"] == "UNSUPPORTED" for claim in claims),
        "num_failed": sum(claim["status"] == "FAILED" for claim in claims),
    }


def test_claim_ledger_accepts_sorted_contiguous_verified_claims():
    claims = [
        {"id": 1, "claim": "First.", "status": "VERIFIED", "evidence": "artifact=a"},
        {"id": 2, "claim": "Second.", "status": "VERIFIED", "evidence": "artifact=b"},
    ]

    payload = audit_claim_ledger_payload(
        payload_with_claims(claims),
        claims_status_md="- Claim 1: **VERIFIED** - First. Evidence: artifact=a\n- Claim 2: **VERIFIED** - Second. Evidence: artifact=b\n",
        claims_report_md=(
            "- verified: `2`\n"
            "- partial: `0`\n"
            "- unsupported: `0`\n"
            "- failed: `0`\n"
            "- README overclaims: `0`\n"
            "- paper_outline overclaims: `0`\n"
            "- report overclaims: `0`\n"
        ),
    )

    assert payload["verified"] is True
    assert payload["n_issues"] == 0


def test_claim_ledger_rejects_duplicate_missing_and_unsorted_ids():
    claims = [
        {"id": 3, "claim": "Third.", "status": "VERIFIED", "evidence": "artifact=d"},
        {"id": 2, "claim": "Second.", "status": "VERIFIED", "evidence": "artifact=b"},
        {"id": 2, "claim": "Duplicate.", "status": "VERIFIED", "evidence": "artifact=c"},
    ]

    payload = audit_claim_ledger_payload(payload_with_claims(claims))

    failures = {check["name"] for check in payload["issues"]}
    assert "claim_ids_unique" in failures
    assert "claim_ids_contiguous" in failures
    assert "claim_ids_sorted" in failures


def test_claim_ledger_requires_all_claims_verified_for_final_gate():
    claims = [
        {"id": 1, "claim": "First.", "status": "VERIFIED", "evidence": "artifact=a"},
        {"id": 2, "claim": "Second.", "status": "PARTIAL", "evidence": "missing CI"},
    ]

    payload = audit_claim_ledger_payload(payload_with_claims(claims))

    failures = {check["name"] for check in payload["issues"]}
    assert "no_nonverified_claims" in failures


def test_claim_ledger_checks_markdown_statuses():
    claims = [{"id": 1, "claim": "First.", "status": "VERIFIED", "evidence": "artifact=a"}]

    payload = audit_claim_ledger_payload(
        payload_with_claims(claims),
        claims_status_md="- Claim 1: **PARTIAL** - First. Evidence: artifact=a\n",
    )

    failures = {check["name"] for check in payload["issues"]}
    assert "claims_status_md_statuses" in failures


def test_claim_ledger_rejects_unstructured_evidence():
    claims = [{"id": 1, "claim": "First.", "status": "VERIFIED", "evidence": "artifact exists"}]

    payload = audit_claim_ledger_payload(payload_with_claims(claims))

    failures = {check["name"] for check in payload["issues"]}
    assert "claim_evidence_structured" in failures


def test_claim_ledger_checks_evidence_paths(tmp_path: Path):
    model_dir = tmp_path / "results" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "present.npz").write_bytes(b"model")
    claims = [
        {"id": 1, "claim": "First.", "status": "VERIFIED", "evidence": "model=results/models/present.npz"},
        {"id": 2, "claim": "Second.", "status": "VERIFIED", "evidence": "model=results/models/missing.npz"},
    ]

    payload = audit_claim_ledger_payload(payload_with_claims(claims), root=tmp_path)

    failures = {check["name"] for check in payload["issues"]}
    assert "claim_evidence_paths_exist" in failures
    assert payload["missing_evidence_path_references"][0]["claim_id"] == 2
