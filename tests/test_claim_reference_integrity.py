import json
from pathlib import Path

from wam_inference_value.claim_reference_integrity import REFERENCE_SURFACES, audit_claim_references


def write_claims(results: Path, statuses: dict[int, str]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    claims = [{"id": cid, "claim": f"claim {cid}", "status": status, "evidence": f"value={cid}"} for cid, status in statuses.items()]
    (results / "claims_status.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")


def write_surfaces(root: Path, text: str) -> None:
    for surface in REFERENCE_SURFACES:
        path = root / surface
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_claim_reference_integrity_accepts_verified_references(tmp_path: Path) -> None:
    write_claims(tmp_path / "results", {idx: "VERIFIED" for idx in range(1, 130)})
    write_surfaces(tmp_path, "VERIFIED CLAIM 1. VERIFIED CLAIM 2. VERIFIED CLAIM 3. VERIFIED CLAIM 4. VERIFIED CLAIM 5. VERIFIED CLAIM 6. VERIFIED CLAIM 7. VERIFIED CLAIM 8. VERIFIED CLAIM 9. VERIFIED CLAIM 10.")

    payload = audit_claim_references(tmp_path, tmp_path / "results")

    assert payload["verified"] is True
    assert payload["n_references"] >= 10
    assert payload["n_issues"] == 0


def test_claim_reference_integrity_rejects_missing_claim_id(tmp_path: Path) -> None:
    write_claims(tmp_path / "results", {idx: "VERIFIED" for idx in range(1, 130)})
    write_surfaces(tmp_path, "VERIFIED CLAIM 999. VERIFIED CLAIM 1. VERIFIED CLAIM 2. VERIFIED CLAIM 3. VERIFIED CLAIM 4. VERIFIED CLAIM 5. VERIFIED CLAIM 6. VERIFIED CLAIM 7. VERIFIED CLAIM 8. VERIFIED CLAIM 9.")

    payload = audit_claim_references(tmp_path, tmp_path / "results")

    issue_names = {issue["name"] for issue in payload["issues"]}
    assert "all_referenced_claims_exist" in issue_names


def test_claim_reference_integrity_rejects_nonverified_claim_id(tmp_path: Path) -> None:
    statuses = {idx: "VERIFIED" for idx in range(1, 130)}
    statuses[3] = "PARTIAL"
    write_claims(tmp_path / "results", statuses)
    write_surfaces(tmp_path, "VERIFIED CLAIM 1. VERIFIED CLAIM 2. VERIFIED CLAIM 3. VERIFIED CLAIM 4. VERIFIED CLAIM 5. VERIFIED CLAIM 6. VERIFIED CLAIM 7. VERIFIED CLAIM 8. VERIFIED CLAIM 9. VERIFIED CLAIM 10.")

    payload = audit_claim_references(tmp_path, tmp_path / "results")

    issue_names = {issue["name"] for issue in payload["issues"]}
    assert "all_referenced_claims_verified" in issue_names
