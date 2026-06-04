import json
from pathlib import Path

from wam_inference_value.claim_evidence_quality import audit_claim_evidence_payload


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_claim_evidence_quality_accepts_mapped_structured_sources(tmp_path: Path):
    write_json(tmp_path / "results" / "source.json", {"metric": 1.0})
    payload = {
        "claims": [
            {"id": 1, "claim": "Metric verified.", "status": "VERIFIED", "evidence": "metric=1.0"},
        ]
    }

    audit = audit_claim_evidence_payload(payload, root=tmp_path, source_map={1: ["results/source.json"]})

    assert audit["verified"]
    assert audit["n_source_links"] == 1
    assert audit["n_issues"] == 0


def test_claim_evidence_quality_flags_missing_source_map(tmp_path: Path):
    payload = {
        "claims": [
            {"id": 1, "claim": "Metric verified.", "status": "VERIFIED", "evidence": "metric=1.0"},
        ]
    }

    audit = audit_claim_evidence_payload(payload, root=tmp_path, source_map={})

    failures = {issue["name"] for issue in audit["issues"]}
    assert "all_current_claims_have_sources" in failures


def test_claim_evidence_quality_flags_placeholder_evidence(tmp_path: Path):
    write_json(tmp_path / "results" / "source.json", {"metric": 1.0})
    payload = {
        "claims": [
            {"id": 1, "claim": "Metric verified.", "status": "VERIFIED", "evidence": "metric=missing"},
        ]
    }

    audit = audit_claim_evidence_payload(payload, root=tmp_path, source_map={1: ["results/source.json"]})

    failures = {issue["name"] for issue in audit["issues"]}
    assert "evidence_has_no_placeholder_literals" in failures
