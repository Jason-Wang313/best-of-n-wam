from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.frontier_integrity import FRONTIER_SURFACES, audit_frontier_integrity


SAFE_TEXT = """# Safe

No real-robot evidence is claimed.
Hardware-in-the-loop evidence is future work.
Modern VLA-style LIBERO policy performance is not claimed.
Full RoboCasa-wide validation is not claimed.
ManiSkill RGB/RGB-D validation is blocker-documented, not verified.
ManiSkill visual or EE-control validation remains a blocker.
"""


def write_claims(results: Path, extra_claim: dict | None = None) -> None:
    results.mkdir(parents=True, exist_ok=True)
    claims = [
        {"id": idx, "claim": f"claim {idx}", "status": "VERIFIED", "evidence": f"value={idx}"}
        for idx in range(1, 126)
    ]
    if extra_claim is not None:
        claims.append(extra_claim)
    (results / "claims_status.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")


def write_surfaces(root: Path, overrides: dict[str, str] | None = None) -> None:
    overrides = overrides or {}
    for surface in FRONTIER_SURFACES:
        path = root / surface
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(overrides.get(surface, SAFE_TEXT), encoding="utf-8")


def write_blocker_artifacts(root: Path) -> None:
    for relative in [
        "results/benchmark_maniskill_visual_probe.json",
        "results/benchmark_maniskill_dependency_probe.json",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"attempted": True, "verified": True}), encoding="utf-8")
    for relative in [
        "reports/maniskill_visual_blocker_report.md",
        "reports/maniskill_dependency_blocker_report.md",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Blocker\n\nAttempted and blocker-documented.\n", encoding="utf-8")


def test_frontier_integrity_accepts_guarded_nonclaims(tmp_path: Path) -> None:
    write_claims(tmp_path / "results")
    write_surfaces(tmp_path)
    write_blocker_artifacts(tmp_path)

    payload = audit_frontier_integrity(tmp_path, tmp_path / "results")

    assert payload["verified"] is True
    assert payload["n_frontier_items"] == 4
    assert payload["n_promoted_frontier_claims"] == 0
    assert payload["n_unguarded_frontier_mentions"] == 0


def test_frontier_integrity_rejects_unguarded_publication_claim(tmp_path: Path) -> None:
    write_claims(tmp_path / "results")
    write_surfaces(tmp_path, {"README.md": "# Results\n\nThis repo has real-robot validation.\n" + SAFE_TEXT})
    write_blocker_artifacts(tmp_path)

    payload = audit_frontier_integrity(tmp_path, tmp_path / "results")

    assert payload["verified"] is False
    assert "real_robot_hil_no_unguarded_publication_mentions" in {issue["name"] for issue in payload["issues"]}


def test_frontier_integrity_rejects_promoted_claim_ledger_entry(tmp_path: Path) -> None:
    write_claims(
        tmp_path / "results",
        {
            "id": 126,
            "claim": "Real-robot validation is verified.",
            "status": "VERIFIED",
            "evidence": "successes=5",
        },
    )
    write_surfaces(tmp_path)
    write_blocker_artifacts(tmp_path)

    payload = audit_frontier_integrity(tmp_path, tmp_path / "results")

    assert payload["verified"] is False
    assert "real_robot_hil_not_promoted_as_verified_claim" in {issue["name"] for issue in payload["issues"]}


def test_frontier_integrity_requires_maniskill_blocker_artifacts(tmp_path: Path) -> None:
    write_claims(tmp_path / "results")
    write_surfaces(tmp_path)

    payload = audit_frontier_integrity(tmp_path, tmp_path / "results")

    assert payload["verified"] is False
    assert "maniskill_visual_ee_required_blocker_artifacts_exist" in {issue["name"] for issue in payload["issues"]}
