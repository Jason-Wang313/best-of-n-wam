from __future__ import annotations

import json
from pathlib import Path

import wam_inference_value.ideal_claim_boundary as boundary
from wam_inference_value.ideal_claim_boundary import HUMAN_BOUNDARY_SURFACES, IDEAL_CLAIM_ROWS, audit_ideal_claim_boundary


def all_required_claim_ids() -> set[int]:
    ids: set[int] = set()
    for row in IDEAL_CLAIM_ROWS:
        ids.update(int(cid) for cid in row.get("required_claim_ids", []))
    ids.update(range(1, 121))
    return ids


def write_claims(results: Path, missing: set[int] | None = None) -> None:
    missing = missing or set()
    results.mkdir(parents=True, exist_ok=True)
    claims = [
        {"id": cid, "claim": f"claim {cid}", "status": "VERIFIED", "evidence": f"value={cid}"}
        for cid in sorted(all_required_claim_ids() - missing)
    ]
    (results / "claims_status.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")


def write_required_files(root: Path, *, omit: str | None = None) -> None:
    for row in IDEAL_CLAIM_ROWS:
        for relative in row.get("required_files", []):
            if relative == omit:
                continue
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(json.dumps({"verified": True}), encoding="utf-8")
            else:
                path.write_text("verified\n", encoding="utf-8")


def write_frontier(results: Path, *, verified: bool = True) -> None:
    items = [
        {
            "frontier_id": frontier_id,
            "status": "guarded_not_promoted" if verified else "promoted",
        }
        for frontier_id in ["real_robot_hil", "modern_vla_libero", "full_robocasa_wide", "maniskill_visual_ee"]
    ]
    (results / "frontier_integrity.json").write_text(
        json.dumps({"verified": verified, "frontier_items": items}),
        encoding="utf-8",
    )


def write_publication(results: Path, *, universal_mentions: int = 1) -> None:
    (results / "publication_scope.json").write_text(
        json.dumps({"verified": True, "mentions_by_pattern": {"universal_wam": universal_mentions}}),
        encoding="utf-8",
    )


def write_human_surfaces(root: Path, *, omit_markers: bool = False) -> None:
    text = (
        "# Boundary\n\n"
        "No real-robot or hardware-in-the-loop artifact exists.\n"
        "Modern VLA-style LIBERO performance is future work.\n"
        "Full RoboCasa-wide validation is not claimed.\n"
        "ManiSkill RGB/RGB-D and end-effector validation remain blocker-documented.\n"
        "Universal WAM and Robot Chinchilla training recipes are future work.\n"
    )
    if omit_markers:
        text = "# Boundary\n\nFuture-only limitations are discussed without naming the endpoint.\n"
    for relative in HUMAN_BOUNDARY_SURFACES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_ideal_claim_boundary_accepts_clean_boundary(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results)
    write_required_files(tmp_path)
    write_frontier(results)
    write_publication(results)
    write_human_surfaces(tmp_path)

    payload = audit_ideal_claim_boundary(tmp_path, results)

    assert payload["verified"] is True
    assert payload["all_ideal_claims_promotable"] is False
    assert payload["n_promotable_claims"] >= 4
    assert payload["n_future_only_claims"] >= 4


def test_ideal_claim_boundary_rejects_missing_required_claim(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results, missing={1})
    write_required_files(tmp_path)
    write_frontier(results)
    write_publication(results)
    write_human_surfaces(tmp_path)

    payload = audit_ideal_claim_boundary(tmp_path, results)

    assert payload["verified"] is False
    assert "exact_math_core_required_claims_verified" in {issue["name"] for issue in payload["issues"]}


def test_ideal_claim_boundary_rejects_missing_required_file(tmp_path: Path) -> None:
    results = tmp_path / "results"
    missing_file = "results/exp1_exact_rollout_law_validation.json"
    write_claims(results)
    write_required_files(tmp_path, omit=missing_file)
    write_frontier(results)
    write_publication(results)
    write_human_surfaces(tmp_path)

    payload = audit_ideal_claim_boundary(tmp_path, results)

    assert payload["verified"] is False
    assert "exact_math_core_required_files_exist" in {issue["name"] for issue in payload["issues"]}


def test_ideal_claim_boundary_rejects_missing_future_guard(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results)
    write_required_files(tmp_path)
    write_frontier(results, verified=False)
    write_publication(results)
    write_human_surfaces(tmp_path)

    payload = audit_ideal_claim_boundary(tmp_path, results)

    assert payload["verified"] is False
    assert "real_robot_hil_frontier_guards_present" in {issue["name"] for issue in payload["issues"]}


def test_ideal_claim_boundary_rejects_missing_human_surface_marker(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results)
    write_required_files(tmp_path)
    write_frontier(results)
    write_publication(results)
    write_human_surfaces(tmp_path, omit_markers=True)

    payload = audit_ideal_claim_boundary(tmp_path, results)

    assert payload["verified"] is False
    assert "real_robot_hil_human_surface_marker_present" in {issue["name"] for issue in payload["issues"]}


def test_ideal_claim_boundary_rejects_missing_promotion_requirements(tmp_path: Path, monkeypatch) -> None:
    rows = []
    for row in IDEAL_CLAIM_ROWS:
        copied = dict(row)
        if copied["id"] == "real_robot_hil":
            copied["promotion_requirements"] = []
        rows.append(copied)
    monkeypatch.setattr(boundary, "IDEAL_CLAIM_ROWS", rows)

    results = tmp_path / "results"
    write_claims(results)
    write_required_files(tmp_path)
    write_frontier(results)
    write_publication(results)
    write_human_surfaces(tmp_path)

    payload = audit_ideal_claim_boundary(tmp_path, results)

    assert payload["verified"] is False
    assert "real_robot_hil_promotion_requirements_present" in {issue["name"] for issue in payload["issues"]}
