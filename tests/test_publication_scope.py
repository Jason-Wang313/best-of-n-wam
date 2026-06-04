from __future__ import annotations

import json
from pathlib import Path

from wam_inference_value.publication_scope import PUBLICATION_SURFACES, audit_publication_scope


def write_claims(results: Path, n_verified: int = 123) -> None:
    results.mkdir(parents=True, exist_ok=True)
    claims = [
        {"id": idx, "claim": f"claim {idx}", "status": "VERIFIED", "evidence": f"value={idx}"}
        for idx in range(1, n_verified + 1)
    ]
    (results / "claims_status.json").write_text(json.dumps({"claims": claims}), encoding="utf-8")


def write_all_surfaces(root: Path, text_by_surface: dict[str, str]) -> None:
    default = (
        "# Safe\n\n"
        "No real-robot evidence is claimed.\n"
        "Modern VLA policy performance is future work.\n"
        "Full RoboCasa-wide validation is not claimed.\n"
        "ManiSkill RGB/RGB-D validation is blocker-documented, not verified.\n"
        "A universal WAM training recipe is discussion-only.\n"
    )
    for surface in PUBLICATION_SURFACES:
        path = root / surface
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_by_surface.get(surface, default), encoding="utf-8")


def test_publication_scope_accepts_guarded_risk_mentions(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results)
    write_all_surfaces(tmp_path, {})

    payload = audit_publication_scope(tmp_path, results)

    assert payload["verified"] is True
    assert payload["n_unguarded_mentions"] == 0
    assert payload["n_risk_mentions"] >= 20


def test_publication_scope_rejects_unguarded_real_robot_claim(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results)
    write_all_surfaces(
        tmp_path,
        {
            "README.md": (
                "# Results\n\n"
                "This project has real-robot validation across all settings.\n"
                "Modern VLA policy performance is future work.\n"
                "Full RoboCasa-wide validation is not claimed.\n"
                "ManiSkill RGB/RGB-D validation is blocker-documented, not verified.\n"
                "A universal WAM training recipe is discussion-only.\n"
            )
        },
    )

    payload = audit_publication_scope(tmp_path, results)

    assert payload["verified"] is False
    assert payload["n_unguarded_mentions"] >= 1
    assert "all_risk_mentions_guarded" in {issue["name"] for issue in payload["issues"]}


def test_publication_scope_accepts_future_heading_context(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_claims(results)
    write_all_surfaces(
        tmp_path,
        {
            "paper_outline.md": (
                "# Paper\n\n"
                "No real-robot evidence is claimed.\n"
                "Modern VLA policy performance is future work.\n"
                "Full RoboCasa-wide validation is not claimed.\n"
                "ManiSkill RGB/RGB-D validation is blocker-documented, not verified.\n"
                "## Future Work\n\n"
                "Robot Chinchilla: a universal WAM train-inference optimizer.\n"
            )
        },
    )

    payload = audit_publication_scope(tmp_path, results)

    assert payload["verified"] is True
    assert payload["n_unguarded_mentions"] == 0
