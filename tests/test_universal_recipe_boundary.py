from __future__ import annotations

from pathlib import Path

from wam_inference_value.universal_recipe_boundary import (
    build_universal_recipe_boundary,
    universal_recipe_boundary_markdown,
)


def test_universal_recipe_boundary_builds_no_free_lunch_artifact(tmp_path: Path) -> None:
    results = tmp_path / "results"

    payload = build_universal_recipe_boundary(tmp_path, results)

    assert payload["verified"] is True
    assert payload["worlds"]["world_A"]["observed_evidence_signature"] == "same_as_world_B"
    assert payload["worlds"]["world_A"]["optimal_recipe"] != payload["worlds"]["world_B"]["optimal_recipe"]
    assert max(payload["deterministic_regret"].values()) > 0.0
    assert payload["randomized_worst_case_regret_lower_bound"] >= 0.5
    assert (results / "universal_recipe_boundary.json").exists()


def test_universal_recipe_boundary_markdown_guards_positive_claim(tmp_path: Path) -> None:
    payload = build_universal_recipe_boundary(tmp_path, tmp_path / "results")

    text = universal_recipe_boundary_markdown(payload)

    assert "not a universal WAM training recipe" in text
    assert "A specified restricted task/environment class." in text
