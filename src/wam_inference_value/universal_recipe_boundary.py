from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluation import results_dir, write_json


def build_universal_recipe_boundary(root: Path, output_results_dir: Path | None = None) -> dict[str, Any]:
    output_results_dir = (output_results_dir or results_dir()).resolve()
    # A finite-evidence no-free-lunch construction. The observed evidence is identical in
    # both worlds; the unobserved deployment context flips which recipe is optimal.
    observed_contexts = ["committed_toy_envs", "committed_benchmark_envs", "committed_visual_smokes"]
    choices = ["recipe_A", "recipe_B"]
    worlds = {
        "world_A": {
            "observed_evidence_signature": "same_as_world_B",
            "unobserved_context": "new_robot_distribution",
            "utility": {"recipe_A": 1.0, "recipe_B": 0.0},
            "optimal_recipe": "recipe_A",
        },
        "world_B": {
            "observed_evidence_signature": "same_as_world_A",
            "unobserved_context": "new_robot_distribution",
            "utility": {"recipe_A": 0.0, "recipe_B": 1.0},
            "optimal_recipe": "recipe_B",
        },
    }
    deterministic_recipe_output = "recipe_A"
    deterministic_regret = {
        "world_A": 0.0,
        "world_B": 1.0,
    }
    randomized_choice_probability_recipe_a = 0.5
    randomized_worst_case_regret_lower_bound = max(
        1.0 - randomized_choice_probability_recipe_a,
        randomized_choice_probability_recipe_a,
    )
    checks = [
        {
            "name": "worlds_share_observed_evidence",
            "ok": worlds["world_A"]["observed_evidence_signature"] == "same_as_world_B"
            and worlds["world_B"]["observed_evidence_signature"] == "same_as_world_A",
            "detail": f"observed_contexts={observed_contexts}",
        },
        {
            "name": "worlds_require_opposite_recipes",
            "ok": worlds["world_A"]["optimal_recipe"] != worlds["world_B"]["optimal_recipe"],
            "detail": f"optimal={worlds['world_A']['optimal_recipe']} vs {worlds['world_B']['optimal_recipe']}",
        },
        {
            "name": "deterministic_recipe_has_positive_worst_case_regret",
            "ok": max(deterministic_regret.values()) > 0.0,
            "detail": f"output={deterministic_recipe_output}, regret={deterministic_regret}",
        },
        {
            "name": "randomized_recipe_has_positive_worst_case_regret_lower_bound",
            "ok": randomized_worst_case_regret_lower_bound >= 0.5,
            "detail": f"minimax lower bound={randomized_worst_case_regret_lower_bound}",
        },
    ]
    payload = {
        "experiment": "universal_recipe_boundary",
        "verified": all(check["ok"] for check in checks),
        "result_type": "no_free_lunch_boundary",
        "claim": "No artifact-limited empirical optimizer can prove a universal WAM train/inference recipe over unrestricted future robot distributions without additional assumptions.",
        "observed_contexts": observed_contexts,
        "choices": choices,
        "worlds": worlds,
        "deterministic_recipe_output": deterministic_recipe_output,
        "deterministic_regret": deterministic_regret,
        "randomized_choice_probability_recipe_a": randomized_choice_probability_recipe_a,
        "randomized_worst_case_regret_lower_bound": randomized_worst_case_regret_lower_bound,
        "what_would_be_needed_for_a_positive_universal_claim": [
            "A specified restricted task/environment class.",
            "A distributional assumption linking observed artifacts to future deployments.",
            "A learnability/realizability assumption for the WAM family and scorer family.",
            "A proof or large heldout benchmark suite matching that restricted claim.",
        ],
        "n_checks": len(checks),
        "n_issues": sum(1 for check in checks if not check["ok"]),
        "checks": checks,
        "note": "Boundary proof only. This blocks an unrestricted universal-recipe claim; it does not replace the evidence-bound optimizer.",
    }
    write_json(output_results_dir / "universal_recipe_boundary.json", payload)
    return payload


def universal_recipe_boundary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Universal Recipe Boundary",
        "",
        f"- verified: `{payload.get('verified')}`",
        f"- result type: `{payload.get('result_type')}`",
        "",
        "## Claim",
        "",
        str(payload.get("claim")),
        "",
        "## Construction",
        "",
        "Two worlds match all committed evidence but assign opposite utilities to the same recipes in an unobserved deployment context.",
        "",
    ]
    worlds = payload.get("worlds") if isinstance(payload.get("worlds"), dict) else {}
    for name, world in worlds.items():
        lines.append(f"- `{name}`: optimal `{world.get('optimal_recipe')}`, utility `{world.get('utility')}`")
    lines.extend(
        [
            "",
            "A deterministic recipe therefore fails in one compatible world. A randomized recipe has positive worst-case regret; with a 50/50 mixture the lower bound is `0.5`.",
            "",
            "## Needed For A Positive Universal Claim",
            "",
        ]
    )
    for item in payload.get("what_would_be_needed_for_a_positive_universal_claim") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "This is a boundary result. It keeps the README and paper honest: the repo can claim exact inference laws and evidence-bound optimization, not a universal WAM training recipe.",
            "",
        ]
    )
    return "\n".join(lines)
