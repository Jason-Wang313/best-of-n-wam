from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IDEAL_CLAIM_ROWS = [
    {
        "id": "exact_math_core",
        "ideal_claim": "Exact best-of-N binary and utility laws are proved and empirically verified.",
        "paper_status": "promotable_result",
        "required_claim_ids": [1, 2, 3, 4],
        "required_files": [
            "results/exp1_exact_rollout_law_validation.json",
            "results/exp2_auc_vs_moment_hierarchy.json",
            "docs/theory.md",
        ],
        "limitation": "",
    },
    {
        "id": "learned_multi_env_core",
        "ideal_claim": "Learned WAM-lite reproduces the inference-value results across several CPU toy robotics environments.",
        "paper_status": "promotable_result",
        "required_claim_ids": [22, 23, 24, 25, 26, 27, 28, 29, 30, 45],
        "required_files": [
            "results/learned_wam_lite_training.json",
            "results/learned_wam_vs_analytic_wam.json",
            "results/multi_env_suite.json",
            "results/exp10_falsification_bad_scorer.json",
        ],
        "limitation": "Toy CPU environments, not real-robot evidence.",
    },
    {
        "id": "contact_benchmark_state_mode",
        "ideal_claim": "External manipulation benchmarks support the claims beyond toy BlockPush-style simulations.",
        "paper_status": "promotable_limited_scope",
        "required_claim_ids": [31, 32, 33, 34, 35, 36, 37, 48, 59, 69, 73, 83, 89, 98],
        "required_files": [
            "results/benchmark_maniskill_suite.json",
            "results/benchmark_gym_robotics_suite.json",
            "results/benchmark_metaworld_suite.json",
            "results/benchmark_robosuite_suite.json",
            "results/benchmark_libero_wam.json",
            "results/benchmark_robocasa_residual35_h1_n4_wam.json",
        ],
        "limitation": "State-mode and rollout-pool/short-horizon benchmark artifacts; not full benchmark-wide validation.",
    },
    {
        "id": "visual_observation_limited",
        "ideal_claim": "Visual observations are covered by learned WAM-lite artifacts.",
        "paper_status": "promotable_limited_scope",
        "required_claim_ids": [38, 39, 40, 55, 56, 57, 58, 64, 65, 66, 67],
        "required_files": [
            "results/visual_optional.json",
            "results/benchmark_visual_wam_lite.json",
            "results/benchmark_gym_robotics_visual_wam.json",
        ],
        "limitation": "Toy/Gymnasium/Fetch RGB artifacts only; ManiSkill RGB/RGB-D is blocker-documented, not claimed.",
    },
    {
        "id": "real_robot_hil",
        "ideal_claim": "The method is validated on a real robot or hardware-in-the-loop setup.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [126],
        "required_files": ["results/frontier_integrity.json"],
        "frontier_ids": ["real_robot_hil"],
        "limitation": "No real-robot or hardware-in-the-loop artifact exists in this repository.",
    },
    {
        "id": "modern_vla_libero",
        "ideal_claim": "The method is validated as modern VLA-style LIBERO policy performance.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [126],
        "required_files": ["results/frontier_integrity.json"],
        "frontier_ids": ["modern_vla_libero"],
        "limitation": "LIBERO artifacts are scripted/BC smokes and dense rollout-pool WAM evidence, not modern VLA performance.",
    },
    {
        "id": "full_robocasa_wide",
        "ideal_claim": "The method is validated across the full RoboCasa-wide task distribution.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [126],
        "required_files": ["results/frontier_integrity.json", "results/benchmark_robocasa_catalog_probe.json"],
        "frontier_ids": ["full_robocasa_wide"],
        "limitation": "RoboCasa has broad committed coverage, but not full RoboCasa-wide validation.",
    },
    {
        "id": "maniskill_visual_ee",
        "ideal_claim": "ManiSkill visual/RGB-D or end-effector-control validation is complete.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [126],
        "required_files": [
            "results/frontier_integrity.json",
            "results/benchmark_maniskill_visual_probe.json",
            "results/benchmark_maniskill_dependency_probe.json",
        ],
        "frontier_ids": ["maniskill_visual_ee"],
        "limitation": "ManiSkill evidence is state-mode; visual and EE-control blockers are artifact-documented.",
    },
    {
        "id": "universal_wam_training_recipe",
        "ideal_claim": "The project provides a universal WAM training or Robot Chinchilla-style train-inference recipe.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [123],
        "required_files": ["results/publication_scope.json"],
        "publication_scope_patterns": ["universal_wam"],
        "limitation": "Universal WAM training optimization is framed as future work, not a current result.",
    },
]


@dataclass(frozen=True)
class IdealBoundaryCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[IdealBoundaryCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(IdealBoundaryCheck(name=name, ok=bool(ok), detail=detail))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def verified_claim_ids(claims_payload: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for claim in claims_payload.get("claims") or []:
        if not isinstance(claim, dict) or claim.get("status") != "VERIFIED":
            continue
        try:
            ids.add(int(claim.get("id")))
        except (TypeError, ValueError):
            continue
    return ids


def frontier_statuses(frontier_payload: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for item in frontier_payload.get("frontier_items") or []:
        if isinstance(item, dict):
            statuses[str(item.get("frontier_id"))] = str(item.get("status"))
    return statuses


def publication_pattern_counts(publication_payload: dict[str, Any]) -> dict[str, int]:
    counts = publication_payload.get("mentions_by_pattern") or {}
    return {str(key): int(value or 0) for key, value in counts.items()} if isinstance(counts, dict) else {}


def audit_ideal_claim_boundary(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claims_payload = load_json(results_dir / "claims_status.json")
    frontier_payload = load_json(results_dir / "frontier_integrity.json")
    publication_payload = load_json(results_dir / "publication_scope.json")
    verified_ids = verified_claim_ids(claims_payload)
    frontier_by_id = frontier_statuses(frontier_payload)
    publication_counts = publication_pattern_counts(publication_payload)

    checks: list[IdealBoundaryCheck] = []
    add(checks, "claims_loaded", len(verified_ids) >= 120, f"verified_claims={len(verified_ids)}")
    add(checks, "frontier_integrity_loaded", frontier_payload.get("verified") is True, f"verified={frontier_payload.get('verified')}")
    add(checks, "publication_scope_loaded", publication_payload.get("verified") is True, f"verified={publication_payload.get('verified')}")

    rows: list[dict[str, Any]] = []
    for row in IDEAL_CLAIM_ROWS:
        required_claim_ids = [int(cid) for cid in row.get("required_claim_ids", [])]
        missing_claim_ids = [cid for cid in required_claim_ids if cid not in verified_ids]
        files = [file_record(root, relative) for relative in row.get("required_files", [])]
        missing_files = [record for record in files if not record["exists"] or record["bytes"] <= 0]
        required_frontier_ids = [str(fid) for fid in row.get("frontier_ids", [])]
        missing_frontier_guards = [
            fid for fid in required_frontier_ids if frontier_by_id.get(fid) != "guarded_not_promoted"
        ]
        required_patterns = [str(pattern) for pattern in row.get("publication_scope_patterns", [])]
        missing_publication_patterns = [pattern for pattern in required_patterns if publication_counts.get(pattern, 0) <= 0]
        promotable = str(row["paper_status"]).startswith("promotable")
        future_only = str(row["paper_status"]) == "future_only_not_promotable"
        ok = not missing_claim_ids and not missing_files and not missing_frontier_guards and not missing_publication_patterns

        add(checks, f"{row['id']}_required_claims_verified", not missing_claim_ids, f"missing={missing_claim_ids}")
        add(checks, f"{row['id']}_required_files_exist", not missing_files, f"missing={missing_files}")
        if required_frontier_ids:
            add(
                checks,
                f"{row['id']}_frontier_guards_present",
                not missing_frontier_guards,
                f"missing={missing_frontier_guards}",
            )
        if required_patterns:
            add(
                checks,
                f"{row['id']}_publication_scope_mentions_present",
                not missing_publication_patterns,
                f"missing={missing_publication_patterns}",
            )
        if future_only:
            add(checks, f"{row['id']}_not_promotable", row["paper_status"] == "future_only_not_promotable", str(row["paper_status"]))

        rows.append(
            {
                "id": row["id"],
                "ideal_claim": row["ideal_claim"],
                "paper_status": row["paper_status"],
                "boundary_evidence_present": ok,
                "endpoint_supported": promotable and ok,
                "promotable": promotable and ok,
                "future_only": future_only,
                "required_claim_ids": required_claim_ids,
                "missing_claim_ids": missing_claim_ids,
                "required_files": files,
                "missing_files": missing_files,
                "frontier_ids": required_frontier_ids,
                "missing_frontier_guards": missing_frontier_guards,
                "publication_scope_patterns": required_patterns,
                "missing_publication_scope_patterns": missing_publication_patterns,
                "limitation": row.get("limitation", ""),
            }
        )

    promotable_rows = [row for row in rows if row["promotable"]]
    future_rows = [row for row in rows if row["future_only"]]
    add(checks, "promotable_rows_present", len(promotable_rows) >= 4, f"promotable={len(promotable_rows)}")
    add(checks, "future_only_rows_present", len(future_rows) >= 4, f"future_only={len(future_rows)}")
    add(
        checks,
        "future_rows_not_promoted",
        all(not row["promotable"] for row in future_rows),
        f"promoted_future={[row['id'] for row in future_rows if row['promotable']]}",
    )

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "ideal_claim_boundary",
        "verified": len(issues) == 0,
        "n_ideal_claims": len(rows),
        "n_promotable_claims": len(promotable_rows),
        "n_future_only_claims": len(future_rows),
        "all_ideal_claims_promotable": all(row["promotable"] for row in rows),
        "rows": rows,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def ideal_claim_boundary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ideal Claim Boundary Report",
        "",
        f"- Verified boundary: {payload.get('verified')}",
        f"- Ideal claims audited: {payload.get('n_ideal_claims')}",
        f"- Promotable artifact-backed claims: {payload.get('n_promotable_claims')}",
        f"- Future-only non-promotable claims: {payload.get('n_future_only_claims')}",
        f"- All ideal claims promotable: {payload.get('all_ideal_claims_promotable')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
        "## Boundary Matrix",
        "",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"- `{row.get('id')}`: status=`{row.get('paper_status')}`, "
            f"endpoint_supported={row.get('endpoint_supported')}, "
            f"boundary_evidence_present={row.get('boundary_evidence_present')}, "
            f"promotable={row.get('promotable')}, "
            f"future_only={row.get('future_only')}"
        )
        limitation = row.get("limitation")
        if limitation:
            lines.append(f"  Limitation: {limitation}")
    issues = payload.get("issues") or []
    if issues:
        lines.extend(["", "## Issues", ""])
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.extend(
            [
                "",
                "The boundary is clean: artifact-backed rows may be promoted with their stated scope, while future-only ideal endpoints remain non-promotable.",
            ]
        )
    lines.append("")
    return "\n".join(lines)
