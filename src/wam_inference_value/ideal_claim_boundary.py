from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HUMAN_BOUNDARY_SURFACES = [
    "README.md",
    "paper_outline.md",
    "reports/final_decision_report.md",
    "reports/reviewer_risk_assessment.md",
]


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
        "surface_markers": ["real-robot", "hardware-in-the-loop"],
        "limitation": "No real-robot or hardware-in-the-loop artifact exists in this repository.",
        "promotion_requirements": [
            "Committed real-robot or hardware-in-the-loop rollout/control artifacts with task definitions, seeds, and success or utility metrics.",
            "A claims_status entry whose evidence points to those artifacts rather than simulator-only benchmark results.",
        ],
        "missing_evidence_classes": [
            "Physical robot or hardware-in-the-loop execution logs.",
            "Task-level real-world success/utility metrics with seeds or trial IDs.",
            "A verified claim ledger entry sourced from real-world artifacts.",
        ],
        "gap_evidence_files": [
            "results/ideal_frontier_readiness.json",
            "reports/ideal_frontier_readiness_report.md",
            "reports/final_decision_report.md",
            "reports/reviewer_risk_assessment.md",
        ],
    },
    {
        "id": "modern_vla_libero",
        "ideal_claim": "The method is validated as modern VLA-style LIBERO policy performance.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [126],
        "required_files": ["results/frontier_integrity.json"],
        "frontier_ids": ["modern_vla_libero"],
        "surface_markers": ["modern vla", "vla-style"],
        "limitation": "LIBERO artifacts are scripted/BC smokes and dense rollout-pool WAM evidence, not modern VLA performance.",
        "promotion_requirements": [
            "A modern VLA-style policy or policy-compatible controller evaluated on LIBERO sparse-success tasks.",
            "Heldout success metrics with confidence intervals that do not rely on scripted phase labels, target-point commands, or simulator object-state shortcuts unless explicitly scoped.",
        ],
        "missing_evidence_classes": [
            "Modern VLA-style policy artifact evaluated as a policy, not a rollout-pool scorer or scripted/BC smoke.",
            "Sparse-success heldout LIBERO metrics with confidence intervals under the promoted observation/action interface.",
            "Evidence that evaluation-time inputs do not use shortcuts beyond the stated policy scope.",
        ],
        "gap_evidence_files": [
            "results/ideal_frontier_readiness.json",
            "reports/ideal_frontier_readiness_report.md",
            "results/benchmark_libero_wam.json",
            "results/benchmark_libero_visual_language_bc_policy.json",
            "reports/final_decision_report.md",
        ],
    },
    {
        "id": "full_robocasa_wide",
        "ideal_claim": "The method is validated across the full RoboCasa-wide task distribution.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [126],
        "required_files": ["results/frontier_integrity.json", "results/benchmark_robocasa_catalog_probe.json"],
        "frontier_ids": ["full_robocasa_wide"],
        "surface_markers": ["full robocasa-wide"],
        "limitation": "RoboCasa has broad committed coverage, but not full RoboCasa-wide validation.",
        "promotion_requirements": [
            "Rollout-pool or policy artifacts covering the full declared RoboCasa task distribution, not only sampled or stratified subsets.",
            "Registry coverage evidence showing the promoted task set matches the full benchmark scope claimed in README and paper text.",
        ],
        "missing_evidence_classes": [
            "Full declared RoboCasa task-distribution rollout-pool or policy artifacts.",
            "Coverage proof that promoted task IDs match the full local benchmark registry scope.",
            "Claim evidence that distinguishes full-suite validation from sampled or stratified-family validation.",
        ],
        "gap_evidence_files": [
            "results/ideal_frontier_readiness.json",
            "reports/ideal_frontier_readiness_report.md",
            "results/benchmark_robocasa_catalog_probe.json",
            "reports/benchmark_blocker_report.md",
            "reports/final_decision_report.md",
        ],
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
        "surface_markers": ["maniskill rgb/rgb-d", "end-effector", "ee-control"],
        "limitation": "ManiSkill evidence is state-mode; visual and EE-control blockers are artifact-documented.",
        "promotion_requirements": [
            "Successful ManiSkill RGB/RGB-D rollout or WAM artifacts generated from rendered observations without the current renderer blocker.",
            "Successful ManiSkill end-effector-control artifacts or a scoped statement that no EE-control claim is being made.",
        ],
        "missing_evidence_classes": [
            "Successful ManiSkill RGB/RGB-D rendered-observation rollout or WAM artifact.",
            "Successful ManiSkill end-effector-control artifact or an explicitly narrower promoted control scope.",
            "Closed-loop or rollout-pool metrics generated after the renderer/control blockers are cleared.",
        ],
        "gap_evidence_files": [
            "results/ideal_frontier_readiness.json",
            "reports/ideal_frontier_readiness_report.md",
            "results/benchmark_maniskill_visual_probe.json",
            "results/benchmark_maniskill_dependency_probe.json",
            "reports/maniskill_visual_blocker_report.md",
            "reports/maniskill_dependency_blocker_report.md",
        ],
    },
    {
        "id": "universal_wam_training_recipe",
        "ideal_claim": "The project provides a universal WAM training or Robot Chinchilla-style train-inference recipe.",
        "paper_status": "future_only_not_promotable",
        "required_claim_ids": [123],
        "required_files": ["results/publication_scope.json"],
        "publication_scope_patterns": ["universal_wam"],
        "surface_markers": ["universal wam", "robot chinchilla"],
        "limitation": "Universal WAM training optimization is framed as future work, not a current result.",
        "promotion_requirements": [
            "A tested train/inference optimizer that chooses data scale, model capacity, rollout horizon, scorer quality, safety constraints, and sampling budget.",
            "Evidence that the optimizer generalizes beyond the current artifact-specific WAM-lite and benchmark recipes.",
        ],
        "missing_evidence_classes": [
            "Executable universal train/inference optimizer artifact.",
            "Cross-environment evidence that the optimizer chooses data, model, scorer, horizon, safety, and sampling budgets.",
            "Claim evidence separating this future recipe from the current exact test-time inference law.",
        ],
        "gap_evidence_files": [
            "results/ideal_frontier_readiness.json",
            "reports/ideal_frontier_readiness_report.md",
            "results/universal_wam_train_inference_optimizer.json",
            "reports/universal_wam_train_inference_optimizer_report.md",
            "results/publication_scope.json",
            "reports/final_decision_report.md",
            "reports/reviewer_risk_assessment.md",
        ],
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


def missing_file_records(root: Path, relatives: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = [file_record(root, relative) for relative in relatives]
    missing = [record for record in records if not record["exists"] or record["bytes"] <= 0]
    return records, missing


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


def load_human_surface_text(root: Path) -> tuple[str, list[str]]:
    texts: list[str] = []
    missing: list[str] = []
    for relative in HUMAN_BOUNDARY_SURFACES:
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(texts).lower(), missing


def marker_hits(surface_text: str, markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker.lower() in surface_text]


def audit_ideal_claim_boundary(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    claims_payload = load_json(results_dir / "claims_status.json")
    frontier_payload = load_json(results_dir / "frontier_integrity.json")
    publication_payload = load_json(results_dir / "publication_scope.json")
    verified_ids = verified_claim_ids(claims_payload)
    frontier_by_id = frontier_statuses(frontier_payload)
    publication_counts = publication_pattern_counts(publication_payload)
    human_surface_text, missing_human_surfaces = load_human_surface_text(root)

    checks: list[IdealBoundaryCheck] = []
    add(checks, "claims_loaded", len(verified_ids) >= 120, f"verified_claims={len(verified_ids)}")
    add(checks, "frontier_integrity_loaded", frontier_payload.get("verified") is True, f"verified={frontier_payload.get('verified')}")
    add(checks, "publication_scope_loaded", publication_payload.get("verified") is True, f"verified={publication_payload.get('verified')}")
    add(checks, "human_boundary_surfaces_exist", not missing_human_surfaces, f"missing={missing_human_surfaces}")

    rows: list[dict[str, Any]] = []
    for row in IDEAL_CLAIM_ROWS:
        required_claim_ids = [int(cid) for cid in row.get("required_claim_ids", [])]
        missing_claim_ids = [cid for cid in required_claim_ids if cid not in verified_ids]
        files = [file_record(root, relative) for relative in row.get("required_files", [])]
        missing_files = [record for record in files if not record["exists"] or record["bytes"] <= 0]
        gap_evidence_files, missing_gap_evidence_files = missing_file_records(
            root,
            [str(relative) for relative in row.get("gap_evidence_files", [])],
        )
        required_frontier_ids = [str(fid) for fid in row.get("frontier_ids", [])]
        missing_frontier_guards = [
            fid for fid in required_frontier_ids if frontier_by_id.get(fid) != "guarded_not_promoted"
        ]
        required_patterns = [str(pattern) for pattern in row.get("publication_scope_patterns", [])]
        missing_publication_patterns = [pattern for pattern in required_patterns if publication_counts.get(pattern, 0) <= 0]
        surface_markers = [str(marker) for marker in row.get("surface_markers", [])]
        surface_marker_hits = marker_hits(human_surface_text, surface_markers)
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
            add(checks, f"{row['id']}_limitation_text_present", bool(str(row.get("limitation") or "").strip()), str(row.get("limitation") or ""))
            promotion_requirements = [str(item) for item in row.get("promotion_requirements", []) if str(item).strip()]
            missing_evidence_classes = [str(item) for item in row.get("missing_evidence_classes", []) if str(item).strip()]
            add(
                checks,
                f"{row['id']}_promotion_requirements_present",
                len(promotion_requirements) >= 2,
                f"requirements={promotion_requirements}",
            )
            add(
                checks,
                f"{row['id']}_missing_evidence_classes_present",
                len(missing_evidence_classes) >= 2,
                f"missing_evidence={missing_evidence_classes}",
            )
            add(
                checks,
                f"{row['id']}_gap_evidence_files_exist",
                bool(gap_evidence_files) and not missing_gap_evidence_files,
                f"missing={missing_gap_evidence_files}",
            )
            add(
                checks,
                f"{row['id']}_human_surface_marker_present",
                bool(surface_marker_hits),
                f"markers={surface_markers}, hits={surface_marker_hits}",
            )

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
                "surface_markers": surface_markers,
                "surface_marker_hits": surface_marker_hits,
                "limitation": row.get("limitation", ""),
                "promotion_requirements": [str(item) for item in row.get("promotion_requirements", []) if str(item).strip()],
                "missing_evidence_classes": [str(item) for item in row.get("missing_evidence_classes", []) if str(item).strip()],
                "gap_evidence_files": gap_evidence_files,
                "missing_gap_evidence_files": missing_gap_evidence_files,
            }
        )

    promotable_rows = [row for row in rows if row["promotable"]]
    future_rows = [row for row in rows if row["future_only"]]
    endpoint_supported_rows = [row for row in rows if row["endpoint_supported"]]
    unsupported_future_rows = [row for row in future_rows if not row["endpoint_supported"]]
    future_with_promotion_requirements = [
        row for row in future_rows if len(row["promotion_requirements"]) >= 2
    ]
    future_with_missing_evidence_classes = [
        row for row in future_rows if len(row["missing_evidence_classes"]) >= 2
    ]
    future_with_gap_evidence_files = [
        row for row in future_rows if row["gap_evidence_files"] and not row["missing_gap_evidence_files"]
    ]
    all_future_only_have_promotion_requirements = bool(future_rows) and len(future_with_promotion_requirements) == len(future_rows)
    all_future_only_have_missing_evidence_classes = bool(future_rows) and len(future_with_missing_evidence_classes) == len(future_rows)
    all_future_only_have_gap_evidence_files = bool(future_rows) and len(future_with_gap_evidence_files) == len(future_rows)
    goal_completion_status = (
        "complete_all_ideal_endpoints_supported"
        if len(endpoint_supported_rows) == len(rows)
        else "incomplete_future_only_gaps_remain"
    )
    add(checks, "promotable_rows_present", len(promotable_rows) >= 4, f"promotable={len(promotable_rows)}")
    add(checks, "future_only_rows_present", len(future_rows) >= 4, f"future_only={len(future_rows)}")
    add(
        checks,
        "future_rows_not_promoted",
        all(not row["promotable"] for row in future_rows),
        f"promoted_future={[row['id'] for row in future_rows if row['promotable']]}",
    )
    add(
        checks,
        "future_rows_have_promotion_requirements",
        all_future_only_have_promotion_requirements,
        f"ready={len(future_with_promotion_requirements)}/{len(future_rows)}",
    )
    add(
        checks,
        "future_rows_have_missing_evidence_classes",
        all_future_only_have_missing_evidence_classes,
        f"ready={len(future_with_missing_evidence_classes)}/{len(future_rows)}",
    )
    add(
        checks,
        "future_rows_have_gap_evidence_files",
        all_future_only_have_gap_evidence_files,
        f"ready={len(future_with_gap_evidence_files)}/{len(future_rows)}",
    )
    add(
        checks,
        "goal_completion_not_claimed",
        goal_completion_status == "incomplete_future_only_gaps_remain",
        f"status={goal_completion_status}",
    )

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "ideal_claim_boundary",
        "verified": len(issues) == 0,
        "n_ideal_claims": len(rows),
        "n_promotable_claims": len(promotable_rows),
        "n_future_only_claims": len(future_rows),
        "n_endpoint_supported_claims": len(endpoint_supported_rows),
        "n_unsupported_future_only_claims": len(unsupported_future_rows),
        "n_future_only_with_promotion_requirements": len(future_with_promotion_requirements),
        "n_future_only_with_missing_evidence_classes": len(future_with_missing_evidence_classes),
        "n_future_only_with_gap_evidence_files": len(future_with_gap_evidence_files),
        "n_human_boundary_surfaces": len(HUMAN_BOUNDARY_SURFACES),
        "n_missing_human_boundary_surfaces": len(missing_human_surfaces),
        "all_ideal_claims_promotable": all(row["promotable"] for row in rows),
        "all_future_only_have_promotion_requirements": all_future_only_have_promotion_requirements,
        "all_future_only_have_missing_evidence_classes": all_future_only_have_missing_evidence_classes,
        "all_future_only_have_gap_evidence_files": all_future_only_have_gap_evidence_files,
        "goal_completion_status": goal_completion_status,
        "completion_blockers": [row["id"] for row in unsupported_future_rows],
        "missing_human_boundary_surfaces": missing_human_surfaces,
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
        f"- Endpoint-supported claims: {payload.get('n_endpoint_supported_claims')}",
        f"- Unsupported future-only endpoints: {payload.get('n_unsupported_future_only_claims')}",
        f"- All ideal claims promotable: {payload.get('all_ideal_claims_promotable')}",
        f"- Goal completion status: {payload.get('goal_completion_status')}",
        f"- Future-only rows with promotion requirements: {payload.get('n_future_only_with_promotion_requirements')}",
        f"- Future-only rows with missing-evidence classes: {payload.get('n_future_only_with_missing_evidence_classes')}",
        f"- Future-only rows with gap evidence files: {payload.get('n_future_only_with_gap_evidence_files')}",
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
        promotion_requirements = row.get("promotion_requirements") or []
        if promotion_requirements:
            lines.append("  Promotion requirements:")
            for requirement in promotion_requirements:
                lines.append(f"  - Future-only promotion requirement, not current evidence: {requirement}")
        missing_evidence_classes = row.get("missing_evidence_classes") or []
        if missing_evidence_classes:
            lines.append("  Missing evidence classes:")
            for evidence_class in missing_evidence_classes:
                lines.append(f"  - Missing future-only evidence class, not current evidence: {evidence_class}")
        gap_evidence_files = row.get("gap_evidence_files") or []
        if gap_evidence_files:
            formatted = ", ".join(f"`{record.get('path')}`" for record in gap_evidence_files)
            lines.append(f"  Gap evidence files: {formatted}")
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
