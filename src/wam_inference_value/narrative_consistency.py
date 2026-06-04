from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NarrativeCheck:
    surface: str
    name: str
    ok: bool
    expected: str
    detail: str


def load_json(results_dir: Path, name: str) -> dict[str, Any]:
    path = results_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def f3(value: Any) -> str:
    return f"{float(value):.3f}"


def f4(value: Any) -> str:
    return f"{float(value):.4f}"


def f5(value: Any) -> str:
    return f"{float(value):.5f}"


def comma_int(value: Any) -> str:
    return f"{int(value):,}"


def markdown_series(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def ci_lo(payload: dict[str, Any], key: str) -> Any:
    return ((payload.get("confidence_intervals") or {}).get(key) or {}).get("lo")


def best_ci_lo(payload: dict[str, Any], n_value: int = 8) -> Any:
    ci = payload.get("confidence_intervals") or {}
    return (ci.get(f"best_learned_minus_random_N{n_value}") or {}).get("lo")


def oracle_best_ci_lo(payload: dict[str, Any], n_value: int = 8) -> Any:
    ci = payload.get("confidence_intervals") or {}
    return (ci.get(f"oracle_minus_best_learned_N{n_value}") or {}).get("lo")


def metric(payload: dict[str, Any], key: str) -> Any:
    return (payload.get("model_metrics") or {}).get(key)


def add_contains(checks: list[NarrativeCheck], surface: str, text: str, name: str, expected: str) -> None:
    checks.append(
        NarrativeCheck(
            surface=surface,
            name=name,
            ok=expected in text,
            expected=expected,
            detail="found" if expected in text else "missing expected snippet",
        )
    )


def add_no_template_markers(checks: list[NarrativeCheck], surface: str, text: str) -> None:
    markers = [
        "{fmt(",
        "{bullet_lines(",
        "{claims_payload",
        "{artifact_integrity.",
        "{artifact_manifest.",
        "{result_consistency.",
        "{raw_result_recompute.",
        "{narrative_consistency.",
        "{script_contracts.",
        "{claim_semantics.",
        "{claim_evidence_quality.",
        "{claim_ledger_integrity.",
        "{figure_quality.",
        "{gym_",
        "{metaworld.",
        "{robosuite.",
        "{maniskill",
        "{robocasa",
        "{libero",
        "{bench_visual_wam",
        "{residual35_",
        "`missing`",
    ]
    found = [marker for marker in markers if marker in text]
    checks.append(
        NarrativeCheck(
            surface=surface,
            name="no_unresolved_template_markers",
            ok=not found,
            expected="no unresolved template markers",
            detail=f"found={found}",
        )
    )


def audit_readme(root: Path, results_dir: Path, checks: list[NarrativeCheck]) -> None:
    text = read_text(root / "README.md")
    learned = load_json(results_dir, "learned_wam_lite_training.json")
    val = (learned.get("metrics") or {}).get("validation") or {}
    multitask = load_json(results_dir, "benchmark_robocasa_multitask_wam.json")
    robocasa_family = [
        load_json(results_dir, "benchmark_robocasa_broad_wam.json"),
        load_json(results_dir, "benchmark_robocasa_family12_wam.json"),
        load_json(results_dir, "benchmark_robocasa_family24_wam.json"),
        load_json(results_dir, "benchmark_robocasa_extra4_wam.json"),
        load_json(results_dir, "benchmark_robocasa_family28_wam.json"),
        load_json(results_dir, "benchmark_robocasa_family32_wam.json"),
        load_json(results_dir, "benchmark_robocasa_stratified55_wam.json"),
        load_json(results_dir, "benchmark_robocasa_stratified97_wam.json"),
    ]
    strat97 = robocasa_family[-1]
    residual35 = load_json(results_dir, "benchmark_robocasa_residual35_h1_n4_wam.json")
    residual_sweep = load_json(results_dir, "benchmark_robocasa_residual_frontier_sweep.json")
    catalog = load_json(results_dir, "benchmark_robocasa_catalog_probe.json")
    libero = load_json(results_dir, "benchmark_libero_wam.json")
    libero_scripted = load_json(results_dir, "benchmark_libero_scripted_policy.json")
    libero_action = load_json(results_dir, "benchmark_libero_learned_action_head.json")
    libero_bc = load_json(results_dir, "benchmark_libero_autonomous_bc_policy.json")
    libero_visual = load_json(results_dir, "benchmark_libero_visual_language_bc_policy.json")

    add_contains(
        checks,
        "README",
        text,
        "learned_blockpush_validation_errors",
        f"validation final-position L2 MAE `{f4(val.get('final_position_l2_mae'))}`, validation utility MAE `{f4(val.get('utility_mae'))}`",
    )
    add_contains(
        checks,
        "README",
        text,
        "robocasa_three_task_samples_and_ci",
        (
            f"trains on {multitask.get('train_samples')} rollouts, validates on {multitask.get('validation_samples')} rollouts "
            f"with utility correlation `{f3(metric(multitask, 'utility_corr'))}`, evaluates on {multitask.get('eval_samples')} heldout rollouts, "
            f"has exact-law MAE `{f5(multitask.get('exact_law_utility_mae'))}`, and beats random at N8 with CI lower `{f3(best_ci_lo(multitask))}`"
        ),
    )
    corr_list = markdown_series([f"`{f3(metric(payload, 'utility_corr'))}`" for payload in robocasa_family])
    ci_list = markdown_series([f"`{f3(best_ci_lo(payload))}`" for payload in robocasa_family])
    max_exact = max(float(payload.get("exact_law_utility_mae")) for payload in robocasa_family)
    add_contains(
        checks,
        "README",
        text,
        "robocasa_family_metric_lists",
        f"utility correlations {corr_list}, exact-law MAE at most `{f5(max_exact)}`, and learned-minus-random CI lower bounds {ci_list}",
    )
    add_contains(
        checks,
        "README",
        text,
        "robocasa_stratified97_scale",
        (
            f"evaluates {comma_int(strat97.get('eval_samples'))} heldout rollouts across {strat97.get('eval_rollout_pools')} rollout pools "
            f"and keeps an oracle-minus-learned N8 CI lower bound of `{f4(oracle_best_ci_lo(strat97))}`"
        ),
    )
    residual_nmax = max(int(n) for n in residual35.get("n_values", []))
    add_contains(
        checks,
        "README",
        text,
        "robocasa_residual35_metrics",
        (
            f"trains on {residual35.get('train_samples')} rollouts, validates on {residual35.get('validation_samples')} rollouts "
            f"with utility correlation `{f3(metric(residual35, 'utility_corr'))}`, evaluates on {residual35.get('eval_samples')} heldout rollouts "
            f"across {residual35.get('eval_rollout_pools')} rollout pools, uses horizon `{residual35.get('horizon')}` and Nmax `{residual_nmax}`, "
            f"has exact-law utility MAE `{f5(residual35.get('exact_law_utility_mae'))}`, the learned scorer beats random at N4 with CI lower `{f3(best_ci_lo(residual35, residual_nmax))}`, "
            f"and the oracle-minus-learned N4 CI lower remains `{f4(oracle_best_ci_lo(residual35, residual_nmax))}`"
        ),
    )
    add_contains(
        checks,
        "README",
        text,
        "robocasa_residual_sweep_and_catalog",
        (
            f"attempted {residual_sweep.get('candidate_task_count')} task IDs, completed {residual_sweep.get('completed_chunk_count')} chunks, "
            f"timed out {residual_sweep.get('timed_out_chunk_count')} chunks, found {residual_sweep.get('runnable_task_count')} runnable IDs, "
            f"and found {residual_sweep.get('nondegenerate_task_count')} nondegenerate IDs. A catalog audit finds {catalog.get('registry_count')} registered local task IDs, "
            f"{catalog.get('verified_artifact_task_count')} task IDs covered by verified rollout-pool artifacts, {catalog.get('micro_rollout_task_count')} task IDs covered by micro-rollout probes, "
            f"and {catalog.get('any_artifact_task_count')} task IDs covered by any committed artifact"
        ),
    )
    add_contains(
        checks,
        "README",
        text,
        "libero_wam_metrics",
        (
            f"with {libero.get('train_samples')} train samples, {libero.get('validation_samples')} validation samples, {libero.get('eval_samples')} heldout eval samples, "
            f"exact-law utility MAE `{f5(libero.get('exact_law_utility_mae'))}`, validation utility correlation `{f3(metric(libero, 'utility_corr'))}`, "
            f"and learned energy-regularized scorer minus random N8 CI lower `{f3(best_ci_lo(libero))}`"
        ),
    )
    add_contains(
        checks,
        "README",
        text,
        "libero_policy_counts",
        (
            f"50/50 successes across 10 tasks and 5 seeds, success-rate CI `[1.0, 1.0]`; a kNN action head trained on {comma_int(libero_action.get('train_examples'))} scripted action examples "
            f"that achieves {libero_action.get('eval_successes')}/{libero_action.get('eval_episodes')} heldout sparse successes"
        ),
    )
    add_contains(
        checks,
        "README",
        text,
        "libero_bc_counts",
        (
            f"trained on {comma_int(libero_bc.get('train_examples'))} scripted action examples that achieves {libero_bc.get('eval_successes')}/{libero_bc.get('eval_episodes')} heldout sparse successes"
        ),
    )
    add_contains(
        checks,
        "README",
        text,
        "libero_visual_bc_counts",
        (
            f"trained on {comma_int(libero_visual.get('train_examples'))} scripted action examples that achieves {libero_visual.get('eval_successes')}/{libero_visual.get('eval_episodes')} heldout sparse successes"
        ),
    )
    # Ensure the scripted JSON itself still supports the fixed 50/50 README sentence.
    checks.append(
        NarrativeCheck(
            surface="README",
            name="libero_scripted_json_support",
            ok=libero_scripted.get("n_episodes") == 50 and libero_scripted.get("n_successes") == 50,
            expected="libero scripted 50/50",
            detail=f"episodes={libero_scripted.get('n_episodes')}, successes={libero_scripted.get('n_successes')}",
        )
    )
    add_no_template_markers(checks, "README", text)


def audit_final_decision(root: Path, results_dir: Path, checks: list[NarrativeCheck]) -> None:
    text = read_text(root / "reports" / "final_decision_report.md")
    artifact_integrity = load_json(results_dir, "artifact_integrity.json")
    artifact_manifest = load_json(results_dir, "artifact_manifest.json")
    figure_quality = load_json(results_dir, "figure_quality.json")
    result_consistency = load_json(results_dir, "result_consistency.json")
    raw_result_recompute = load_json(results_dir, "raw_result_recompute.json")
    script_contracts = load_json(results_dir, "script_contracts.json")
    claim_semantics = load_json(results_dir, "claim_semantics.json")
    claim_evidence_quality = load_json(results_dir, "claim_evidence_quality.json")
    learned = load_json(results_dir, "learned_wam_lite_training.json")
    val = (learned.get("metrics") or {}).get("validation") or {}
    learned_cmp = load_json(results_dir, "learned_wam_vs_analytic_wam.json")
    learned_delta = (((learned_cmp.get("confidence_intervals") or {}).get("deltas") or {}).get("learned_minus_analytic_real_utility_N64") or {})
    gym_robotics = load_json(results_dir, "benchmark_gym_robotics_suite.json")
    metaworld = load_json(results_dir, "benchmark_metaworld_suite.json")
    robosuite = load_json(results_dir, "benchmark_robosuite_suite.json")
    maniskill = load_json(results_dir, "benchmark_maniskill_suite.json")
    visual = load_json(results_dir, "visual_optional.json")
    benchmark_visual = load_json(results_dir, "benchmark_visual_wam_lite.json")
    gym_visual = load_json(results_dir, "benchmark_gym_robotics_visual_wam.json")

    add_contains(checks, "final_decision_report", text, "pytest_count", "`python -m pytest -q`: passed with `71 passed`")
    add_contains(
        checks,
        "final_decision_report",
        text,
        "learned_wam_command_metrics",
        (
            f"learned validation utility MAE `{f4(val.get('utility_mae'))}`, final-position L2 MAE `{f4(val.get('final_position_l2_mae'))}`; "
            f"learned-vs-analytic N64 real-utility delta `{f3(learned_delta.get('mean'))} +/- {f3(learned_delta.get('ci95'))}`"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "benchmark_fetch_exact_mae",
        f"Fetch exact-law utility MAE `{f4(gym_robotics.get('exact_law_utility_mae'))}`",
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "benchmark_metaworld_exact_mae",
        f"Meta-World exact-law utility MAE `{f4(metaworld.get('exact_law_utility_mae'))}`",
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "benchmark_robosuite_exact_mae",
        f"RoboSuite exact-law utility MAE `{f4(robosuite.get('exact_law_utility_mae'))}`",
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "maniskill_command_metrics",
        (
            f"ManiSkill exact-law utility MAE `{f4(maniskill.get('exact_law_utility_mae'))}`; "
            f"ManiSkill closed-loop learned-random CI lower bound `{f4(ci_lo(maniskill, 'closed_loop_learned_minus_random_utility_N8'))}`"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "visual_command_metrics",
        (
            f"toy visual MAE `{f4(visual.get('test_mae'))}`; Reacher RGB WAM utility corr `{f4((benchmark_visual.get('validation') or {}).get('utility_corr'))}`, "
            f"utility MAE `{f4((benchmark_visual.get('validation') or {}).get('utility_mae'))}`, visual-random N32 CI lower `{f4(ci_lo(benchmark_visual, 'visual_minus_random_N32'))}`; "
            f"Fetch RGB WAM mean corr `{f4(gym_visual.get('mean_validation_utility_corr'))}`, visual-random N32 CI lower `{f4(ci_lo(gym_visual, 'visual_minus_random_N32'))}`"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "artifact_integrity_command",
        (
            f"`python scripts/artifact_integrity.py --fail-on-error`: passed with `{artifact_integrity.get('n_references')}` artifact references checked "
            f"and `{artifact_integrity.get('n_issues')}` issues"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "artifact_manifest_command",
        (
            f"`python scripts/artifact_manifest.py --fail-on-error`: passed with `{artifact_manifest.get('n_files')}` scientific artifacts, "
            f"`{artifact_manifest.get('total_bytes')}` bytes, `{artifact_manifest.get('n_checks')}` manifest checks, "
            f"and `{artifact_manifest.get('n_issues')}` issues"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "figure_quality_command",
        (
            f"`python scripts/figure_quality.py --fail-on-error`: passed with `{figure_quality.get('n_figures')}` figures, "
            f"`{figure_quality.get('n_checks')}` image-quality checks, and `{figure_quality.get('n_issues')}` issues"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "result_consistency_command",
        (
            f"`python scripts/result_consistency.py --fail-on-error`: passed with `{result_consistency.get('n_checks')}` consistency checks "
            f"and `{result_consistency.get('n_issues')}` issues"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "raw_result_recompute_command",
        (
            f"`python scripts/raw_result_recompute.py --fail-on-error`: passed with `{raw_result_recompute.get('aggregate_metrics_compared')}` aggregate metrics, "
            f"`{raw_result_recompute.get('exact_law_mae_files')}` exact-law files, `{raw_result_recompute.get('seed_metric_ci_columns')}` seed CI columns, "
            f"and `{raw_result_recompute.get('n_issues')}` issues"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "script_contracts_command",
        (
            f"`python scripts/script_contracts.py --fail-on-error`: passed with `{script_contracts.get('n_scripts')}` scripts, "
            f"`{script_contracts.get('n_checks')}` contract checks, and `{script_contracts.get('n_issues')}` issues"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "claim_semantics_command",
        (
            f"`python scripts/claim_semantics.py --fail-on-error`: passed with `{claim_semantics.get('n_claims')}` claims, "
            f"`{claim_semantics.get('n_checks')}` semantic checks, `{claim_semantics.get('n_ci_claims')}` CI-backed claims, "
            f"and `{claim_semantics.get('n_issues')}` issues"
        ),
    )
    add_contains(
        checks,
        "final_decision_report",
        text,
        "claim_evidence_quality_command",
        (
            f"`python scripts/claim_evidence_quality.py --fail-on-error`: passed with `{claim_evidence_quality.get('n_claims')}` claims, "
            f"`{claim_evidence_quality.get('n_source_links')}` source links, `{claim_evidence_quality.get('n_checks')}` evidence checks, "
            f"and `{claim_evidence_quality.get('n_issues')}` issues"
        ),
    )
    add_no_template_markers(checks, "final_decision_report", text)


def audit_report_templates(root: Path, checks: list[NarrativeCheck]) -> None:
    for name in [
        "maxout_initial_audit.md",
        "maxout_completion_audit.md",
        "reviewer_risk_assessment.md",
        "ablation_report.md",
        "falsification_report.md",
        "claims_report.md",
        "paper_result_summary.md",
    ]:
        add_no_template_markers(checks, name, read_text(root / "reports" / name))


def audit_narrative_consistency(root: Path, results_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    checks: list[NarrativeCheck] = []
    audit_readme(root, results_dir, checks)
    audit_final_decision(root, results_dir, checks)
    audit_report_templates(root, checks)
    issues = [check for check in checks if not check.ok]
    return {
        "verified": len(issues) == 0,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def narrative_consistency_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Narrative Consistency Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('surface')}:{issue.get('name')}` expected: {issue.get('expected')}")
    else:
        lines.append("README and final decision report numerical snippets match the current JSON artifacts for audited high-impact claims.")
    lines.append("")
    return "\n".join(lines)
