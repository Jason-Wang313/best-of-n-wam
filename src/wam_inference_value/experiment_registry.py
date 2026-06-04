from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRegistryEntry:
    name: str
    category: str
    script: str
    result: str
    run_scripts: tuple[str, ...] = ()
    wrapper_snippets: tuple[str, ...] = ()
    table_globs: tuple[str, ...] = ()
    figure_globs: tuple[str, ...] = ()
    require_verified_true: bool = False


@dataclass(frozen=True)
class ExperimentRegistryCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ExperimentRegistryCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ExperimentRegistryCheck(name=name, ok=bool(ok), detail=detail))


def default_experiment_entries() -> list[ExperimentRegistryEntry]:
    return [
        ExperimentRegistryEntry("exp1_exact_law", "core_analytic", "experiments/exact_rollout_law_validation.py", "exp1_exact_rollout_law_validation.json", ("scripts/run_all.sh", "scripts/run_smoke.sh"), table_globs=("exp1_exact_rollout_law_validation.csv",), figure_globs=("exp1_exact_vs_mc_success.png",)),
        ExperimentRegistryEntry("exp1_exact_law_learned", "learned", "experiments/exact_rollout_law_validation.py", "exp1_exact_rollout_law_validation_learned.json", ("scripts/run_learned_wam_toy.sh",), wrapper_snippets=("--dynamics-backend learned",), table_globs=("exp1_exact_rollout_law_validation_learned.csv",), figure_globs=("exp1_exact_vs_mc_success_learned.png",)),
        ExperimentRegistryEntry("exp2_auc_moment", "core_analytic", "experiments/auc_vs_moment_hierarchy.py", "exp2_auc_vs_moment_hierarchy.json", ("scripts/run_all.sh", "scripts/run_smoke.sh"), table_globs=("exp2_*.csv",), figure_globs=("exp2_auc_vs_moment_error.png",)),
        ExperimentRegistryEntry("exp3_pilot", "core_analytic", "experiments/pilot_to_heldout_prediction.py", "exp3_pilot_to_heldout_prediction.json", ("scripts/run_all.sh", "scripts/run_smoke.sh"), table_globs=("exp3_pilot_to_heldout_prediction.csv",), figure_globs=("exp3_pilot_to_heldout_mae.png",)),
        ExperimentRegistryEntry("exp4_scorer", "core_analytic", "experiments/score_function_comparison.py", "exp4_score_function_comparison.json", ("scripts/run_all.sh", "scripts/run_smoke.sh"), table_globs=("exp4_score_function_comparison*.csv",), figure_globs=("exp4_score_function_curves.png",)),
        ExperimentRegistryEntry("exp4_scorer_learned", "learned", "experiments/score_function_comparison.py", "exp4_score_function_comparison_learned.json", ("scripts/run_learned_wam_toy.sh",), wrapper_snippets=("--dynamics-backend learned",), table_globs=("exp4_*_learned.csv",), figure_globs=("exp4_score_function_curves_learned.png",)),
        ExperimentRegistryEntry("exp5_gap", "core_analytic", "experiments/real_vs_imagined_utility_gap.py", "exp5_real_vs_imagined_utility_gap.json", ("scripts/run_all.sh", "scripts/run_smoke.sh"), table_globs=("exp5_*gap*.csv",), figure_globs=("exp5_imagined_vs_real_gap.png",)),
        ExperimentRegistryEntry("exp5_gap_learned", "learned", "experiments/real_vs_imagined_utility_gap.py", "exp5_real_vs_imagined_utility_gap_learned.json", ("scripts/run_learned_wam_toy.sh",), wrapper_snippets=("--dynamics-backend learned",), table_globs=("exp5_*learned.csv",), figure_globs=("exp5_imagined_vs_real_gap_learned.png",)),
        ExperimentRegistryEntry("exp6_allocation", "core_analytic", "experiments/adaptive_rollout_allocation.py", "exp6_adaptive_rollout_allocation.json", ("scripts/run_all.sh", "scripts/run_smoke.sh"), table_globs=("exp6_adaptive_rollout_allocation.csv",), figure_globs=("exp6_adaptive_allocation.png",)),
        ExperimentRegistryEntry("exp6_allocation_learned", "learned", "experiments/adaptive_rollout_allocation.py", "exp6_adaptive_rollout_allocation_learned.json", ("scripts/run_learned_wam_toy.sh",), wrapper_snippets=("--dynamics-backend learned",), table_globs=("exp6_adaptive_rollout_allocation_learned.csv",), figure_globs=("exp6_adaptive_allocation_learned.png",)),
        ExperimentRegistryEntry("exp7_closed_loop", "core_analytic", "experiments/closed_loop_receding_horizon_eval.py", "exp7_closed_loop_receding_horizon_eval.json", ("scripts/run_all.sh", "scripts/run_smoke.sh"), table_globs=("exp7_closed_loop_receding_horizon*.csv",), figure_globs=("exp7_closed_loop_success.png",)),
        ExperimentRegistryEntry("exp7_closed_loop_learned", "learned", "experiments/closed_loop_receding_horizon_eval.py", "exp7_closed_loop_receding_horizon_eval_learned.json", ("scripts/run_learned_wam_toy.sh",), wrapper_snippets=("--dynamics-backend learned",), table_globs=("exp7_closed_loop_receding_horizon*_learned.csv",), figure_globs=("exp7_closed_loop_success_learned.png",)),
        ExperimentRegistryEntry("exp8_nonstationary", "core_analytic", "experiments/nonstationary_dynamics_extension.py", "exp8_nonstationary_dynamics_extension.json", ("scripts/run_all.sh", "scripts/run_smoke.sh", "scripts/run_maxout_all.sh"), table_globs=("exp8_nonstationary*.csv",), figure_globs=("exp8_nonstationary_shift.png",)),
        ExperimentRegistryEntry("learned_wam_training", "learned", "experiments/train_learned_wam_lite.py", "learned_wam_lite_training.json", ("scripts/run_learned_wam_toy.sh", "scripts/run_smoke.sh"), table_globs=("learned_wam_lite_*.csv",)),
        ExperimentRegistryEntry("learned_vs_analytic", "learned", "experiments/learned_wam_vs_analytic_wam.py", "learned_wam_vs_analytic_wam.json", ("scripts/run_learned_wam_toy.sh", "scripts/run_smoke.sh"), table_globs=("learned_wam_vs_analytic_wam*.csv",), figure_globs=("learned_wam_vs_analytic_wam.png",)),
        ExperimentRegistryEntry("multi_env_suite", "multi_env", "experiments/multi_env_suite.py", "multi_env_suite.json", ("scripts/run_multi_env.sh",), table_globs=("multi_env_*.csv", "maxout_model_metrics.csv"), figure_globs=("multi_env_inference_curves.png",)),
        ExperimentRegistryEntry("exp10_falsification", "falsification", "experiments/multi_env_suite.py", "exp10_falsification_bad_scorer.json", ("scripts/run_multi_env.sh",), table_globs=("exp10_falsification_bad_scorer.csv",), figure_globs=("exp10_falsification_bad_scorer.png",)),
        ExperimentRegistryEntry("benchmark_smoke", "benchmark", "experiments/benchmark_smoke.py", "benchmark_smoke.json", ("scripts/run_benchmark_smoke.sh",)),
        ExperimentRegistryEntry("benchmark_rollout_pools", "benchmark", "experiments/benchmark_smoke.py", "benchmark_rollout_pools.json", ("scripts/run_benchmark_smoke.sh",)),
        ExperimentRegistryEntry("benchmark_exact_law", "benchmark", "experiments/benchmark_smoke.py", "benchmark_exact_law_validation.json", ("scripts/run_benchmark_smoke.sh",)),
        ExperimentRegistryEntry("benchmark_score_comparison", "benchmark", "experiments/benchmark_smoke.py", "benchmark_score_comparison.json", ("scripts/run_benchmark_smoke.sh",)),
        ExperimentRegistryEntry("benchmark_gap", "benchmark", "experiments/benchmark_smoke.py", "benchmark_real_vs_imagined_gap.json", ("scripts/run_benchmark_smoke.sh",)),
        ExperimentRegistryEntry("benchmark_closed_loop", "benchmark", "experiments/benchmark_smoke.py", "benchmark_closed_loop_eval.json", ("scripts/run_benchmark_smoke.sh",)),
        ExperimentRegistryEntry("benchmark_wam_training", "benchmark", "experiments/benchmark_smoke.py", "benchmark_wam_training.json", ("scripts/run_benchmark_smoke.sh",)),
        ExperimentRegistryEntry("gym_manip_suite", "benchmark", "experiments/benchmark_gym_manip_suite.py", "benchmark_gym_manip_suite.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_gym_manip_*.csv",), figure_globs=("benchmark_gym_manip_curves.png",)),
        ExperimentRegistryEntry("gym_robotics_suite", "benchmark", "experiments/benchmark_gym_robotics_suite.py", "benchmark_gym_robotics_suite.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_gym_robotics_*.csv",), figure_globs=("benchmark_gym_robotics_curves.png", "benchmark_gym_robotics_*_frame.png")),
        ExperimentRegistryEntry("metaworld_suite", "benchmark", "experiments/benchmark_metaworld_suite.py", "benchmark_metaworld_suite.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_metaworld_*.csv",), figure_globs=("benchmark_metaworld_curves.png",)),
        ExperimentRegistryEntry("robosuite_suite", "benchmark", "experiments/benchmark_robosuite_suite.py", "benchmark_robosuite_suite.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_robosuite_*.csv",), figure_globs=("benchmark_robosuite_curves.png",)),
        ExperimentRegistryEntry("maniskill_suite", "benchmark", "experiments/benchmark_maniskill_suite.py", "benchmark_maniskill_suite.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_maniskill_*.csv",), figure_globs=("benchmark_maniskill_curves.png",)),
        ExperimentRegistryEntry("visual_toy", "visual", "experiments/visual_optional.py", "visual_optional.json", ("scripts/run_visual_optional.sh",), figure_globs=("visual_toy_render_example.png",)),
        ExperimentRegistryEntry("benchmark_visual_optional", "visual", "experiments/benchmark_visual_optional.py", "benchmark_visual_optional.json", ("scripts/run_benchmark_visual_optional.sh",), figure_globs=("benchmark_visual_reacher_frame.png",)),
        ExperimentRegistryEntry("benchmark_visual_wam", "visual", "experiments/benchmark_visual_wam_lite.py", "benchmark_visual_wam_lite.json", ("scripts/run_benchmark_visual_optional.sh",), table_globs=("benchmark_visual_wam_lite_*.csv",), figure_globs=("benchmark_visual_wam_lite*.png",)),
        ExperimentRegistryEntry("gym_robotics_visual_wam", "visual", "experiments/benchmark_gym_robotics_visual_wam.py", "benchmark_gym_robotics_visual_wam.json", ("scripts/run_benchmark_visual_optional.sh",), table_globs=("benchmark_gym_robotics_visual_wam_*.csv",), figure_globs=("benchmark_gym_robotics_visual_*.png", "benchmark_gym_robotics_visual_wam_curves.png")),
        ExperimentRegistryEntry("maniskill_visual_probe", "visual", "experiments/benchmark_maniskill_visual_probe.py", "benchmark_maniskill_visual_probe.json", ("scripts/run_benchmark_visual_optional.sh",), table_globs=("benchmark_maniskill_visual_probe.csv",)),
        ExperimentRegistryEntry("inference_audit", "audit", "experiments/inference_audit_framework.py", "inference_audit_framework.json", ("scripts/run_inference_audit.sh",), table_globs=("inference_audit_curves.csv", "inference_audit_decision_counts.csv", "inference_audit_framework.csv", "inference_audit_profile_counts.csv", "inference_audit_seed_metrics.csv"), figure_globs=("inference_audit_tail_alignment.png",)),
        ExperimentRegistryEntry("inference_audit_learned", "audit", "experiments/inference_audit_framework.py", "inference_audit_framework_learned.json", ("scripts/run_inference_audit.sh",), wrapper_snippets=("--dynamics-backend learned",), table_globs=("inference_audit_*_learned.csv",), figure_globs=("inference_audit_tail_alignment_learned.png",)),
        ExperimentRegistryEntry("scorer_repair", "audit", "experiments/scorer_repair_experiment.py", "scorer_repair_experiment.json", ("scripts/run_inference_audit.sh",), table_globs=("scorer_repair*.csv",), figure_globs=("scorer_repair_experiment.png",)),
        ExperimentRegistryEntry("imagination_scaling", "audit", "experiments/imagination_scaling_law.py", "imagination_scaling_law.json", ("scripts/run_inference_audit.sh",), table_globs=("imagination_scaling*.csv",), figure_globs=("imagination_scaling_frontier.png",)),
        ExperimentRegistryEntry("robocasa_smoke", "robocasa", "experiments/benchmark_robocasa_smoke.py", "benchmark_robocasa_smoke.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_robocasa_*.csv",)),
        ExperimentRegistryEntry("robocasa_learned", "robocasa", "experiments/benchmark_robocasa_learned_wam.py", "benchmark_robocasa_learned_wam.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_robocasa_learned_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_multitask", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_multitask_wam.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_robocasa_multitask_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_broad", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_broad_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag broad",), table_globs=("benchmark_robocasa_broad_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_family12", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_family12_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag family12",), table_globs=("benchmark_robocasa_family12_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_family24", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_family24_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag family24",), table_globs=("benchmark_robocasa_family24_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_extra4", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_extra4_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag extra4",), table_globs=("benchmark_robocasa_extra4_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_family28", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_family28_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag family28",), table_globs=("benchmark_robocasa_family28_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_family32", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_family32_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag family32",), table_globs=("benchmark_robocasa_family32_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_stratified55", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_stratified55_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag stratified55",), table_globs=("benchmark_robocasa_stratified55_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_stratified97", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_stratified97_wam.json", ("scripts/run_benchmark_full.sh",), wrapper_snippets=("--output-tag stratified97",), table_globs=("benchmark_robocasa_stratified97_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_residual_sweep", "robocasa", "experiments/benchmark_robocasa_residual_frontier_sweep.py", "benchmark_robocasa_residual_frontier_sweep.json", ("scripts/run_robocasa_residual_probes.sh",), table_globs=("benchmark_robocasa_residual_frontier_sweep_chunks.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_residual35", "robocasa", "experiments/benchmark_robocasa_multitask_wam.py", "benchmark_robocasa_residual35_h1_n4_wam.json", ("scripts/run_robocasa_residual_probes.sh",), wrapper_snippets=("--output-tag residual35_h1_n4",), table_globs=("benchmark_robocasa_residual35_h1_n4_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("robocasa_catalog", "robocasa", "experiments/benchmark_robocasa_catalog_probe.py", "benchmark_robocasa_catalog_probe.json", ("scripts/run_benchmark_full.sh", "scripts/run_robocasa_residual_probes.sh"), table_globs=("benchmark_robocasa_catalog_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("libero_wam", "libero", "experiments/benchmark_libero_wam.py", "benchmark_libero_wam.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_libero_*.csv",), require_verified_true=True),
        ExperimentRegistryEntry("libero_scripted", "libero", "experiments/benchmark_libero_scripted_policy.py", "benchmark_libero_scripted_policy.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_libero_scripted_policy_episodes.csv",), require_verified_true=True),
        ExperimentRegistryEntry("libero_action_head", "libero", "experiments/benchmark_libero_learned_action_head.py", "benchmark_libero_learned_action_head.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_libero_learned_action_head_episodes.csv",), require_verified_true=True),
        ExperimentRegistryEntry("libero_autonomous_bc", "libero", "experiments/benchmark_libero_autonomous_bc_policy.py", "benchmark_libero_autonomous_bc_policy.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_libero_autonomous_bc_policy_episodes.csv",), require_verified_true=True),
        ExperimentRegistryEntry("libero_visual_language_bc", "libero", "experiments/benchmark_libero_visual_language_bc_policy.py", "benchmark_libero_visual_language_bc_policy.json", ("scripts/run_benchmark_full.sh",), table_globs=("benchmark_libero_visual_language_bc_policy_episodes.csv",), require_verified_true=True),
    ]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def glob_tables(root: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    records = []
    table_dir = root / "results" / "tables"
    for pattern in patterns:
        for path in sorted(table_dir.glob(pattern)):
            if path.is_file():
                records.append({"path": path.relative_to(root).as_posix(), "rows": csv_row_count(path)})
    return records


def glob_figures(root: Path, patterns: tuple[str, ...]) -> list[str]:
    records = []
    figure_dir = root / "results" / "figures"
    for pattern in patterns:
        for path in sorted(figure_dir.glob(pattern)):
            if path.is_file():
                records.append(path.relative_to(root).as_posix())
    return records


def audit_experiment_registry(
    root: Path,
    results_dir: Path | None = None,
    entries: list[ExperimentRegistryEntry] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    entries = default_experiment_entries() if entries is None else entries
    checks: list[ExperimentRegistryCheck] = []
    records: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    total_tables = 0
    total_table_rows = 0
    total_figures = 0
    total_wrapper_links = 0

    for entry in entries:
        script_path = root / entry.script
        result_path = results_dir / entry.result
        payload = load_json(result_path)
        run_script_records = []
        wrapper_ok = True
        for run_script in entry.run_scripts:
            path = root / run_script
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            snippets = (entry.script, *entry.wrapper_snippets)
            missing = [snippet for snippet in snippets if snippet not in text]
            run_script_records.append({"path": run_script, "exists": path.exists(), "missing_snippets": missing})
            wrapper_ok = wrapper_ok and path.exists() and not missing
            total_wrapper_links += int(path.exists() and not missing)
        tables = glob_tables(root, entry.table_globs)
        figures = glob_figures(root, entry.figure_globs)
        table_rows = sum(int(table["rows"]) for table in tables)
        verified_field_ok = payload.get("verified") is True if entry.require_verified_true else payload.get("verified") is not False
        record = {
            "name": entry.name,
            "category": entry.category,
            "script": entry.script,
            "result": f"results/{entry.result}",
            "script_exists": script_path.exists(),
            "result_exists": result_path.exists(),
            "result_nonempty": bool(payload),
            "result_experiment": payload.get("experiment"),
            "verified_field_ok": verified_field_ok,
            "run_scripts": run_script_records,
            "tables": tables,
            "figures": figures,
            "ok": script_path.exists()
            and result_path.exists()
            and bool(payload)
            and verified_field_ok
            and wrapper_ok
            and (not entry.table_globs or bool(tables))
            and (not entry.figure_globs or bool(figures))
            and all(int(table["rows"]) > 0 for table in tables),
        }
        records.append(record)
        categories[entry.category] = categories.get(entry.category, 0) + 1
        total_tables += len(tables)
        total_table_rows += table_rows
        total_figures += len(figures)

    failed_records = [record for record in records if not record["ok"]]
    required_categories = {"core_analytic", "learned", "multi_env", "benchmark", "visual", "audit", "robocasa", "libero", "falsification"}
    add(checks, "experiment_registry_entries_present", len(entries) >= 50, f"entries={len(entries)}")
    add(checks, "experiment_registry_categories_present", required_categories.issubset(categories), f"categories={categories}")
    add(checks, "experiment_registry_all_records_ok", not failed_records, f"failed={[record['name'] for record in failed_records]}")
    add(checks, "experiment_registry_wrapper_coverage", total_wrapper_links >= 60, f"wrapper_links={total_wrapper_links}")
    add(checks, "experiment_registry_table_coverage", total_tables >= 90, f"tables={total_tables}")
    add(checks, "experiment_registry_table_row_coverage", total_table_rows >= 50_000, f"rows={total_table_rows}")
    add(checks, "experiment_registry_figure_coverage", total_figures >= 30, f"figures={total_figures}")
    add(checks, "experiment_registry_benchmark_breadth", categories.get("benchmark", 0) >= 10, f"benchmark={categories.get('benchmark', 0)}")
    add(checks, "experiment_registry_robocasa_breadth", categories.get("robocasa", 0) >= 10, f"robocasa={categories.get('robocasa', 0)}")
    add(checks, "experiment_registry_libero_breadth", categories.get("libero", 0) >= 5, f"libero={categories.get('libero', 0)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "experiment_registry",
        "verified": len(issues) == 0,
        "n_entries": len(entries),
        "categories": categories,
        "n_failed_records": len(failed_records),
        "failed_records": failed_records,
        "n_wrapper_links": total_wrapper_links,
        "n_table_artifacts": total_tables,
        "n_table_rows": total_table_rows,
        "n_figure_artifacts": total_figures,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "records": records,
    }


def experiment_registry_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Experiment Registry Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Entries: {payload.get('n_entries')}",
        f"- Categories: {payload.get('categories')}",
        f"- Wrapper links: {payload.get('n_wrapper_links')}",
        f"- Table artifacts: {payload.get('n_table_artifacts')}",
        f"- Table rows: {payload.get('n_table_rows')}",
        f"- Figure artifacts: {payload.get('n_figure_artifacts')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
        failed = payload.get("failed_records") or []
        if failed:
            lines.append("")
            lines.append("## Failed Records")
            lines.append("")
            for record in failed[:50]:
                lines.append(f"- `{record.get('name')}`: script={record.get('script_exists')}, result={record.get('result_exists')}, verified={record.get('verified_field_ok')}, tables={len(record.get('tables') or [])}, figures={len(record.get('figures') or [])}")
    else:
        lines.append("Canonical experiment families have scripts, JSON summaries, wrapper-script coverage, table artifacts, and figure artifacts where expected.")
    lines.append("")
    return "\n".join(lines)
