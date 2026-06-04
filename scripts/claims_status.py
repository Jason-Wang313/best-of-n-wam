from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wam_inference_value.claim_ledger import audit_claim_ledger_payload

_results_override = os.environ.get("WAM_RESULTS_DIR")
RESULTS = Path(_results_override).expanduser() if _results_override else ROOT / "results"
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS
STATUS_JSON = RESULTS / "claims_status.json"
STATUS_MD = RESULTS / "claims_status.md"
README = ROOT / "README.md"
PAPER = ROOT / "paper_outline.md"
REPORTS = ROOT / "reports"
NARRATIVE_SURFACES = [
    ("README", README),
    ("paper_outline", PAPER),
    ("artifact_integrity_report", REPORTS / "artifact_integrity_report.md"),
    ("artifact_manifest_report", REPORTS / "artifact_manifest_report.md"),
    ("figure_quality_report", REPORTS / "figure_quality_report.md"),
    ("result_consistency_report", REPORTS / "result_consistency_report.md"),
    ("raw_result_recompute_report", REPORTS / "raw_result_recompute_report.md"),
    ("table_schema_report", REPORTS / "table_schema_report.md"),
    ("source_manifest_report", REPORTS / "source_manifest_report.md"),
    ("runtime_environment_report", REPORTS / "runtime_environment_report.md"),
    ("experiment_registry_report", REPORTS / "experiment_registry_report.md"),
    ("narrative_consistency_report", REPORTS / "narrative_consistency_report.md"),
    ("claim_semantics_report", REPORTS / "claim_semantics_report.md"),
    ("claim_ledger_integrity_report", REPORTS / "claim_ledger_integrity_report.md"),
    ("script_contracts_report", REPORTS / "script_contracts_report.md"),
    ("claim_evidence_quality_report", REPORTS / "claim_evidence_quality_report.md"),
    ("final_decision_report", REPORTS / "final_decision_report.md"),
    ("paper_result_summary", REPORTS / "paper_result_summary.md"),
    ("reviewer_risk_assessment", REPORTS / "reviewer_risk_assessment.md"),
    ("maxout_completion_audit", REPORTS / "maxout_completion_audit.md"),
]
NARRATIVE_GUARDS = [
    "not ",
    "no ",
    "lacks",
    "missing",
    "future",
    "discussion",
    "do not claim",
    "without claiming",
    "still weaker",
    "unresolved",
    "limitation",
    "blocker",
    "next step",
    "beyond the current",
    "beyond current",
    "rather than",
]
NARRATIVE_RISK_PATTERNS = [
    ("real_robot", re.compile(r"\breal[- ]robot\s+(evidence|validation|validated|result|artifacts?)\b", re.I)),
    ("full_robocasa", re.compile(r"\bfull\s+RoboCasa-wide\s+(validation|learned-WAM validation|rollout collection)\b", re.I)),
    ("modern_vla", re.compile(r"\bmodern\s+VLA\b.*\b(validation|performance|policy)\b", re.I)),
    ("universal_wam_training", re.compile(r"\buniversal\s+WAM\s+(training|train-inference|training recipe|training laws?)\b", re.I)),
    ("dreamzero_uwm", re.compile(r"\b(DreamZero|UWM)(?:-level)?\s+(evidence|integration|validation)\b", re.I)),
]
DICT_VALUES = type({}.values())
SECTION_GUARDS = [
    "future",
    "discussion",
    "do not claim",
    "limitation",
    "unresolved",
    "reviewer attack",
    "remaining gap",
    "weakest claims",
]


def load_json(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status(ok: bool, partial: bool = False, failed: bool = False) -> str:
    if failed:
        return "FAILED"
    if ok:
        return "VERIFIED"
    if partial:
        return "PARTIAL"
    return "UNSUPPORTED"


def ci_positive(payload: dict, key: str, threshold: float = 0.0) -> bool:
    ci = (payload.get("confidence_intervals") or {}).get(key) or {}
    lo = ci.get("lo")
    return lo is not None and lo > threshold


def nested_ci_positive(payload: dict, section: str, key: str, threshold: float = 0.0) -> bool:
    ci = ((payload.get("confidence_intervals") or {}).get(section) or {}).get(key) or {}
    lo = ci.get("lo")
    return lo is not None and lo > threshold


def add(claims: list[dict[str, Any]], cid: int, claim: str, stat: str, evidence: str) -> None:
    claims.append({"id": cid, "claim": claim, "status": stat, "evidence": evidence})


def build_payload(
    claims: list[dict[str, Any]],
    readme_overclaims: list[dict[str, Any]],
    paper_overclaims: list[dict[str, Any]],
    report_overclaims: list[dict[str, Any]],
    narrative: list[dict[str, Any]],
    all_overclaims: list[dict[str, Any]],
) -> dict[str, Any]:
    claims = sorted(claims, key=lambda c: int(c["id"]))
    return {
        "claims": claims,
        "readme_overclaims": readme_overclaims,
        "paper_overclaims": paper_overclaims,
        "report_overclaims": report_overclaims,
        "narrative_overclaims": narrative,
        "overclaims": all_overclaims,
        "num_verified": sum(c["status"] == "VERIFIED" for c in claims),
        "num_partial": sum(c["status"] == "PARTIAL" for c in claims),
        "num_unsupported": sum(c["status"] == "UNSUPPORTED" for c in claims),
        "num_failed": sum(c["status"] == "FAILED" for c in claims),
    }


def artifact_path(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value.expanduser()
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def artifact_exists(value: Any) -> bool:
    path = artifact_path(value)
    return path is not None and path.exists()


def artifacts_exist(values: Any) -> bool:
    if isinstance(values, dict):
        values = values.values()
    if not isinstance(values, (list, tuple, set, DICT_VALUES)):
        values = [values]
    checked = list(values)
    return bool(checked) and all(artifact_exists(value) for value in checked)


def row_artifacts_exist(rows: Any, field: str, minimum: int = 1) -> bool:
    if not isinstance(rows, list):
        return False
    values = [row.get(field) for row in rows if isinstance(row, dict) and row.get(field)]
    return len(values) >= minimum and all(artifact_exists(value) for value in values)


def csv_field_values(path_value: Any, field: str) -> set[str]:
    path = artifact_path(path_value)
    if path is None or not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if field not in (reader.fieldnames or []):
            return set()
        return {row[field] for row in reader if row.get(field)}


def csv_row_count(path_value: Any) -> int:
    path = artifact_path(path_value)
    if path is None or not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def prefixed_tables_ok(prefix: str, requirements: dict[str, int]) -> bool:
    return all(csv_row_count(RESULTS / "tables" / f"{prefix}_{suffix}.csv") >= minimum for suffix, minimum in requirements.items())


def guarded_narrative_line(line: str) -> bool:
    lower = line.lower()
    return any(guard in lower for guard in NARRATIVE_GUARDS)


def guarded_narrative_section(section: str) -> bool:
    lower = section.lower()
    return any(guard in lower for guard in SECTION_GUARDS)


def narrative_overclaims() -> list[dict[str, Any]]:
    overclaims: list[dict[str, Any]] = []
    for surface, path in NARRATIVE_SURFACES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?ims)^##\s*3\.\s*Weakest Claims\s*\n\s*-\s*none\s*$", text):
            overclaims.append(
                {
                    "surface": surface,
                    "id": "narrative",
                    "pattern": "Weakest Claims: none",
                    "status": "UNSUPPORTED",
                    "line": None,
                }
            )
        current_section = ""
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                current_section = line.strip("#").strip()
            for label, pattern in NARRATIVE_RISK_PATTERNS:
                if pattern.search(line) and not guarded_narrative_line(line) and not guarded_narrative_section(current_section):
                    overclaims.append(
                        {
                            "surface": surface,
                            "id": "narrative",
                            "pattern": label,
                            "status": "UNSUPPORTED",
                            "line": lineno,
                            "text": line.strip(),
                        }
                    )
    return overclaims


def main() -> None:
    exp1 = load_json("exp1_exact_rollout_law_validation.json")
    exp2 = load_json("exp2_auc_vs_moment_hierarchy.json")
    exp3 = load_json("exp3_pilot_to_heldout_prediction.json")
    exp4 = load_json("exp4_score_function_comparison.json")
    exp5 = load_json("exp5_real_vs_imagined_utility_gap.json")
    exp6 = load_json("exp6_adaptive_rollout_allocation.json")
    exp7 = load_json("exp7_closed_loop_receding_horizon_eval.json")
    exp8 = load_json("exp8_nonstationary_dynamics_extension.json")
    learned_train = load_json("learned_wam_lite_training.json")
    learned_cmp = load_json("learned_wam_vs_analytic_wam.json")
    learned_exp4 = load_json("exp4_score_function_comparison_learned.json")
    learned_exp5 = load_json("exp5_real_vs_imagined_utility_gap_learned.json")
    learned_exp6 = load_json("exp6_adaptive_rollout_allocation_learned.json")
    learned_exp7 = load_json("exp7_closed_loop_receding_horizon_eval_learned.json")
    multi = load_json("multi_env_suite.json")
    fals = load_json("exp10_falsification_bad_scorer.json")
    bench = load_json("benchmark_smoke.json")
    benchmark_pools = load_json("benchmark_rollout_pools.json")
    benchmark_exact = load_json("benchmark_exact_law_validation.json")
    benchmark_score = load_json("benchmark_score_comparison.json")
    benchmark_gap = load_json("benchmark_real_vs_imagined_gap.json")
    benchmark_closed = load_json("benchmark_closed_loop_eval.json")
    benchmark_wam = load_json("benchmark_wam_training.json")
    maniskill = load_json("benchmark_maniskill_suite.json")
    visual = load_json("visual_optional.json")
    benchmark_visual = load_json("benchmark_visual_optional.json")
    benchmark_visual_wam = load_json("benchmark_visual_wam_lite.json")
    gym_robotics = load_json("benchmark_gym_robotics_suite.json")
    gym_robotics_visual = load_json("benchmark_gym_robotics_visual_wam.json")
    maniskill_visual_probe = load_json("benchmark_maniskill_visual_probe.json")
    maniskill_dependency_probe = load_json("benchmark_maniskill_dependency_probe.json")
    metaworld = load_json("benchmark_metaworld_suite.json")
    robosuite = load_json("benchmark_robosuite_suite.json")
    robocasa = load_json("benchmark_robocasa_smoke.json")
    robocasa_learned = load_json("benchmark_robocasa_learned_wam.json")
    robocasa_multitask = load_json("benchmark_robocasa_multitask_wam.json")
    robocasa_broad = load_json("benchmark_robocasa_broad_wam.json")
    robocasa_family12 = load_json("benchmark_robocasa_family12_wam.json")
    robocasa_family24 = load_json("benchmark_robocasa_family24_wam.json")
    robocasa_extra4 = load_json("benchmark_robocasa_extra4_wam.json")
    robocasa_family28 = load_json("benchmark_robocasa_family28_wam.json")
    robocasa_family32 = load_json("benchmark_robocasa_family32_wam.json")
    robocasa_stratified55 = load_json("benchmark_robocasa_stratified55_wam.json")
    robocasa_stratified97 = load_json("benchmark_robocasa_stratified97_wam.json")
    robocasa_residual35 = load_json("benchmark_robocasa_residual35_h1_n4_wam.json")
    robocasa_catalog = load_json("benchmark_robocasa_catalog_probe.json")
    robocasa_micro = load_json("benchmark_robocasa_micro_rollout_extra.json")
    libero_wam = load_json("benchmark_libero_wam.json")
    libero_scripted = load_json("benchmark_libero_scripted_policy.json")
    libero_action_head = load_json("benchmark_libero_learned_action_head.json")
    libero_autonomous_bc = load_json("benchmark_libero_autonomous_bc_policy.json")
    libero_visual_language_bc = load_json("benchmark_libero_visual_language_bc_policy.json")
    audit = load_json("inference_audit_framework.json")
    audit_learned = load_json("inference_audit_framework_learned.json")
    repair = load_json("scorer_repair_experiment.json")
    scaling = load_json("imagination_scaling_law.json")
    artifact_integrity = load_json("artifact_integrity.json")
    artifact_manifest = load_json("artifact_manifest.json")
    figure_quality = load_json("figure_quality.json")
    result_consistency = load_json("result_consistency.json")
    raw_result_recompute = load_json("raw_result_recompute.json")
    table_schema = load_json("table_schema.json")
    source_manifest = load_json("source_manifest.json")
    runtime_environment = load_json("runtime_environment.json")
    experiment_registry = load_json("experiment_registry.json")
    narrative_consistency = load_json("narrative_consistency.json")
    script_contracts = load_json("script_contracts.json")
    claim_semantics = load_json("claim_semantics.json")
    claim_evidence_quality = load_json("claim_evidence_quality.json")

    claims: list[dict[str, Any]] = []
    add(claims, 1, "Exact finite binary law verified.", status(bool(exp1) and exp1.get("mean_success_mc_mae", 1.0) < 0.018, bool(exp1)), f"success MAE={exp1.get('mean_success_mc_mae')}")
    add(claims, 2, "Utility-valued finite law verified.", status(bool(exp1) and exp1.get("mean_utility_mc_mae", 1.0) < 0.08, bool(exp1)), f"utility MAE={exp1.get('mean_utility_mc_mae')}")
    add(claims, 3, "N=2 AUC identity verified.", status(bool(exp2) and exp2.get("max_n2_identity_error", 1.0) < 1e-12, bool(exp2)), f"max identity error={exp2.get('max_n2_identity_error')}")
    add(claims, 4, "High-N moment hierarchy verified.", status(bool(exp2) and exp2.get("same_p_kappa_counterexample_gap_N64", 0.0) > 0.45, bool(exp2)), f"same-p/kappa gap={exp2.get('same_p_kappa_counterexample_gap_N64')}")
    add(claims, 5, "Pilot-to-heldout improves with K.", status(bool(exp3) and exp3.get("relative_mae_reduction", 0.0) > 0.25, bool(exp3)), f"relative MAE reduction={exp3.get('relative_mae_reduction')}")
    exp3_ci = (exp3.get("confidence_intervals") or {}).get("mae_reduction_first_to_last") or {}
    add(claims, 6, "Pilot uncertainty is reported.", status(exp3_ci.get("n", 0) > 0, bool(exp3)), f"pilot improvement CI={exp3_ci}")
    add(claims, 7, "Score function controls inference value.", status(bool(exp4) and exp4.get("oracle_minus_random_N64", 0.0) > 0.6, bool(exp4)), f"oracle-random N64={exp4.get('oracle_minus_random_N64')}")
    add(claims, 8, "Best non-oracle beats random with CI.", status(ci_positive(learned_exp4, "best_nonoracle_minus_random_N64"), bool(learned_exp4)), f"learned CI={((learned_exp4.get('confidence_intervals') or {}).get('best_nonoracle_minus_random_N64'))}")
    add(claims, 9, "Oracle remains above learned/non-oracle.", status(nested_ci_positive(learned_cmp, "deltas", "oracle_minus_learned_real_utility_N64"), bool(learned_cmp)), f"oracle-learned CI={((learned_cmp.get('confidence_intervals') or {}).get('deltas') or {}).get('oracle_minus_learned_real_utility_N64')}")
    add(claims, 10, "Real-vs-imagined utility gap verified.", status(bool(exp5) and exp5.get("severe_gap_growth_minus_none", 0.0) > 0.35, bool(exp5)), f"severe-none={exp5.get('severe_gap_growth_minus_none')}")
    add(claims, 11, "Mismatch gap grows with N.", status(ci_positive(learned_exp5, "severe_gap_growth_minus_none"), bool(learned_exp5)), f"learned severe gap CI={((learned_exp5.get('confidence_intervals') or {}).get('severe_gap_growth_minus_none'))}")
    add(claims, 12, "Bad scorer falsification verified.", status(bool(fals) and fals.get("anti_scorer_mean_N64", 0.0) < fals.get("anti_scorer_mean_N1", -1e9), bool(fals)), f"anti N64={fals.get('anti_scorer_mean_N64')}, N1={fals.get('anti_scorer_mean_N1')}")
    rnd_gap = fals.get("randomized_dynamics_oracle_gap_N64")
    add(
        claims,
        13,
        "Randomized dynamics falsification verified.",
        status(rnd_gap is not None and rnd_gap > 1.0, bool(multi)),
        f"randomized-oracle N64 gap={rnd_gap}",
    )
    add(claims, 14, "Moment/adaptive allocation beats uniform with CI.", status(ci_positive(learned_exp6, "moment_law_improvement_over_uniform"), bool(learned_exp6)), f"moment-uniform CI={((learned_exp6.get('confidence_intervals') or {}).get('moment_law_improvement_over_uniform'))}")
    add(claims, 15, "Adaptive allocation reduces oracle regret.", status(bool(learned_exp6) and learned_exp6.get("oracle_improvement_over_uniform", 0.0) > learned_exp6.get("moment_law_improvement_over_uniform", -1.0), bool(learned_exp6)), f"oracle-uniform={learned_exp6.get('oracle_improvement_over_uniform')}")
    learned_high_n_ci = (learned_exp7.get("confidence_intervals") or {}).get("useful_success_gain_N64_minus_N1") or {}
    analytic_high_n_ok = bool(exp7) and exp7.get("useful_success_gain_N64_minus_N1", 0.0) > 0.12
    learned_high_n_ok = learned_high_n_ci.get("lo") is not None and learned_high_n_ci.get("lo") > 0.0
    add(
        claims,
        16,
        "Closed-loop high-N gain verified.",
        status(analytic_high_n_ok or learned_high_n_ok, bool(exp7) or bool(learned_exp7)),
        f"analytic useful N64-N1={exp7.get('useful_success_gain_N64_minus_N1')}; learned CI={learned_high_n_ci}",
    )
    add(claims, 17, "Useful scorer beats random in closed loop.", status(ci_positive(learned_exp7, "useful_minus_random_success_N64"), bool(learned_exp7)), f"learned useful-random CI={((learned_exp7.get('confidence_intervals') or {}).get('useful_minus_random_success_N64'))}")
    add(claims, 18, "Oracle first-action remains upper bound.", status(ci_positive(learned_exp7, "oracle_first_action_minus_useful_success_N64"), bool(learned_exp7)), f"oracle-useful CI={((learned_exp7.get('confidence_intervals') or {}).get('oracle_first_action_minus_useful_success_N64'))}")
    add(claims, 19, "Conditional law verified under distribution shift.", status(bool(exp8) and exp8.get("mean_abs_error_N16", 1.0) < 0.025, bool(exp8)), f"MAE={exp8.get('mean_abs_error_N16')}")
    exp8_ci = exp8.get("confidence_intervals") or {}
    stale_shift_ci = exp8_ci.get("stale_post_minus_pre_abs_error_N16") or {}
    adaptive_shift_ci = exp8_ci.get("stale_minus_adaptive_post_abs_error_N16") or {}
    add(
        claims,
        20,
        "Stale estimates fail/degrade under shift.",
        status(stale_shift_ci.get("lo") is not None and stale_shift_ci.get("lo") > 0.0, bool(exp8)),
        f"stale post-pre CI={stale_shift_ci}",
    )
    add(
        claims,
        21,
        "Adaptive re-estimation helps under shift.",
        status(adaptive_shift_ci.get("lo") is not None and adaptive_shift_ci.get("lo") > 0.0, bool(exp8)),
        f"stale-adaptive post CI={adaptive_shift_ci}",
    )
    learned_metrics = learned_train.get("metrics") or {}
    learned_validation = learned_metrics.get("validation") or {}
    learned_ood = learned_metrics.get("ood") or []
    learned_dataset_artifacts = learned_train.get("dataset_artifacts") or {}
    add(
        claims,
        22,
        "Learned WAM trained.",
        status(
            bool(learned_train)
            and artifact_exists(learned_train.get("model_path"))
            and artifacts_exist(learned_dataset_artifacts)
            and (learned_validation.get("n_samples") or 0) >= 500,
            bool(learned_train),
        ),
        f"model={learned_train.get('model_path')}",
    )
    add(
        claims,
        23,
        "Learned WAM ID error reported.",
        status(
            bool(learned_train)
            and (learned_validation.get("n_samples") or 0) >= 500
            and learned_validation.get("utility_mae") is not None
            and learned_validation.get("utility_corr") is not None,
            bool(learned_train),
        ),
        f"validation={learned_validation}",
    )
    add(
        claims,
        24,
        "Learned WAM OOD error reported.",
        status(
            bool(learned_train)
            and len(learned_ood) >= 3
            and all((row.get("n_samples") or 0) > 0 and row.get("utility_mae") is not None for row in learned_ood),
            bool(learned_train),
        ),
        f"ood count={len(learned_ood)}",
    )
    add(claims, 25, "Learned WAM reproduces key inference-value claims.", status(nested_ci_positive(learned_cmp, "deltas", "learned_minus_analytic_real_utility_N64"), bool(learned_cmp)), f"learned-analytic CI={((learned_cmp.get('confidence_intervals') or {}).get('deltas') or {}).get('learned_minus_analytic_real_utility_N64')}")
    envs = set(multi.get("envs") or [])
    backbones = set(multi.get("backbones") or [])
    expected_multi_envs = {"block_push", "drawer_pull", "slippery_grasp", "nonstationary_shift", "deformable_toy"}
    expected_backbones = {"horizon_wam", "mlp_dynamics_wam", "ensemble_wam"}
    multi_curves_path = RESULTS / "tables" / "multi_env_curves.csv"
    multi_agg_path = RESULTS / "tables" / "multi_env_curves_aggregate.csv"
    multi_metrics_path = RESULTS / "tables" / "maxout_model_metrics.csv"
    multi_curves_envs = csv_field_values(multi_curves_path, "env")
    multi_agg_envs = csv_field_values(multi_agg_path, "env")
    multi_metric_envs = csv_field_values(multi_metrics_path, "env")
    multi_metric_models = csv_field_values(multi_metrics_path, "model")
    multi_curves_rows = csv_row_count(multi_curves_path)
    multi_metrics_rows = csv_row_count(multi_metrics_path)
    multi_model_files_ok = all(
        artifact_exists(RESULTS / "models" / f"maxout_{env_name}_{model_name}.npz")
        for env_name in expected_multi_envs
        for model_name in expected_backbones
    )
    multi_tables_ok = (
        expected_multi_envs.issubset(envs)
        and expected_multi_envs.issubset(multi_curves_envs)
        and expected_multi_envs.issubset(multi_agg_envs)
        and expected_multi_envs.issubset(multi_metric_envs)
        and expected_backbones.issubset(backbones)
        and expected_backbones.issubset(multi_metric_models)
        and multi_curves_rows >= 1000
        and multi_metrics_rows >= 45
        and multi_model_files_ok
    )
    multi_env_evidence = (
        f"curves_rows={multi_curves_rows}, metrics_rows={multi_metrics_rows}, "
        f"envs={sorted(multi_curves_envs)}, backbones={sorted(multi_metric_models)}, model_files_ok={multi_model_files_ok}"
    )
    add(claims, 26, "BlockPush verified.", status("block_push" in multi_curves_envs and (multi_tables_ok or bool(exp1)), bool(exp1)), f"env=block_push, {multi_env_evidence}")
    add(claims, 27, "DrawerPull verified.", status("drawer_pull" in multi_curves_envs and multi_tables_ok, bool(multi)), f"env=drawer_pull, {multi_env_evidence}")
    add(claims, 28, "SlipperyGrasp verified.", status("slippery_grasp" in multi_curves_envs and multi_tables_ok, bool(multi)), f"env=slippery_grasp, {multi_env_evidence}")
    add(claims, 29, "Nonstationary verified.", status(("nonstationary_shift" in multi_curves_envs and multi_tables_ok) or bool(exp8), bool(exp8)), f"env=nonstationary_shift, exp8_mae={exp8.get('mean_abs_error_N16')}, {multi_env_evidence}")
    add(claims, 30, "Deformable optional.", status("deformable_toy" in multi_curves_envs and multi_tables_ok, bool(multi)), f"env=deformable_toy, implemented={'deformable_toy' in envs}, {multi_env_evidence}")
    benchmark_adapter_files = [
        ROOT / "src" / "wam_inference_value" / "benchmarks" / "base.py",
        ROOT / "src" / "wam_inference_value" / "benchmarks" / "registry.py",
        ROOT / "src" / "wam_inference_value" / "benchmarks" / "maniskill_adapter.py",
        ROOT / "src" / "wam_inference_value" / "benchmarks" / "gym_manip_adapter.py",
    ]
    add(
        claims,
        31,
        "Benchmark adapter available.",
        status(bool(bench) and bench.get("attempted", False) and all(path.exists() for path in benchmark_adapter_files), False),
        f"attempted={bench.get('attempted')}, any_available={bench.get('any_available')}",
    )
    bench_score_ci = (benchmark_score.get("confidence_intervals") or {}).get("oracle_minus_random_real_utility_N32") or {}
    bench_closed_ci = (benchmark_closed.get("confidence_intervals") or {}).get("closed_loop_learned_minus_random_utility_N32") or {}
    benchmark_curves_path = RESULTS / "tables" / "benchmark_gym_manip_curves.csv"
    benchmark_exact_path = benchmark_exact.get("artifact")
    add(
        claims,
        32,
        "Benchmark rollout pools collected.",
        status(
            (benchmark_pools.get("n_rollout_pools", 0) >= 25)
            and (benchmark_pools.get("n_rollouts", 0) >= 64)
            and csv_row_count(benchmark_curves_path) >= 500,
            bool(bench) and bench.get("any_available", False),
        ),
        f"pools={benchmark_pools.get('n_rollout_pools')}",
    )
    add(
        claims,
        33,
        "Benchmark exact law verified.",
        status(
            benchmark_exact.get("utility_mae") is not None
            and benchmark_exact.get("utility_mae") < 0.08
            and artifact_exists(benchmark_exact_path)
            and csv_row_count(benchmark_exact_path) >= 100,
            bool(benchmark_exact),
        ),
        f"utility MAE={benchmark_exact.get('utility_mae')}",
    )
    add(claims, 34, "Benchmark score comparison verified.", status(bench_score_ci.get("lo") is not None and bench_score_ci.get("lo") > 0.0, bool(benchmark_score)), f"oracle-random CI={bench_score_ci}")
    add(
        claims,
        35,
        "Benchmark real-vs-imagined gap verified.",
        status(
            benchmark_gap.get("gap_growth_N32_minus_N1") is not None
            and benchmark_gap.get("gap_growth_N32_minus_N1") > 0.05
            and artifact_exists(benchmark_gap.get("artifact")),
            bool(benchmark_gap),
        ),
        f"gap growth={benchmark_gap.get('gap_growth_N32_minus_N1')}",
    )
    add(claims, 36, "Benchmark closed-loop verified.", status(bench_closed_ci.get("lo") is not None and bench_closed_ci.get("lo") > 0.0, bool(benchmark_closed)), f"learned-random closed-loop CI={bench_closed_ci}")
    add(
        claims,
        37,
        "Benchmark learned WAM trained.",
        status(
            bool(benchmark_wam.get("model_path"))
            and artifact_exists(benchmark_wam.get("model_path"))
            and len(benchmark_wam.get("model_metrics") or []) >= 2,
            bool(benchmark_wam),
        ),
        f"model={benchmark_wam.get('model_path')}",
    )
    add(claims, 38, "Visual toy WAM attempted.", status(bool(visual) and visual.get("attempted", False) and artifact_exists(visual.get("artifact")), False), f"visual={visual.get('attempted')}")
    add(
        claims,
        39,
        "Visual toy WAM verified if artifacts exist.",
        status(bool(visual) and visual.get("verified", False) and (visual.get("test_mae") or 1.0) < 0.05 and artifact_exists(visual.get("artifact")), bool(visual)),
        f"test MAE={visual.get('test_mae')}",
    )
    add(
        claims,
        40,
        "Benchmark visual optional.",
        status(
            bool(benchmark_visual)
            and benchmark_visual.get("attempted", False)
            and benchmark_visual.get("verified", False)
            and (benchmark_visual.get("frame_std") or 0.0) > 1.0
            and artifact_exists(benchmark_visual.get("artifact")),
            bool(benchmark_visual),
        ),
        f"verified={benchmark_visual.get('verified')}, frame_std={benchmark_visual.get('frame_std')}",
    )

    audit_ci = audit.get("confidence_intervals") or {}
    learned_audit_ci = audit_learned.get("confidence_intervals") or {}
    repair_ci = repair.get("confidence_intervals") or {}
    scaling_ci = scaling.get("confidence_intervals") or {}
    audit_artifacts = audit.get("artifacts") or {}
    add(
        claims,
        41,
        "Inference-value audit profiles generated.",
        status(
            bool(audit)
            and bool(audit.get("profile_counts"))
            and bool(audit.get("decision_counts"))
            and artifacts_exist(audit_artifacts),
            bool(audit),
        ),
        f"profiles={len(audit.get('profile_counts') or [])}, decisions={len(audit.get('decision_counts') or [])}",
    )
    add(
        claims,
        42,
        "Tail alignment predicts high-N inference value.",
        status((audit_ci.get("tail_alignment_gain_corr") or {}).get("lo") is not None and (audit_ci.get("tail_alignment_gain_corr") or {}).get("lo") > 0.5, bool(audit)),
        f"tail-gain corr CI={audit_ci.get('tail_alignment_gain_corr')}",
    )
    anti_block_ci = audit_ci.get("anti_block_high_n_rate") or {}
    anti_harm_ci = audit_ci.get("anti_harm_magnitude") or {}
    add(
        claims,
        43,
        "Audit gate blocks harmful high-N bad-scorer deployments.",
        status(
            anti_block_ci.get("lo") is not None
            and anti_block_ci.get("lo") > 0.95
            and anti_harm_ci.get("lo") is not None
            and anti_harm_ci.get("lo") > 0.10,
            bool(audit),
        ),
        f"anti block CI={anti_block_ci}; anti harm CI={anti_harm_ci}",
    )
    add(
        claims,
        44,
        "Stop-rule compute savings are reported.",
        status((audit_ci.get("stop_rule_saved_rollout_fraction") or {}).get("lo") is not None and (audit_ci.get("stop_rule_saved_rollout_fraction") or {}).get("lo") > 0.05, bool(audit)),
        f"saved rollout fraction CI={audit_ci.get('stop_rule_saved_rollout_fraction')}",
    )
    add(
        claims,
        45,
        "Learned-backend inference audit reproduced.",
        status(
            bool(audit_learned)
            and (learned_audit_ci.get("tail_alignment_gain_corr") or {}).get("lo") is not None
            and (learned_audit_ci.get("anti_block_high_n_rate") or {}).get("lo") is not None,
            bool(audit_learned),
        ),
        f"learned tail-gain CI={learned_audit_ci.get('tail_alignment_gain_corr')}; anti block CI={learned_audit_ci.get('anti_block_high_n_rate')}",
    )
    add(
        claims,
        46,
        "Pilot-calibrated scorer repair improves heldout high-N utility.",
        status((repair_ci.get("repair_minus_predicted_N64") or {}).get("lo") is not None and (repair_ci.get("repair_minus_predicted_N64") or {}).get("lo") > 0.0, bool(repair)),
        f"repair-predicted CI={repair_ci.get('repair_minus_predicted_N64')}",
    )
    add(
        claims,
        47,
        "Robot-imagination compute frontier is measured.",
        status(
            (scaling_ci.get("predicted_gain_N128_minus_N1") or {}).get("lo") is not None
            and (scaling_ci.get("predicted_gain_N128_minus_N1") or {}).get("lo") > 0.0
            and (scaling_ci.get("oracle_minus_predicted_gain") or {}).get("lo") is not None
            and (scaling_ci.get("oracle_minus_predicted_gain") or {}).get("lo") > 0.0,
            bool(scaling),
        ),
        f"pred gain CI={scaling_ci.get('predicted_gain_N128_minus_N1')}; oracle-pred gain CI={scaling_ci.get('oracle_minus_predicted_gain')}",
    )

    maniskill_ci = maniskill.get("confidence_intervals") or {}
    maniskill_artifacts = maniskill.get("artifacts") or {}
    maniskill_curves_rows = csv_row_count(maniskill_artifacts.get("curves"))
    maniskill_exact_rows = csv_row_count(maniskill_artifacts.get("exact_law"))
    maniskill_model_rows = csv_row_count(maniskill_artifacts.get("model_metrics"))
    maniskill_closed_rows = csv_row_count(maniskill_artifacts.get("closed_loop"))
    maniskill_model_files_ok = all(
        artifact_exists(RESULTS / "models" / f"benchmark_maniskill_{env_id}_horizon_wam.npz")
        for env_id in (maniskill.get("env_ids") or [])
    )
    add(
        claims,
        48,
        "ManiSkill state benchmark suite verified.",
        status(
            bool(maniskill)
            and maniskill.get("available", False)
            and len(maniskill.get("env_ids") or []) >= 3
            and artifacts_exist(maniskill_artifacts),
            bool(maniskill),
        ),
        f"envs={maniskill.get('env_ids')}, control={maniskill.get('control_mode')}",
    )
    add(
        claims,
        49,
        "ManiSkill rollout pools collected.",
        status(
            (maniskill.get("n_rollout_pools") or 0) >= 25
            and (maniskill.get("n_rollouts") or 0) >= 32
            and maniskill_curves_rows >= 500,
            bool(maniskill),
        ),
        f"pools={maniskill.get('n_rollout_pools')}, rollouts={maniskill.get('n_rollouts')}, rows={maniskill_curves_rows}",
    )
    add(
        claims,
        50,
        "ManiSkill exact law verified.",
        status(
            maniskill.get("exact_law_utility_mae") is not None
            and maniskill.get("exact_law_utility_mae") < 0.03
            and maniskill_exact_rows >= 100,
            bool(maniskill),
        ),
        f"utility MAE={maniskill.get('exact_law_utility_mae')}, exact rows={maniskill_exact_rows}",
    )
    dense_ci = maniskill_ci.get("dense_minus_random_real_utility_N32") or {}
    oracle_mani_ci = maniskill_ci.get("oracle_minus_random_real_utility_N32") or {}
    add(
        claims,
        51,
        "ManiSkill score comparison verified.",
        status(
            dense_ci.get("lo") is not None
            and dense_ci.get("lo") > 0.0
            and oracle_mani_ci.get("lo") is not None
            and oracle_mani_ci.get("lo") > 0.0
            and maniskill_curves_rows >= 500,
            bool(maniskill),
        ),
        f"dense-random CI={dense_ci}; oracle-random CI={oracle_mani_ci}",
    )
    add(
        claims,
        52,
        "ManiSkill WAM-lite trained and evaluated.",
        status(
            len(maniskill.get("model_metrics") or []) >= 6
            and maniskill_model_rows >= 6
            and maniskill_model_files_ok,
            bool(maniskill),
        ),
        f"model metric rows={len(maniskill.get('model_metrics') or [])}, table rows={maniskill_model_rows}",
    )
    mani_closed_ci = maniskill_ci.get("closed_loop_learned_minus_random_utility_N8") or {}
    add(
        claims,
        53,
        "ManiSkill closed-loop learned scorer beats random.",
        status(
            mani_closed_ci.get("lo") is not None
            and mani_closed_ci.get("lo") > 0.0
            and maniskill_closed_rows >= 50,
            bool(maniskill),
        ),
        f"learned-random closed-loop CI={mani_closed_ci}, rows={maniskill_closed_rows}",
    )
    learned_open_ci = maniskill_ci.get("learned_minus_random_real_utility_N32") or {}
    add(
        claims,
        54,
        "ManiSkill learned open-loop scorer is honestly reported.",
        status(learned_open_ci.get("n", 0) >= 5 and maniskill_curves_rows >= 500, bool(maniskill)),
        f"learned-random open-loop CI={learned_open_ci}, rows={maniskill_curves_rows}",
    )
    visual_wam_ci = benchmark_visual_wam.get("confidence_intervals") or {}
    benchmark_visual_wam_artifacts = benchmark_visual_wam.get("artifacts") or {}
    benchmark_visual_wam_curves_rows = csv_row_count(benchmark_visual_wam_artifacts.get("table"))
    benchmark_visual_wam_exact_rows = csv_row_count(benchmark_visual_wam_artifacts.get("exact_law"))
    add(
        claims,
        55,
        "Benchmark RGB visual WAM-lite trained and evaluated.",
        status(
            bool(benchmark_visual_wam)
            and benchmark_visual_wam.get("verified", False)
            and (benchmark_visual_wam.get("train_samples") or 0) >= 1000
            and (benchmark_visual_wam.get("validation_samples") or 0) >= 300
            and artifact_exists(benchmark_visual_wam.get("model_path"))
            and artifacts_exist(benchmark_visual_wam_artifacts)
            and (benchmark_visual_wam.get("validation") or {}).get("utility_corr", 0.0) > 0.20,
            bool(benchmark_visual_wam),
        ),
        f"model={benchmark_visual_wam.get('model_type')}, validation={benchmark_visual_wam.get('validation')}",
    )
    add(
        claims,
        56,
        "Benchmark RGB visual WAM exact law verified.",
        status(
            benchmark_visual_wam.get("exact_law_utility_mae") is not None
            and benchmark_visual_wam.get("exact_law_utility_mae") < 0.05
            and benchmark_visual_wam_exact_rows >= 100,
            bool(benchmark_visual_wam),
        ),
        f"utility MAE={benchmark_visual_wam.get('exact_law_utility_mae')}, exact rows={benchmark_visual_wam_exact_rows}",
    )
    add(
        claims,
        57,
        "Benchmark RGB visual WAM scorer beats random with CI.",
        status(
            (visual_wam_ci.get("visual_minus_random_N32") or {}).get("lo") is not None
            and (visual_wam_ci.get("visual_minus_random_N32") or {}).get("lo") > 0.0
            and benchmark_visual_wam_curves_rows >= 500,
            bool(benchmark_visual_wam),
        ),
        f"visual-random CI={visual_wam_ci.get('visual_minus_random_N32')}, rows={benchmark_visual_wam_curves_rows}",
    )
    add(
        claims,
        58,
        "Benchmark RGB visual WAM oracle gap reported.",
        status(
            (visual_wam_ci.get("oracle_minus_visual_N32") or {}).get("lo") is not None
            and (visual_wam_ci.get("oracle_minus_visual_N32") or {}).get("lo") > 0.0
            and benchmark_visual_wam_curves_rows >= 500,
            bool(benchmark_visual_wam),
        ),
        f"oracle-visual CI={visual_wam_ci.get('oracle_minus_visual_N32')}, rows={benchmark_visual_wam_curves_rows}",
    )
    gym_robotics_ci = gym_robotics.get("confidence_intervals") or {}
    gym_robotics_artifacts = gym_robotics.get("artifacts") or {}
    gym_robotics_curves_rows = csv_row_count(gym_robotics_artifacts.get("curves"))
    gym_robotics_exact_rows = csv_row_count(gym_robotics_artifacts.get("exact_law"))
    gym_robotics_closed_rows = csv_row_count(gym_robotics_artifacts.get("closed_loop"))
    gym_robotics_model_files_ok = row_artifacts_exist(gym_robotics.get("model_metrics"), "model_path", minimum=3)
    add(
        claims,
        59,
        "Gymnasium Robotics Fetch benchmark suite verified.",
        status(
            bool(gym_robotics)
            and gym_robotics.get("available", False)
            and len(gym_robotics.get("env_ids") or []) >= 3
            and (gym_robotics.get("n_rollout_pools") or 0) >= 50
            and artifacts_exist(gym_robotics_artifacts)
            and gym_robotics_model_files_ok
            and gym_robotics_curves_rows >= 1000,
            bool(gym_robotics),
        ),
        f"envs={gym_robotics.get('env_ids')}, pools={gym_robotics.get('n_rollout_pools')}, rows={gym_robotics_curves_rows}",
    )
    add(
        claims,
        60,
        "Gymnasium Robotics Fetch exact law verified.",
        status(
            gym_robotics.get("exact_law_utility_mae") is not None
            and gym_robotics.get("exact_law_utility_mae") < 0.06
            and gym_robotics_exact_rows >= 200,
            bool(gym_robotics),
        ),
        f"utility MAE={gym_robotics.get('exact_law_utility_mae')}, exact rows={gym_robotics_exact_rows}",
    )
    add(
        claims,
        61,
        "Gymnasium Robotics learned WAM scorer beats random with CI.",
        status(
            (gym_robotics_ci.get("learned_minus_random_N32") or {}).get("lo") is not None
            and (gym_robotics_ci.get("learned_minus_random_N32") or {}).get("lo") > 0.0
            and gym_robotics_model_files_ok
            and gym_robotics_curves_rows >= 1000,
            bool(gym_robotics),
        ),
        f"learned-random CI={gym_robotics_ci.get('learned_minus_random_N32')}",
    )
    add(
        claims,
        62,
        "Gymnasium Robotics closed-loop learned scorer beats random.",
        status(
            (gym_robotics_ci.get("closed_loop_learned_minus_random_N32") or {}).get("lo") is not None
            and (gym_robotics_ci.get("closed_loop_learned_minus_random_N32") or {}).get("lo") > 0.0
            and gym_robotics_closed_rows >= 100,
            bool(gym_robotics),
        ),
        f"closed-loop learned-random CI={gym_robotics_ci.get('closed_loop_learned_minus_random_N32')}, rows={gym_robotics_closed_rows}",
    )
    add(
        claims,
        63,
        "Gymnasium Robotics oracle gap reported.",
        status(
            (gym_robotics_ci.get("oracle_minus_learned_N32") or {}).get("lo") is not None
            and (gym_robotics_ci.get("oracle_minus_learned_N32") or {}).get("lo") > 0.0
            and gym_robotics_curves_rows >= 1000,
            bool(gym_robotics),
        ),
        f"oracle-learned CI={gym_robotics_ci.get('oracle_minus_learned_N32')}",
    )
    gym_robotics_visual_ci = gym_robotics_visual.get("confidence_intervals") or {}
    gym_robotics_visual_artifacts = gym_robotics_visual.get("artifacts") or {}
    gym_robotics_visual_curves_rows = csv_row_count(gym_robotics_visual_artifacts.get("curves"))
    gym_robotics_visual_exact_rows = csv_row_count(gym_robotics_visual_artifacts.get("exact_law"))
    gym_robotics_visual_model_files_ok = row_artifacts_exist(gym_robotics_visual.get("model_metrics"), "model_path", minimum=3)
    gym_robotics_visual_frames_ok = row_artifacts_exist(gym_robotics_visual.get("model_metrics"), "frame_path", minimum=3)
    add(
        claims,
        64,
        "Gymnasium Robotics RGB visual WAM trained and evaluated.",
        status(
            bool(gym_robotics_visual)
            and gym_robotics_visual.get("verified", False)
            and len(gym_robotics_visual.get("env_ids") or []) >= 3
            and (gym_robotics_visual.get("mean_validation_utility_corr") or 0.0) > 0.25
            and artifacts_exist(gym_robotics_visual_artifacts)
            and gym_robotics_visual_model_files_ok
            and gym_robotics_visual_frames_ok
            and gym_robotics_visual_curves_rows >= 1000,
            bool(gym_robotics_visual),
        ),
        f"envs={gym_robotics_visual.get('env_ids')}, mean corr={gym_robotics_visual.get('mean_validation_utility_corr')}, rows={gym_robotics_visual_curves_rows}",
    )
    add(
        claims,
        65,
        "Gymnasium Robotics RGB visual exact law verified.",
        status(
            gym_robotics_visual.get("exact_law_utility_mae") is not None
            and gym_robotics_visual.get("exact_law_utility_mae") < 0.08
            and gym_robotics_visual_exact_rows >= 200,
            bool(gym_robotics_visual),
        ),
        f"utility MAE={gym_robotics_visual.get('exact_law_utility_mae')}, exact rows={gym_robotics_visual_exact_rows}",
    )
    add(
        claims,
        66,
        "Gymnasium Robotics RGB visual scorer beats random with CI.",
        status(
            (gym_robotics_visual_ci.get("visual_minus_random_N32") or {}).get("lo") is not None
            and (gym_robotics_visual_ci.get("visual_minus_random_N32") or {}).get("lo") > 0.0
            and gym_robotics_visual_curves_rows >= 1000,
            bool(gym_robotics_visual),
        ),
        f"visual-random CI={gym_robotics_visual_ci.get('visual_minus_random_N32')}",
    )
    add(
        claims,
        67,
        "Gymnasium Robotics RGB visual oracle gap is reported without requiring significance.",
        status(
            (gym_robotics_visual_ci.get("oracle_minus_visual_N32") or {}).get("n", 0) >= 5
            and gym_robotics_visual_curves_rows >= 1000,
            bool(gym_robotics_visual),
        ),
        f"oracle-visual CI={gym_robotics_visual_ci.get('oracle_minus_visual_N32')}, rows={gym_robotics_visual_curves_rows}",
    )
    maniskill_visual_probe_artifacts = maniskill_visual_probe.get("artifacts") or {}
    maniskill_dependency_artifacts = maniskill_dependency_probe.get("artifacts") or {}
    add(
        claims,
        68,
        "ManiSkill RGB/RGB-D and EE-control probe is artifact-documented.",
        status(
            bool(maniskill_visual_probe)
            and maniskill_visual_probe.get("attempted", False)
            and maniskill_visual_probe.get("state_baseline_ok", False)
            and (maniskill_visual_probe.get("visual_attempt_count") or 0) >= 5
            and "ErrorOutOfPoolMemory" in str(maniskill_visual_probe.get("visual_blocker", ""))
            and bool(maniskill_dependency_probe)
            and maniskill_dependency_probe.get("attempted", False)
            and not maniskill_dependency_probe.get("pinocchio_import_available", True)
            and not maniskill_dependency_probe.get("pin_binary_wheel_available", True)
            and artifacts_exist(maniskill_visual_probe_artifacts)
            and artifacts_exist(maniskill_dependency_artifacts)
            and csv_row_count(maniskill_visual_probe_artifacts.get("table")) >= 5,
            bool(maniskill_visual_probe),
        ),
        f"visual_success={maniskill_visual_probe.get('any_visual_success')}, blocker={maniskill_visual_probe.get('visual_blocker')}; pinocchio={maniskill_dependency_probe.get('pinocchio_import_available')}, pin_binary={maniskill_dependency_probe.get('pin_binary_wheel_available')}",
    )
    metaworld_ci = metaworld.get("confidence_intervals") or {}
    metaworld_artifacts = metaworld.get("artifacts") or {}
    metaworld_curves_rows = csv_row_count(metaworld_artifacts.get("curves"))
    metaworld_exact_rows = csv_row_count(metaworld_artifacts.get("exact_law"))
    metaworld_model_files_ok = row_artifacts_exist(metaworld.get("model_metrics"), "model_path", minimum=3)
    add(
        claims,
        69,
        "Meta-World ML1 benchmark suite verified.",
        status(
            bool(metaworld)
            and metaworld.get("available", False)
            and (metaworld.get("n_tasks_verified") or 0) >= 3
            and (metaworld.get("n_rollout_pools") or 0) >= 45
            and artifacts_exist(metaworld_artifacts)
            and metaworld_model_files_ok
            and metaworld_curves_rows >= 1000,
            bool(metaworld),
        ),
        f"tasks={metaworld.get('task_names')}, pools={metaworld.get('n_rollout_pools')}, rows={metaworld_curves_rows}",
    )
    add(
        claims,
        70,
        "Meta-World exact law verified.",
        status(
            metaworld.get("exact_law_utility_mae") is not None
            and metaworld.get("exact_law_utility_mae") < 0.04
            and metaworld_exact_rows >= 200,
            bool(metaworld),
        ),
        f"utility MAE={metaworld.get('exact_law_utility_mae')}, exact rows={metaworld_exact_rows}",
    )
    add(
        claims,
        71,
        "Meta-World learned WAM scorer beats random open-loop with CI.",
        status(
            (metaworld_ci.get("learned_minus_random_N32") or {}).get("lo") is not None
            and (metaworld_ci.get("learned_minus_random_N32") or {}).get("lo") > 0.0
            and metaworld_model_files_ok
            and metaworld_curves_rows >= 1000,
            bool(metaworld),
        ),
        f"learned-random CI={metaworld_ci.get('learned_minus_random_N32')}",
    )
    add(
        claims,
        72,
        "Meta-World oracle and benchmark-reward scorers beat random with CI.",
        status(
            (metaworld_ci.get("reward_minus_random_N32") or {}).get("lo") is not None
            and (metaworld_ci.get("reward_minus_random_N32") or {}).get("lo") > 0.0
            and (metaworld_ci.get("oracle_minus_random_N32") or {}).get("lo") is not None
            and (metaworld_ci.get("oracle_minus_random_N32") or {}).get("lo") > 0.0
            and metaworld_curves_rows >= 1000,
            bool(metaworld),
        ),
        f"reward-random CI={metaworld_ci.get('reward_minus_random_N32')}; oracle-random CI={metaworld_ci.get('oracle_minus_random_N32')}",
    )
    robosuite_ci = robosuite.get("confidence_intervals") or {}
    robosuite_artifacts = robosuite.get("artifacts") or {}
    robosuite_curves_rows = csv_row_count(robosuite_artifacts.get("curves"))
    robosuite_exact_rows = csv_row_count(robosuite_artifacts.get("exact_law"))
    robosuite_closed_rows = csv_row_count(robosuite_artifacts.get("closed_loop"))
    robosuite_model_files_ok = row_artifacts_exist(robosuite.get("model_metrics"), "model_path", minimum=3)
    add(
        claims,
        73,
        "RoboSuite Panda manipulation benchmark suite verified.",
        status(
            bool(robosuite)
            and robosuite.get("available", False)
            and (robosuite.get("n_tasks_verified") or 0) >= 3
            and (robosuite.get("n_rollout_pools") or 0) >= 30
            and artifacts_exist(robosuite_artifacts)
            and robosuite_model_files_ok
            and robosuite_curves_rows >= 1000,
            bool(robosuite),
        ),
        f"envs={robosuite.get('env_names')}, pools={robosuite.get('n_rollout_pools')}, rows={robosuite_curves_rows}",
    )
    add(
        claims,
        74,
        "RoboSuite exact law verified.",
        status(
            robosuite.get("exact_law_utility_mae") is not None
            and robosuite.get("exact_law_utility_mae") < 0.02
            and robosuite_exact_rows >= 150,
            bool(robosuite),
        ),
        f"utility MAE={robosuite.get('exact_law_utility_mae')}, exact rows={robosuite_exact_rows}",
    )
    add(
        claims,
        75,
        "RoboSuite learned WAM scorer beats random open-loop with CI.",
        status(
            (robosuite_ci.get("learned_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("learned_minus_random_N32") or {}).get("lo") > 0.0
            and robosuite_model_files_ok
            and robosuite_curves_rows >= 1000,
            bool(robosuite),
        ),
        f"learned-random CI={robosuite_ci.get('learned_minus_random_N32')}",
    )
    add(
        claims,
        76,
        "RoboSuite reward, progress, and oracle scorers beat random with CI.",
        status(
            (robosuite_ci.get("reward_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("reward_minus_random_N32") or {}).get("lo") > 0.0
            and (robosuite_ci.get("progress_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("progress_minus_random_N32") or {}).get("lo") > 0.0
            and (robosuite_ci.get("oracle_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("oracle_minus_random_N32") or {}).get("lo") > 0.0
            and robosuite_curves_rows >= 1000,
            bool(robosuite),
        ),
        f"reward-random CI={robosuite_ci.get('reward_minus_random_N32')}; progress-random CI={robosuite_ci.get('progress_minus_random_N32')}; oracle-random CI={robosuite_ci.get('oracle_minus_random_N32')}",
    )
    add(
        claims,
        77,
        "RoboSuite closed-loop learned and reward scorers beat random.",
        status(
            (robosuite_ci.get("closed_loop_learned_minus_random_N8") or {}).get("lo") is not None
            and (robosuite_ci.get("closed_loop_learned_minus_random_N8") or {}).get("lo") > 0.0
            and (robosuite_ci.get("closed_loop_reward_minus_random_N8") or {}).get("lo") is not None
            and (robosuite_ci.get("closed_loop_reward_minus_random_N8") or {}).get("lo") > 0.0
            and robosuite_closed_rows >= 100,
            bool(robosuite),
        ),
        f"learned-random CI={robosuite_ci.get('closed_loop_learned_minus_random_N8')}; reward-random CI={robosuite_ci.get('closed_loop_reward_minus_random_N8')}, rows={robosuite_closed_rows}",
    )
    robocasa_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa",
        {"curves": 80, "exact_law": 40, "seed_metrics": 5, "rollouts": 80},
    )
    add(
        claims,
        78,
        "RoboCasa kitchen benchmark smoke verified.",
        status(
            bool(robocasa)
            and robocasa.get("available", False)
            and robocasa.get("verified", False)
            and (robocasa.get("n_rollout_pools") or 0) >= 5
            and (robocasa.get("n_rollouts_total") or 0) >= 80
            and robocasa.get("exact_law_utility_mae") is not None
            and robocasa.get("exact_law_utility_mae") < 0.01
            and (((robocasa.get("confidence_intervals") or {}).get("oracle_minus_random_N8") or {}).get("n") or 0) >= 5
            and (((robocasa.get("confidence_intervals") or {}).get("oracle_minus_random_N8") or {}).get("lo") or 0.0) > 0.0
            and robocasa_tables_ok,
            bool(robocasa),
        ),
        f"env={robocasa.get('env_id')}, pools={robocasa.get('n_rollout_pools')}, rollouts={robocasa.get('n_rollouts_total')}, exact MAE={robocasa.get('exact_law_utility_mae')}, oracle-random CI={((robocasa.get('confidence_intervals') or {}).get('oracle_minus_random_N8'))}",
    )
    robocasa_learned_ci = robocasa_learned.get("confidence_intervals") or {}
    robocasa_learned_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_learned",
        {"curves": 120, "exact_law": 40, "train_validation": 112, "eval_rollouts": 80},
    )
    add(
        claims,
        79,
        "RoboCasa learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_learned)
            and robocasa_learned.get("available", False)
            and robocasa_learned.get("verified", False)
            and (robocasa_learned.get("train_samples") or 0) >= 80
            and (robocasa_learned.get("validation_samples") or 0) >= 32
            and (robocasa_learned.get("eval_samples") or 0) >= 80
            and ((robocasa_learned.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_learned.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_learned_ci.get("learned_minus_random_N8") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_learned.get("model_path"))
            and robocasa_learned_tables_ok,
            bool(robocasa_learned),
        ),
        f"train={robocasa_learned.get('train_samples')}, val={robocasa_learned.get('validation_samples')}, eval={robocasa_learned.get('eval_samples')}, utility corr={((robocasa_learned.get('model_metrics') or {}).get('utility_corr'))}, learned-random CI={robocasa_learned_ci.get('learned_minus_random_N8')}",
    )
    robocasa_multitask_ci = robocasa_multitask.get("confidence_intervals") or {}
    robocasa_multitask_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_multitask",
        {"curves": 500, "exact_law": 200, "train_validation": 200, "eval_rollouts": 200, "seed_metrics": 15, "task_metrics": 3},
    )
    add(
        claims,
        80,
        "RoboCasa three-task learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_multitask)
            and robocasa_multitask.get("available", False)
            and robocasa_multitask.get("verified", False)
            and len(robocasa_multitask.get("env_ids") or []) >= 3
            and (robocasa_multitask.get("train_samples") or 0) >= 144
            and (robocasa_multitask.get("validation_samples") or 0) >= 96
            and (robocasa_multitask.get("eval_samples") or 0) >= 240
            and (robocasa_multitask.get("eval_rollout_pools") or 0) >= 15
            and ((robocasa_multitask.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_multitask.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_multitask_ci.get("best_learned_minus_random_N8") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_multitask_ci.get("oracle_minus_best_learned_N8") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_multitask.get("model_path"))
            and robocasa_multitask_tables_ok,
            bool(robocasa_multitask),
        ),
        f"tasks={robocasa_multitask.get('env_ids')}, train={robocasa_multitask.get('train_samples')}, val={robocasa_multitask.get('validation_samples')}, eval={robocasa_multitask.get('eval_samples')}, utility corr={((robocasa_multitask.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_multitask.get('promoted_scorer')}, learned-random CI={robocasa_multitask_ci.get('best_learned_minus_random_N8')}, oracle-learned CI={robocasa_multitask_ci.get('oracle_minus_best_learned_N8')}",
    )

    libero_ci = libero_wam.get("confidence_intervals") or {}
    libero_max_n = max(libero_wam.get("n_values") or [8])
    libero_tables_ok = (
        csv_row_count(libero_wam.get("curves_path")) >= 600
        and csv_row_count(libero_wam.get("exact_path")) >= 240
        and csv_row_count(libero_wam.get("data_path")) >= 288
        and csv_row_count(libero_wam.get("eval_path")) >= 240
        and csv_row_count(libero_wam.get("seed_metrics_path")) >= 15
    )
    add(
        claims,
        83,
        "LIBERO rollout-pool learned WAM-lite benchmark verified.",
        status(
            bool(libero_wam)
            and libero_wam.get("available", False)
            and libero_wam.get("verified", False)
            and len(libero_wam.get("tasks") or []) >= 1
            and (libero_wam.get("train_samples") or 0) >= 64
            and (libero_wam.get("validation_samples") or 0) >= 32
            and (libero_wam.get("eval_samples") or 0) >= 80
            and (libero_wam.get("eval_rollout_pools") or 0) >= 5
            and (libero_wam.get("exact_law_utility_mae") or 1.0) < 0.03
            and ((libero_wam.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and ((libero_ci.get(f"best_learned_minus_random_N{libero_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(libero_wam.get("model_path"))
            and libero_tables_ok,
            bool(libero_wam),
        ),
        f"tasks={libero_wam.get('tasks')}, train={libero_wam.get('train_samples')}, val={libero_wam.get('validation_samples')}, eval={libero_wam.get('eval_samples')}, exact MAE={libero_wam.get('exact_law_utility_mae')}, utility corr={((libero_wam.get('model_metrics') or {}).get('utility_corr'))}, learned-random CI={libero_ci.get(f'best_learned_minus_random_N{libero_max_n}')}",
    )
    robocasa_broad_ci = robocasa_broad.get("confidence_intervals") or {}
    robocasa_broad_max_n = max(robocasa_broad.get("n_values") or [8])
    robocasa_broad_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_broad",
        {"curves": 500, "exact_law": 200, "train_validation": 90, "eval_rollouts": 120, "seed_metrics": 16, "task_metrics": 4},
    )
    add(
        claims,
        84,
        "RoboCasa broad task family learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_broad)
            and robocasa_broad.get("available", False)
            and robocasa_broad.get("verified", False)
            and len(robocasa_broad.get("env_ids") or []) >= 4
            and (robocasa_broad.get("train_samples") or 0) >= 64
            and (robocasa_broad.get("validation_samples") or 0) >= 32
            and (robocasa_broad.get("eval_samples") or 0) >= 128
            and (robocasa_broad.get("eval_rollout_pools") or 0) >= 16
            and ((robocasa_broad.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_broad.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_broad_ci.get(f"best_learned_minus_random_N{robocasa_broad_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_broad.get("model_path"))
            and robocasa_broad_tables_ok,
            bool(robocasa_broad),
        ),
        f"tasks={robocasa_broad.get('env_ids')}, train={robocasa_broad.get('train_samples')}, val={robocasa_broad.get('validation_samples')}, eval={robocasa_broad.get('eval_samples')}, utility corr={((robocasa_broad.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_broad.get('promoted_scorer')}, learned-random CI={robocasa_broad_ci.get(f'best_learned_minus_random_N{robocasa_broad_max_n}')}",
    )
    robocasa_family12_ci = robocasa_family12.get("confidence_intervals") or {}
    robocasa_family12_max_n = max(robocasa_family12.get("n_values") or [8])
    robocasa_family12_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_family12",
        {"curves": 800, "exact_law": 350, "train_validation": 180, "eval_rollouts": 180, "seed_metrics": 24, "task_metrics": 12},
    )
    add(
        claims,
        85,
        "RoboCasa 12-task family learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_family12)
            and robocasa_family12.get("available", False)
            and robocasa_family12.get("verified", False)
            and len(robocasa_family12.get("env_ids") or []) >= 12
            and (robocasa_family12.get("train_samples") or 0) >= 96
            and (robocasa_family12.get("validation_samples") or 0) >= 96
            and (robocasa_family12.get("eval_samples") or 0) >= 192
            and (robocasa_family12.get("eval_rollout_pools") or 0) >= 24
            and ((robocasa_family12.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_family12.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_family12_ci.get(f"best_learned_minus_random_N{robocasa_family12_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_family12_ci.get(f"oracle_minus_best_learned_N{robocasa_family12_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_family12.get("model_path"))
            and robocasa_family12_tables_ok,
            bool(robocasa_family12),
        ),
        f"tasks={robocasa_family12.get('env_ids')}, train={robocasa_family12.get('train_samples')}, val={robocasa_family12.get('validation_samples')}, eval={robocasa_family12.get('eval_samples')}, utility corr={((robocasa_family12.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_family12.get('promoted_scorer')}, learned-random CI={robocasa_family12_ci.get(f'best_learned_minus_random_N{robocasa_family12_max_n}')}",
    )
    robocasa_family24_ci = robocasa_family24.get("confidence_intervals") or {}
    robocasa_family24_max_n = max(robocasa_family24.get("n_values") or [8])
    robocasa_family24_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_family24",
        {"curves": 1600, "exact_law": 700, "train_validation": 350, "eval_rollouts": 350, "seed_metrics": 48, "task_metrics": 24},
    )
    add(
        claims,
        90,
        "RoboCasa 24-task family learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_family24)
            and robocasa_family24.get("available", False)
            and robocasa_family24.get("verified", False)
            and len(robocasa_family24.get("env_ids") or []) >= 24
            and (robocasa_family24.get("train_samples") or 0) >= 192
            and (robocasa_family24.get("validation_samples") or 0) >= 192
            and (robocasa_family24.get("eval_samples") or 0) >= 384
            and (robocasa_family24.get("eval_rollout_pools") or 0) >= 48
            and ((robocasa_family24.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_family24.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_family24_ci.get(f"best_learned_minus_random_N{robocasa_family24_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_family24_ci.get(f"oracle_minus_best_learned_N{robocasa_family24_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_family24.get("model_path"))
            and robocasa_family24_tables_ok,
            bool(robocasa_family24),
        ),
        f"tasks={len(robocasa_family24.get('env_ids') or [])}, train={robocasa_family24.get('train_samples')}, val={robocasa_family24.get('validation_samples')}, eval={robocasa_family24.get('eval_samples')}, utility corr={((robocasa_family24.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_family24.get('promoted_scorer')}, learned-random CI={robocasa_family24_ci.get(f'best_learned_minus_random_N{robocasa_family24_max_n}')}",
    )
    add(
        claims,
        91,
        "RoboCasa registry coverage audit is artifacted.",
        status(
            bool(robocasa_catalog)
            and robocasa_catalog.get("available", False)
            and robocasa_catalog.get("verified", False)
            and (robocasa_catalog.get("registry_count") or 0) >= 300
            and (robocasa_catalog.get("verified_artifact_task_count") or 0) >= 28
            and robocasa_catalog.get("coverage_fraction") is not None
            and csv_row_count(robocasa_catalog.get("registry_path")) >= 300
            and csv_row_count(robocasa_catalog.get("artifact_coverage_path")) >= 10,
            bool(robocasa_catalog),
        ),
        f"registered={robocasa_catalog.get('registry_count')}, rollout_pool_covered={robocasa_catalog.get('verified_artifact_task_count')}, micro_covered={robocasa_catalog.get('micro_rollout_task_count')}, any_artifact={robocasa_catalog.get('any_artifact_task_count')}",
    )
    add(
        claims,
        92,
        "RoboCasa extra-task micro-rollout viability probe verified.",
        status(
            bool(robocasa_micro)
            and robocasa_micro.get("available", False)
            and robocasa_micro.get("verified", False)
            and (robocasa_micro.get("candidate_task_count") or 0) >= 4
            and (robocasa_micro.get("nondegenerate_task_count") or 0) >= 4
            and (robocasa_micro.get("rollouts_per_task") or 0) >= 2
            and (robocasa_micro.get("horizon") or 0) >= 1
            and csv_row_count(robocasa_micro.get("table_path")) >= 4
            and artifact_exists(robocasa_micro.get("report_path")),
            bool(robocasa_micro),
        ),
        f"candidates={robocasa_micro.get('candidate_task_count')}, nondegenerate={robocasa_micro.get('nondegenerate_task_count')}, envs={robocasa_micro.get('nondegenerate_env_ids')}",
    )
    robocasa_extra4_ci = robocasa_extra4.get("confidence_intervals") or {}
    robocasa_extra4_max_n = max(robocasa_extra4.get("n_values") or [8])
    robocasa_extra4_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_extra4",
        {"curves": 500, "exact_law": 200, "train_validation": 90, "eval_rollouts": 120, "seed_metrics": 16, "task_metrics": 4},
    )
    add(
        claims,
        93,
        "RoboCasa extra four-task learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_extra4)
            and robocasa_extra4.get("available", False)
            and robocasa_extra4.get("verified", False)
            and len(robocasa_extra4.get("env_ids") or []) >= 4
            and (robocasa_extra4.get("train_samples") or 0) >= 64
            and (robocasa_extra4.get("validation_samples") or 0) >= 32
            and (robocasa_extra4.get("eval_samples") or 0) >= 128
            and (robocasa_extra4.get("eval_rollout_pools") or 0) >= 16
            and ((robocasa_extra4.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_extra4.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_extra4_ci.get(f"best_learned_minus_random_N{robocasa_extra4_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_extra4_ci.get(f"oracle_minus_best_learned_N{robocasa_extra4_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_extra4.get("model_path"))
            and robocasa_extra4_tables_ok,
            bool(robocasa_extra4),
        ),
        f"tasks={robocasa_extra4.get('env_ids')}, train={robocasa_extra4.get('train_samples')}, val={robocasa_extra4.get('validation_samples')}, eval={robocasa_extra4.get('eval_samples')}, utility corr={((robocasa_extra4.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_extra4.get('promoted_scorer')}, learned-random CI={robocasa_extra4_ci.get(f'best_learned_minus_random_N{robocasa_extra4_max_n}')}, oracle-learned CI={robocasa_extra4_ci.get(f'oracle_minus_best_learned_N{robocasa_extra4_max_n}')}",
    )
    robocasa_family28_ci = robocasa_family28.get("confidence_intervals") or {}
    robocasa_family28_max_n = max(robocasa_family28.get("n_values") or [8])
    robocasa_family28_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_family28",
        {"curves": 1900, "exact_law": 850, "train_validation": 650, "eval_rollouts": 420, "seed_metrics": 56, "task_metrics": 28},
    )
    add(
        claims,
        94,
        "RoboCasa combined 28-task family learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_family28)
            and robocasa_family28.get("available", False)
            and robocasa_family28.get("verified", False)
            and len(robocasa_family28.get("env_ids") or []) >= 28
            and (robocasa_family28.get("train_samples") or 0) >= 448
            and (robocasa_family28.get("validation_samples") or 0) >= 224
            and (robocasa_family28.get("eval_samples") or 0) >= 448
            and (robocasa_family28.get("eval_rollout_pools") or 0) >= 56
            and ((robocasa_family28.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_family28.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_family28_ci.get(f"best_learned_minus_random_N{robocasa_family28_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_family28_ci.get(f"oracle_minus_best_learned_N{robocasa_family28_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_family28.get("model_path"))
            and robocasa_family28_tables_ok,
            bool(robocasa_family28),
        ),
        f"tasks={len(robocasa_family28.get('env_ids') or [])}, train={robocasa_family28.get('train_samples')}, val={robocasa_family28.get('validation_samples')}, eval={robocasa_family28.get('eval_samples')}, utility corr={((robocasa_family28.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_family28.get('promoted_scorer')}, learned-random CI={robocasa_family28_ci.get(f'best_learned_minus_random_N{robocasa_family28_max_n}')}, oracle-learned CI={robocasa_family28_ci.get(f'oracle_minus_best_learned_N{robocasa_family28_max_n}')}",
    )
    robocasa_family32_ci = robocasa_family32.get("confidence_intervals") or {}
    robocasa_family32_max_n = max(robocasa_family32.get("n_values") or [8])
    robocasa_family32_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_family32",
        {"curves": 2200, "exact_law": 950, "train_validation": 700, "eval_rollouts": 500, "seed_metrics": 64, "task_metrics": 32},
    )
    add(
        claims,
        95,
        "RoboCasa combined 32-task family learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_family32)
            and robocasa_family32.get("available", False)
            and robocasa_family32.get("verified", False)
            and len(robocasa_family32.get("env_ids") or []) >= 32
            and (robocasa_family32.get("train_samples") or 0) >= 512
            and (robocasa_family32.get("validation_samples") or 0) >= 256
            and (robocasa_family32.get("eval_samples") or 0) >= 512
            and (robocasa_family32.get("eval_rollout_pools") or 0) >= 64
            and ((robocasa_family32.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_family32.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_family32_ci.get(f"best_learned_minus_random_N{robocasa_family32_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_family32_ci.get(f"oracle_minus_best_learned_N{robocasa_family32_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_family32.get("model_path"))
            and robocasa_family32_tables_ok,
            bool(robocasa_family32),
        ),
        f"tasks={len(robocasa_family32.get('env_ids') or [])}, train={robocasa_family32.get('train_samples')}, val={robocasa_family32.get('validation_samples')}, eval={robocasa_family32.get('eval_samples')}, utility corr={((robocasa_family32.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_family32.get('promoted_scorer')}, learned-random CI={robocasa_family32_ci.get(f'best_learned_minus_random_N{robocasa_family32_max_n}')}, oracle-learned CI={robocasa_family32_ci.get(f'oracle_minus_best_learned_N{robocasa_family32_max_n}')}",
    )
    robocasa_stratified55_ci = robocasa_stratified55.get("confidence_intervals") or {}
    robocasa_stratified55_max_n = max(robocasa_stratified55.get("n_values") or [8])
    robocasa_stratified55_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_stratified55",
        {"curves": 3800, "exact_law": 1600, "train_validation": 1200, "eval_rollouts": 800, "seed_metrics": 100, "task_metrics": 55},
    )
    add(
        claims,
        96,
        "RoboCasa stratified 55-task learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_stratified55)
            and robocasa_stratified55.get("available", False)
            and robocasa_stratified55.get("verified", False)
            and len(robocasa_stratified55.get("env_ids") or []) >= 55
            and (robocasa_stratified55.get("train_samples") or 0) >= 880
            and (robocasa_stratified55.get("validation_samples") or 0) >= 440
            and (robocasa_stratified55.get("eval_samples") or 0) >= 880
            and (robocasa_stratified55.get("eval_rollout_pools") or 0) >= 110
            and ((robocasa_stratified55.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_stratified55.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_stratified55_ci.get(f"best_learned_minus_random_N{robocasa_stratified55_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_stratified55_ci.get(f"oracle_minus_best_learned_N{robocasa_stratified55_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_stratified55.get("model_path"))
            and robocasa_stratified55_tables_ok,
            bool(robocasa_stratified55),
        ),
        f"tasks={len(robocasa_stratified55.get('env_ids') or [])}, train={robocasa_stratified55.get('train_samples')}, val={robocasa_stratified55.get('validation_samples')}, eval={robocasa_stratified55.get('eval_samples')}, utility corr={((robocasa_stratified55.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_stratified55.get('promoted_scorer')}, learned-random CI={robocasa_stratified55_ci.get(f'best_learned_minus_random_N{robocasa_stratified55_max_n}')}, oracle-learned CI={robocasa_stratified55_ci.get(f'oracle_minus_best_learned_N{robocasa_stratified55_max_n}')}",
    )
    robocasa_stratified97_ci = robocasa_stratified97.get("confidence_intervals") or {}
    robocasa_stratified97_max_n = max(robocasa_stratified97.get("n_values") or [8])
    robocasa_stratified97_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_stratified97",
        {"curves": 6500, "exact_law": 3000, "train_validation": 2200, "eval_rollouts": 1500, "seed_metrics": 190, "task_metrics": 97},
    )
    add(
        claims,
        97,
        "RoboCasa stratified 97-task learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_stratified97)
            and robocasa_stratified97.get("available", False)
            and robocasa_stratified97.get("verified", False)
            and len(robocasa_stratified97.get("env_ids") or []) >= 97
            and (robocasa_stratified97.get("train_samples") or 0) >= 1552
            and (robocasa_stratified97.get("validation_samples") or 0) >= 776
            and (robocasa_stratified97.get("eval_samples") or 0) >= 1552
            and (robocasa_stratified97.get("eval_rollout_pools") or 0) >= 194
            and ((robocasa_stratified97.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_stratified97.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_stratified97_ci.get(f"best_learned_minus_random_N{robocasa_stratified97_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_stratified97_ci.get(f"oracle_minus_best_learned_N{robocasa_stratified97_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_stratified97.get("model_path"))
            and robocasa_stratified97_tables_ok,
            bool(robocasa_stratified97),
        ),
        f"tasks={len(robocasa_stratified97.get('env_ids') or [])}, train={robocasa_stratified97.get('train_samples')}, val={robocasa_stratified97.get('validation_samples')}, eval={robocasa_stratified97.get('eval_samples')}, pools={robocasa_stratified97.get('eval_rollout_pools')}, utility corr={((robocasa_stratified97.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_stratified97.get('promoted_scorer')}, learned-random CI={robocasa_stratified97_ci.get(f'best_learned_minus_random_N{robocasa_stratified97_max_n}')}, oracle-learned CI={robocasa_stratified97_ci.get(f'oracle_minus_best_learned_N{robocasa_stratified97_max_n}')}",
    )
    robocasa_residual35_ci = robocasa_residual35.get("confidence_intervals") or {}
    robocasa_residual35_max_n = max(robocasa_residual35.get("n_values") or [4])
    robocasa_residual35_tables_ok = prefixed_tables_ok(
        "benchmark_robocasa_residual35_h1_n4",
        {"curves": 900, "exact_law": 400, "train_validation": 250, "eval_rollouts": 250, "seed_metrics": 35, "task_metrics": 35},
    )
    add(
        claims,
        98,
        "RoboCasa residual 35-task clean/cook learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_residual35)
            and robocasa_residual35.get("available", False)
            and robocasa_residual35.get("verified", False)
            and len(robocasa_residual35.get("env_ids") or []) >= 35
            and (robocasa_residual35.get("train_samples") or 0) >= 140
            and (robocasa_residual35.get("validation_samples") or 0) >= 140
            and (robocasa_residual35.get("eval_samples") or 0) >= 280
            and (robocasa_residual35.get("eval_rollout_pools") or 0) >= 35
            and ((robocasa_residual35.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_residual35.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_residual35_ci.get(f"best_learned_minus_random_N{robocasa_residual35_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_residual35_ci.get(f"oracle_minus_best_learned_N{robocasa_residual35_max_n}") or {}).get("lo") or 0.0) > 0.0
            and artifact_exists(robocasa_residual35.get("model_path"))
            and robocasa_residual35_tables_ok,
            bool(robocasa_residual35),
        ),
        f"tasks={len(robocasa_residual35.get('env_ids') or [])}, train={robocasa_residual35.get('train_samples')}, val={robocasa_residual35.get('validation_samples')}, eval={robocasa_residual35.get('eval_samples')}, pools={robocasa_residual35.get('eval_rollout_pools')}, horizon={robocasa_residual35.get('horizon')}, Nmax={robocasa_residual35_max_n}, utility corr={((robocasa_residual35.get('model_metrics') or {}).get('utility_corr'))}, exact MAE={robocasa_residual35.get('exact_law_utility_mae')}, promoted={robocasa_residual35.get('promoted_scorer')}, learned-random CI={robocasa_residual35_ci.get(f'best_learned_minus_random_N{robocasa_residual35_max_n}')}, oracle-learned CI={robocasa_residual35_ci.get(f'oracle_minus_best_learned_N{robocasa_residual35_max_n}')}",
    )

    libero_scripted_ci = (libero_scripted.get("confidence_intervals") or {}).get("success_rate") or {}
    libero_scripted_rows = csv_row_count(RESULTS / "tables" / "benchmark_libero_scripted_policy_episodes.csv")
    add(
        claims,
        86,
        "LIBERO sparse-success scripted policy smoke verified.",
        status(
            bool(libero_scripted)
            and libero_scripted.get("available", False)
            and libero_scripted.get("verified", False)
            and (libero_scripted.get("n_tasks") or 0) >= 10
            and (libero_scripted.get("n_seeds") or 0) >= 5
            and (libero_scripted.get("n_episodes") or 0) >= 50
            and (libero_scripted.get("n_successes") or 0) == (libero_scripted.get("n_episodes") or -1)
            and (libero_scripted.get("success_rate") or 0.0) >= 0.95
            and (libero_scripted_ci.get("lo") or 0.0) >= 0.9
            and libero_scripted_rows >= 50,
            bool(libero_scripted),
        ),
        f"episodes={libero_scripted.get('n_episodes')}, successes={libero_scripted.get('n_successes')}, rows={libero_scripted_rows}, success CI={libero_scripted_ci}",
    )
    libero_action_head_ci = (libero_action_head.get("confidence_intervals") or {}).get("eval_success_rate") or {}
    libero_action_head_rows = csv_row_count(RESULTS / "tables" / "benchmark_libero_learned_action_head_episodes.csv")
    add(
        claims,
        87,
        "LIBERO learned action-head sparse-success smoke verified.",
        status(
            bool(libero_action_head)
            and libero_action_head.get("available", False)
            and libero_action_head.get("verified", False)
            and len(libero_action_head.get("tasks") or []) >= 10
            and (libero_action_head.get("train_episodes") or 0) >= 20
            and (libero_action_head.get("train_examples") or 0) >= 2000
            and (libero_action_head.get("eval_episodes") or 0) >= 30
            and (libero_action_head.get("eval_successes") or 0) == (libero_action_head.get("eval_episodes") or -1)
            and (libero_action_head_ci.get("lo") or 0.0) >= 0.9
            and artifact_exists(libero_action_head.get("model_path"))
            and libero_action_head_rows >= 30,
            bool(libero_action_head),
        ),
        f"tasks={libero_action_head.get('tasks')}, eval={libero_action_head.get('eval_successes')}/{libero_action_head.get('eval_episodes')}, rows={libero_action_head_rows}, success CI={libero_action_head_ci}",
    )
    libero_autonomous_bc_ci = (libero_autonomous_bc.get("confidence_intervals") or {}).get("eval_success_rate") or {}
    libero_autonomous_bc_policy = libero_autonomous_bc.get("policy") or {}
    libero_autonomous_bc_rows = csv_row_count(RESULTS / "tables" / "benchmark_libero_autonomous_bc_policy_episodes.csv")
    add(
        claims,
        88,
        "LIBERO time-conditioned autonomous low-dimensional BC sparse-success smoke verified.",
        status(
            bool(libero_autonomous_bc)
            and libero_autonomous_bc.get("available", False)
            and libero_autonomous_bc.get("verified", False)
            and len(libero_autonomous_bc.get("tasks") or []) >= 10
            and (libero_autonomous_bc.get("train_episodes") or 0) >= 50
            and (libero_autonomous_bc.get("train_examples") or 0) >= 6000
            and (libero_autonomous_bc.get("eval_episodes") or 0) >= 50
            and (libero_autonomous_bc.get("eval_success_rate") or 0.0) >= 0.95
            and (libero_autonomous_bc_ci.get("lo") or 0.0) >= 0.9
            and libero_autonomous_bc_policy.get("uses_phase_index") is False
            and libero_autonomous_bc_policy.get("uses_target_point_command") is False
            and libero_autonomous_bc_policy.get("uses_step_clock") is True
            and artifact_exists(libero_autonomous_bc.get("model_path"))
            and libero_autonomous_bc_rows >= 50,
            bool(libero_autonomous_bc),
        ),
        f"tasks={libero_autonomous_bc.get('tasks')}, train={libero_autonomous_bc.get('train_examples')}, eval={libero_autonomous_bc.get('eval_successes')}/{libero_autonomous_bc.get('eval_episodes')}, rows={libero_autonomous_bc_rows}, success CI={libero_autonomous_bc_ci}, policy={libero_autonomous_bc_policy}",
    )
    libero_visual_language_bc_ci = (libero_visual_language_bc.get("confidence_intervals") or {}).get("eval_success_rate") or {}
    libero_visual_language_bc_policy = libero_visual_language_bc.get("policy") or {}
    libero_visual_language_bc_rows = csv_row_count(RESULTS / "tables" / "benchmark_libero_visual_language_bc_policy_episodes.csv")
    add(
        claims,
        89,
        "LIBERO RGB/proprio/language BC sparse-success smoke verified.",
        status(
            bool(libero_visual_language_bc)
            and libero_visual_language_bc.get("available", False)
            and libero_visual_language_bc.get("verified", False)
            and len(libero_visual_language_bc.get("tasks") or []) >= 10
            and (libero_visual_language_bc.get("train_episodes") or 0) >= 30
            and (libero_visual_language_bc.get("train_examples") or 0) >= 6000
            and (libero_visual_language_bc.get("eval_episodes") or 0) >= 30
            and (libero_visual_language_bc.get("eval_success_rate") or 0.0) >= 0.9
            and (libero_visual_language_bc_ci.get("lo") or 0.0) >= 0.8
            and libero_visual_language_bc_policy.get("uses_rgb") is True
            and libero_visual_language_bc_policy.get("uses_language") is True
            and libero_visual_language_bc_policy.get("uses_robot_proprio") is True
            and libero_visual_language_bc_policy.get("uses_simulator_object_state") is False
            and libero_visual_language_bc_policy.get("uses_task_id") is False
            and libero_visual_language_bc_policy.get("uses_phase_index") is False
            and libero_visual_language_bc_policy.get("uses_target_point_command") is False
            and artifact_exists(libero_visual_language_bc.get("model_path"))
            and libero_visual_language_bc_rows >= 30,
            bool(libero_visual_language_bc),
        ),
        f"tasks={libero_visual_language_bc.get('tasks')}, train={libero_visual_language_bc.get('train_examples')}, eval={libero_visual_language_bc.get('eval_successes')}/{libero_visual_language_bc.get('eval_episodes')}, rows={libero_visual_language_bc_rows}, success CI={libero_visual_language_bc_ci}, policy={libero_visual_language_bc_policy}",
    )

    readme_text = README.read_text(encoding="utf-8") if README.exists() else ""
    paper_text = PAPER.read_text(encoding="utf-8") if PAPER.exists() else ""
    claim_by_id = {c["id"]: c for c in claims}
    overclaims = []
    for cid, c in claim_by_id.items():
        pattern = f"VERIFIED CLAIM {cid}"
        if pattern.lower() in readme_text.lower() and c["status"] not in {"VERIFIED", "PARTIAL"}:
            overclaims.append({"surface": "README", "id": cid, "pattern": pattern, "status": c["status"]})
        if pattern.lower() in paper_text.lower() and c["status"] not in {"VERIFIED", "PARTIAL"}:
            overclaims.append({"surface": "paper_outline", "id": cid, "pattern": pattern, "status": c["status"]})

    narrative = narrative_overclaims()
    all_overclaims = overclaims + narrative
    readme_overclaims = [o for o in all_overclaims if o["surface"] == "README"]
    paper_overclaims = [o for o in all_overclaims if o["surface"] == "paper_outline"]
    report_overclaims = [o for o in all_overclaims if o["surface"] not in {"README", "paper_outline"}]

    add(claims, 81, "README has no unsupported claims.", status(len(readme_overclaims) == 0), f"README overclaims={len(readme_overclaims)}")
    add(claims, 82, "paper_outline has no unsupported claims.", status(len(paper_overclaims) == 0), f"paper overclaims={len(paper_overclaims)}")
    add(claims, 99, "Narrative reports have no unsupported overclaims.", status(len(report_overclaims) == 0), f"report overclaims={len(report_overclaims)}")
    add(
        claims,
        100,
        "Published result artifact references are internally consistent.",
        status(
            bool(artifact_integrity)
            and artifact_integrity.get("verified", False)
            and (artifact_integrity.get("n_references") or 0) >= 250
            and artifact_integrity.get("n_issues") == 0
            and artifact_exists(RESULTS / "artifact_integrity.json")
            and artifact_exists(REPORTS / "artifact_integrity_report.md"),
            bool(artifact_integrity),
        ),
        f"refs={artifact_integrity.get('n_references')}, issues={artifact_integrity.get('n_issues')}, status_counts={artifact_integrity.get('status_counts')}",
    )
    add(
        claims,
        101,
        "Published result summaries agree with canonical tables.",
        status(
            bool(result_consistency)
            and result_consistency.get("verified", False)
            and (result_consistency.get("n_checks") or 0) >= 150
            and result_consistency.get("n_issues") == 0
            and artifact_exists(RESULTS / "result_consistency.json")
            and artifact_exists(REPORTS / "result_consistency_report.md"),
            bool(result_consistency),
        ),
        f"checks={result_consistency.get('n_checks')}, issues={result_consistency.get('n_issues')}",
    )
    add(
        claims,
        102,
        "Published narrative numbers match current artifacts.",
        status(
            bool(narrative_consistency)
            and narrative_consistency.get("verified", False)
            and (narrative_consistency.get("n_checks") or 0) >= 20
            and narrative_consistency.get("n_issues") == 0
            and artifact_exists(RESULTS / "narrative_consistency.json")
            and artifact_exists(REPORTS / "narrative_consistency_report.md"),
            bool(narrative_consistency),
        ),
        f"checks={narrative_consistency.get('n_checks')}, issues={narrative_consistency.get('n_issues')}",
    )

    script_contract_claim = {
        "id": 104,
        "claim": "Canonical execution scripts preserve required gate contracts.",
        "status": status(
            bool(script_contracts)
            and script_contracts.get("verified", False)
            and (script_contracts.get("n_scripts") or 0) >= 7
            and (script_contracts.get("n_checks") or 0) >= 35
            and script_contracts.get("n_issues") == 0
            and artifact_exists(RESULTS / "script_contracts.json")
            and artifact_exists(REPORTS / "script_contracts_report.md"),
            bool(script_contracts),
        ),
        "evidence": f"scripts={script_contracts.get('n_scripts')}, checks={script_contracts.get('n_checks')}, issues={script_contracts.get('n_issues')}",
    }
    claim_evidence_quality_claim = {
        "id": 105,
        "claim": "Verified claims have mapped source artifacts and quality-checked evidence.",
        "status": status(
            bool(claim_evidence_quality)
            and claim_evidence_quality.get("verified", False)
            and (claim_evidence_quality.get("n_claims") or 0) >= 104
            and (claim_evidence_quality.get("n_source_mapped_claims") or 0) >= (claim_evidence_quality.get("n_claims") or 0)
            and (claim_evidence_quality.get("n_source_links") or 0) >= 120
            and claim_evidence_quality.get("n_issues") == 0
            and artifact_exists(RESULTS / "claim_evidence_quality.json")
            and artifact_exists(REPORTS / "claim_evidence_quality_report.md"),
            bool(claim_evidence_quality),
        ),
        "evidence": (
            f"claims={claim_evidence_quality.get('n_claims')}, mapped={claim_evidence_quality.get('n_source_mapped_claims')}, "
            f"sources={claim_evidence_quality.get('n_source_links')}, issues={claim_evidence_quality.get('n_issues')}"
        ),
    }
    raw_result_recompute_claim = {
        "id": 106,
        "claim": "Published summary metrics recompute from raw result tables.",
        "status": status(
            bool(raw_result_recompute)
            and raw_result_recompute.get("verified", False)
            and (raw_result_recompute.get("aggregate_metrics_compared") or 0) >= 10_000
            and (raw_result_recompute.get("exact_law_mae_files") or 0) >= 20
            and (raw_result_recompute.get("seed_metric_ci_columns") or 0) >= 120
            and raw_result_recompute.get("n_issues") == 0
            and artifact_exists(RESULTS / "raw_result_recompute.json")
            and artifact_exists(REPORTS / "raw_result_recompute_report.md"),
            bool(raw_result_recompute),
        ),
        "evidence": (
            f"aggregate metrics={raw_result_recompute.get('aggregate_metrics_compared')}, "
            f"exact files={raw_result_recompute.get('exact_law_mae_files')}, "
            f"seed CI columns={raw_result_recompute.get('seed_metric_ci_columns')}, "
            f"issues={raw_result_recompute.get('n_issues')}"
        ),
    }
    claim_semantics_claim = {
        "id": 107,
        "claim": "Verified claim wording satisfies semantic threshold checks.",
        "status": status(
            bool(claim_semantics)
            and claim_semantics.get("verified", False)
            and (claim_semantics.get("n_claims") or 0) >= 106
            and (claim_semantics.get("n_checks") or 0) >= 160
            and (claim_semantics.get("n_ci_claims") or 0) >= 50
            and (claim_semantics.get("n_positive_ci_claims") or 0) >= 30
            and (claim_semantics.get("n_error_threshold_claims") or 0) >= 10
            and (claim_semantics.get("n_sane_ci_objects") or 0) >= 60
            and claim_semantics.get("n_issues") == 0
            and artifact_exists(RESULTS / "claim_semantics.json")
            and artifact_exists(REPORTS / "claim_semantics_report.md"),
            bool(claim_semantics),
        ),
        "evidence": (
            f"claims={claim_semantics.get('n_claims')}, checks={claim_semantics.get('n_checks')}, "
            f"CI claims={claim_semantics.get('n_ci_claims')}, positive CI claims={claim_semantics.get('n_positive_ci_claims')}, "
            f"error claims={claim_semantics.get('n_error_threshold_claims')}, issues={claim_semantics.get('n_issues')}"
        ),
    }
    artifact_manifest_claim = {
        "id": 108,
        "claim": "Scientific result artifacts have a deterministic hash manifest.",
        "status": status(
            bool(artifact_manifest)
            and artifact_manifest.get("verified", False)
            and (artifact_manifest.get("n_files") or 0) >= 350
            and (artifact_manifest.get("total_bytes") or 0) >= 10_000_000
            and ((artifact_manifest.get("counts_by_suffix") or {}).get(".csv") or 0) >= 150
            and ((artifact_manifest.get("counts_by_suffix") or {}).get(".json") or 0) >= 80
            and ((artifact_manifest.get("counts_by_suffix") or {}).get(".npz") or 0) >= 20
            and ((artifact_manifest.get("counts_by_suffix") or {}).get(".png") or 0) >= 20
            and artifact_manifest.get("n_issues") == 0
            and artifact_exists(RESULTS / "artifact_manifest.json")
            and artifact_exists(REPORTS / "artifact_manifest_report.md"),
            bool(artifact_manifest),
        ),
        "evidence": (
            f"files={artifact_manifest.get('n_files')}, bytes={artifact_manifest.get('total_bytes')}, "
            f"suffixes={artifact_manifest.get('counts_by_suffix')}, issues={artifact_manifest.get('n_issues')}"
        ),
    }
    figure_quality_claim = {
        "id": 109,
        "claim": "Publication figure artifacts pass image-quality checks.",
        "status": status(
            bool(figure_quality)
            and figure_quality.get("verified", False)
            and (figure_quality.get("n_figures") or 0) >= 30
            and (figure_quality.get("n_expected_figures") or 0) >= 20
            and figure_quality.get("n_issues") == 0
            and not figure_quality.get("missing_expected_figures")
            and artifact_exists(RESULTS / "figure_quality.json")
            and artifact_exists(REPORTS / "figure_quality_report.md"),
            bool(figure_quality),
        ),
        "evidence": (
            f"figures={figure_quality.get('n_figures')}, expected={figure_quality.get('n_expected_figures')}, "
            f"checks={figure_quality.get('n_checks')}, issues={figure_quality.get('n_issues')}"
        ),
    }
    table_schema_claim = {
        "id": 110,
        "claim": "Canonical CSV result tables pass schema and numeric-sanity checks.",
        "status": status(
            bool(table_schema)
            and table_schema.get("verified", False)
            and (table_schema.get("n_tables") or 0) >= 200
            and (table_schema.get("total_rows") or 0) >= 200_000
            and (table_schema.get("numeric_column_instances") or 0) >= 500
            and (table_schema.get("n_checks") or 0) >= 15
            and table_schema.get("n_issues") == 0
            and artifact_exists(RESULTS / "table_schema.json")
            and artifact_exists(REPORTS / "table_schema_report.md"),
            bool(table_schema),
        ),
        "evidence": (
            f"tables={table_schema.get('n_tables')}, rows={table_schema.get('total_rows')}, "
            f"numeric_columns={table_schema.get('numeric_column_instances')}, checks={table_schema.get('n_checks')}, "
            f"issues={table_schema.get('n_issues')}"
        ),
    }
    source_manifest_claim = {
        "id": 111,
        "claim": "Source and verification code have a deterministic hash manifest.",
        "status": status(
            bool(source_manifest)
            and source_manifest.get("verified", False)
            and (source_manifest.get("n_files") or 0) >= 150
            and (source_manifest.get("total_bytes") or 0) >= 1_000_000
            and (source_manifest.get("n_checks") or 0) >= 14
            and source_manifest.get("n_issues") == 0
            and artifact_exists(RESULTS / "source_manifest.json")
            and artifact_exists(REPORTS / "source_manifest_report.md"),
            bool(source_manifest),
        ),
        "evidence": (
            f"files={source_manifest.get('n_files')}, bytes={source_manifest.get('total_bytes')}, "
            f"dirs={source_manifest.get('counts_by_dir')}, checks={source_manifest.get('n_checks')}, "
            f"issues={source_manifest.get('n_issues')}"
        ),
    }
    runtime_environment_claim = {
        "id": 112,
        "claim": "Runtime and dependency environment metadata has a verified manifest.",
        "status": status(
            bool(runtime_environment)
            and runtime_environment.get("verified", False)
            and ((runtime_environment.get("python") or {}).get("version_info") or [0, 0])[:2] >= [3, 10]
            and (runtime_environment.get("n_core_requirements") or 0) >= 4
            and (runtime_environment.get("n_core_missing") or 0) == 0
            and (runtime_environment.get("n_core_version_issues") or 0) == 0
            and (runtime_environment.get("n_requirement_files") or 0) >= 3
            and (runtime_environment.get("n_module_probes") or 0) >= 10
            and (runtime_environment.get("n_command_probes") or 0) >= 5
            and (runtime_environment.get("n_checks") or 0) >= 15
            and runtime_environment.get("n_issues") == 0
            and artifact_exists(RESULTS / "runtime_environment.json")
            and artifact_exists(REPORTS / "runtime_environment_report.md"),
            bool(runtime_environment),
        ),
        "evidence": (
            f"python={(runtime_environment.get('python') or {}).get('version')}, "
            f"core={runtime_environment.get('n_core_requirements')}, absent={runtime_environment.get('n_core_missing')}, "
            f"version_issues={runtime_environment.get('n_core_version_issues')}, "
            f"optional_available={runtime_environment.get('n_optional_available')}/{runtime_environment.get('n_optional_requirements')}, "
            f"modules={runtime_environment.get('n_module_probes')}, commands={runtime_environment.get('n_command_probes')}, "
            f"checks={runtime_environment.get('n_checks')}, issues={runtime_environment.get('n_issues')}"
        ),
    }
    experiment_registry_claim = {
        "id": 113,
        "claim": "Canonical experiment families have verified registry coverage.",
        "status": status(
            bool(experiment_registry)
            and experiment_registry.get("verified", False)
            and (experiment_registry.get("n_entries") or 0) >= 55
            and (experiment_registry.get("n_wrapper_links") or 0) >= 60
            and (experiment_registry.get("n_table_artifacts") or 0) >= 250
            and (experiment_registry.get("n_table_rows") or 0) >= 200_000
            and (experiment_registry.get("n_figure_artifacts") or 0) >= 30
            and (experiment_registry.get("n_failed_records") or 0) == 0
            and (experiment_registry.get("n_checks") or 0) >= 10
            and experiment_registry.get("n_issues") == 0
            and artifact_exists(RESULTS / "experiment_registry.json")
            and artifact_exists(REPORTS / "experiment_registry_report.md"),
            bool(experiment_registry),
        ),
        "evidence": (
            f"entries={experiment_registry.get('n_entries')}, categories={experiment_registry.get('categories')}, "
            f"wrapper_links={experiment_registry.get('n_wrapper_links')}, tables={experiment_registry.get('n_table_artifacts')}, "
            f"rows={experiment_registry.get('n_table_rows')}, figures={experiment_registry.get('n_figure_artifacts')}, "
            f"failed={experiment_registry.get('n_failed_records')}, checks={experiment_registry.get('n_checks')}, "
            f"issues={experiment_registry.get('n_issues')}"
        ),
    }
    candidate_claims = claims + [
        {
            "id": 103,
            "claim": "Claim ledger is structurally consistent.",
            "status": "VERIFIED",
            "evidence": "claims=113, max_id=113, checks=pending, issues=0",
        },
        script_contract_claim,
        claim_evidence_quality_claim,
        raw_result_recompute_claim,
        claim_semantics_claim,
        artifact_manifest_claim,
        figure_quality_claim,
        table_schema_claim,
        source_manifest_claim,
        runtime_environment_claim,
        experiment_registry_claim,
    ]
    candidate_payload = build_payload(candidate_claims, readme_overclaims, paper_overclaims, report_overclaims, narrative, all_overclaims)
    ledger_audit = audit_claim_ledger_payload(candidate_payload, root=ROOT)
    add(
        claims,
        103,
        "Claim ledger is structurally consistent.",
        status(ledger_audit.get("verified", False), partial=True),
        (
            f"claims={ledger_audit.get('n_claims')}, max_id={ledger_audit.get('max_claim_id')}, "
            f"checks={ledger_audit.get('n_checks')}, issues={ledger_audit.get('n_issues')}"
        ),
    )
    claims.append(script_contract_claim)
    claims.append(claim_evidence_quality_claim)
    claims.append(raw_result_recompute_claim)
    claims.append(claim_semantics_claim)
    claims.append(artifact_manifest_claim)
    claims.append(figure_quality_claim)
    claims.append(table_schema_claim)
    claims.append(source_manifest_claim)
    claims.append(runtime_environment_claim)
    claims.append(experiment_registry_claim)

    payload = build_payload(claims, readme_overclaims, paper_overclaims, report_overclaims, narrative, all_overclaims)
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Claims Status", ""]
    for c in payload["claims"]:
        lines.append(f"- Claim {c['id']}: **{c['status']}** - {c['claim']} Evidence: {c['evidence']}")
    if all_overclaims:
        lines.append("")
        lines.append("## Overclaims")
        for item in all_overclaims:
            location = f":{item['line']}" if item.get("line") else ""
            text = f" Text: {item['text']}" if item.get("text") else ""
            lines.append(f"- {item['surface']}{location} claim {item['id']} pattern `{item['pattern']}` has status {item['status']}.{text}")
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if all_overclaims:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
