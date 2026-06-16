from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "v4_frozen_evidence"
FIG_OUT = OUT / "figures"
PAPER_FIG_OUT = ROOT / "paper_figures" / "v4"
MACROS = ROOT / "v4_results_macros.tex"


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ci(payload: dict[str, Any], key: str, field: str = "lo") -> float | None:
    item = (payload.get("confidence_intervals") or {}).get(key) or {}
    value = item.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def mean_model_corr(payload: dict[str, Any]) -> float | None:
    if isinstance(payload.get("mean_validation_utility_corr"), (int, float)):
        return float(payload["mean_validation_utility_corr"])
    metrics = payload.get("model_metrics") or []
    values: list[float] = []
    for row in metrics:
        model = row.get("model_metrics") if isinstance(row, dict) else None
        if isinstance(model, dict) and isinstance(model.get("utility_corr"), (int, float)):
            values.append(float(model["utility_corr"]))
        elif isinstance(row, dict) and isinstance(row.get("utility_corr"), (int, float)):
            values.append(float(row["utility_corr"]))
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return "NA"


def macro_line(name: str, value: str | int | float, digits: int = 2) -> str:
    if isinstance(value, float):
        rendered = fmt(value, digits)
    else:
        rendered = str(value)
    return f"\\newcommand{{\\{name}}}{{{rendered}}}\n"


def barh(path: Path, labels: list[str], values: list[float], title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.38 * len(labels))))
    ax.barh(labels, values, color="#2c7fb8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(
        path,
        metadata={
            "Creator": "experiments/v4_frozen_evidence.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_OUT.mkdir(parents=True, exist_ok=True)

    claims = load_json("results/claims_status.json")
    manifest = load_json("results/artifact_manifest.json")
    exp1 = load_json("results/exp1_exact_rollout_law_validation.json")
    exp2 = load_json("results/exp2_auc_vs_moment_hierarchy.json")
    exp5 = load_json("results/exp5_real_vs_imagined_utility_gap.json")
    exp6 = load_json("results/exp6_adaptive_rollout_allocation.json")
    exp10 = load_json("results/exp10_falsification_bad_scorer.json")
    fetch = load_json("results/benchmark_gym_robotics_suite.json")
    fetch_rgb = load_json("results/benchmark_gym_robotics_visual_wam.json")
    metaworld = load_json("results/benchmark_metaworld_suite.json")
    robosuite = load_json("results/benchmark_robosuite_suite.json")
    maniskill = load_json("results/benchmark_maniskill_suite.json")
    robocasa97 = load_json("results/benchmark_robocasa_stratified97_wam.json")
    robocasa35 = load_json("results/benchmark_robocasa_residual35_h1_n4_wam.json")
    robocasa_catalog = load_json("results/benchmark_robocasa_catalog_probe.json")
    libero = load_json("results/benchmark_libero_wam.json")

    core_rows = [
        {
            "claim": "finite binary identity",
            "metric": "mean success MC MAE",
            "value": exp1["mean_success_mc_mae"],
            "evidence": "results/exp1_exact_rollout_law_validation.json",
        },
        {
            "claim": "utility-valued identity",
            "metric": "mean utility MC MAE",
            "value": exp1["mean_utility_mc_mae"],
            "evidence": "results/exp1_exact_rollout_law_validation.json",
        },
        {
            "claim": "N=2 AUC boundary",
            "metric": "max identity error",
            "value": exp2["max_n2_identity_error"],
            "evidence": "results/exp2_auc_vs_moment_hierarchy.json",
        },
        {
            "claim": "higher tail moments matter",
            "metric": "same-p-kappa N64 gap",
            "value": exp2["same_p_kappa_counterexample_gap_N64"],
            "evidence": "results/exp2_auc_vs_moment_hierarchy.json",
        },
        {
            "claim": "adaptive allocation",
            "metric": "moment-law gain over uniform",
            "value": exp6["moment_law_improvement_over_uniform"],
            "evidence": "results/exp6_adaptive_rollout_allocation.json",
        },
    ]

    failure_rows = [
        {
            "failure_mode": "severe imagined-real mismatch",
            "metric": "gap growth over no mismatch",
            "value": exp5["severe_gap_growth_minus_none"],
            "evidence": "results/exp5_real_vs_imagined_utility_gap.json",
        },
        {
            "failure_mode": "stuck/slip imagined-real mismatch",
            "metric": "gap growth over no mismatch",
            "value": exp5["stuck_slip_gap_growth_minus_none"],
            "evidence": "results/exp5_real_vs_imagined_utility_gap.json",
        },
        {
            "failure_mode": "anti-real-utility scorer",
            "metric": "N64 minus N1 selected utility",
            "value": exp10["anti_scorer_mean_N64"] - exp10["anti_scorer_mean_N1"],
            "evidence": "results/exp10_falsification_bad_scorer.json",
        },
        {
            "failure_mode": "randomized dynamics scorer",
            "metric": "oracle gap at N64",
            "value": exp10["randomized_dynamics_oracle_gap_N64"],
            "evidence": "results/exp10_falsification_bad_scorer.json",
        },
    ]

    benchmark_rows = [
        {
            "benchmark": "Fetch state WAM",
            "pools": fetch["n_rollout_pools"],
            "exact_mae": fetch["exact_law_utility_mae"],
            "utility_corr": mean_model_corr(fetch),
            "ci_key": "learned_minus_random_N32",
            "ci_lo": ci(fetch, "learned_minus_random_N32"),
            "scope": "Gymnasium Robotics state/action rollout pools",
        },
        {
            "benchmark": "Fetch RGB WAM",
            "pools": fetch_rgb["n_rollout_pools"],
            "exact_mae": fetch_rgb["exact_law_utility_mae"],
            "utility_corr": mean_model_corr(fetch_rgb),
            "ci_key": "visual_minus_random_N32",
            "ci_lo": ci(fetch_rgb, "visual_minus_random_N32"),
            "scope": "Gymnasium Robotics RGB-frame/action rollout pools",
        },
        {
            "benchmark": "Meta-World",
            "pools": metaworld["n_rollout_pools"],
            "exact_mae": metaworld["exact_law_utility_mae"],
            "utility_corr": mean_model_corr(metaworld),
            "ci_key": "learned_minus_random_N32",
            "ci_lo": ci(metaworld, "learned_minus_random_N32"),
            "scope": "ML1 reach/push/drawer-open rollout pools",
        },
        {
            "benchmark": "RoboSuite",
            "pools": robosuite["n_rollout_pools"],
            "exact_mae": robosuite["exact_law_utility_mae"],
            "utility_corr": mean_model_corr(robosuite),
            "ci_key": "learned_minus_random_N32",
            "ci_lo": ci(robosuite, "learned_minus_random_N32"),
            "scope": "Panda Lift/Stack/Door rollout pools",
        },
        {
            "benchmark": "ManiSkill state",
            "pools": maniskill["n_rollout_pools"],
            "exact_mae": maniskill["exact_law_utility_mae"],
            "utility_corr": mean_model_corr(maniskill),
            "ci_key": "dense_minus_random_real_utility_N32",
            "ci_lo": ci(maniskill, "dense_minus_random_real_utility_N32"),
            "scope": "CPU state-mode pd_joint_delta_pos rollout pools",
        },
        {
            "benchmark": "RoboCasa 97-task",
            "pools": robocasa97["eval_rollout_pools"],
            "exact_mae": robocasa97["exact_law_utility_mae"],
            "utility_corr": robocasa97["model_metrics"]["utility_corr"],
            "ci_key": "promoted_learned_minus_random_N8",
            "ci_lo": ci(robocasa97, "promoted_learned_minus_random_N8"),
            "scope": "stratified kitchen-family rollout pools",
        },
        {
            "benchmark": "RoboCasa residual 35",
            "pools": robocasa35["eval_rollout_pools"],
            "exact_mae": robocasa35["exact_law_utility_mae"],
            "utility_corr": robocasa35["model_metrics"]["utility_corr"],
            "ci_key": "promoted_learned_minus_random_N4",
            "ci_lo": ci(robocasa35, "promoted_learned_minus_random_N4"),
            "scope": "clean/cook residual rollout pools",
        },
        {
            "benchmark": "LIBERO Spatial",
            "pools": libero["eval_rollout_pools"],
            "exact_mae": libero["exact_law_utility_mae"],
            "utility_corr": libero["model_metrics"]["utility_corr"],
            "ci_key": "learned_energy_regularized_minus_random_N8",
            "ci_lo": ci(libero, "learned_energy_regularized_minus_random_N8"),
            "scope": "three-task rollout-pool WAM-lite",
        },
    ]

    coverage_rows = [
        {
            "coverage": "artifact files",
            "count": manifest["n_files"],
            "note": "files tracked by artifact manifest",
        },
        {
            "coverage": "CSV tables",
            "count": manifest["counts_by_suffix"][".csv"],
            "note": "artifact manifest suffix count",
        },
        {
            "coverage": "JSON summaries",
            "count": manifest["counts_by_suffix"][".json"],
            "note": "artifact manifest suffix count",
        },
        {
            "coverage": "RoboCasa registered tasks",
            "count": robocasa_catalog["registry_count"],
            "note": "local task registry",
        },
        {
            "coverage": "RoboCasa rollout-pool task IDs",
            "count": robocasa_catalog["verified_artifact_task_count"],
            "note": "verified learned-WAM rollout-pool artifacts",
        },
        {
            "coverage": "RoboCasa any-artifact task IDs",
            "count": robocasa_catalog["any_artifact_task_count"],
            "note": "rollout-pool or micro-rollout evidence",
        },
    ]

    write_csv(OUT / "v4_core_claims.csv", core_rows, ["claim", "metric", "value", "evidence"])
    write_csv(OUT / "v4_failure_modes.csv", failure_rows, ["failure_mode", "metric", "value", "evidence"])
    write_csv(
        OUT / "v4_benchmark_summary.csv",
        benchmark_rows,
        ["benchmark", "pools", "exact_mae", "utility_corr", "ci_key", "ci_lo", "scope"],
    )
    write_csv(OUT / "v4_coverage_summary.csv", coverage_rows, ["coverage", "count", "note"])

    benchmark_pools = sum(int(row["pools"]) for row in benchmark_rows)
    positive_ci_rows = sum(1 for row in benchmark_rows if float(row["ci_lo"] or 0.0) > 0.0)
    max_exact_mae = max(float(row["exact_mae"]) for row in benchmark_rows)
    min_benchmark_ci_lo = min(float(row["ci_lo"] or 0.0) for row in benchmark_rows)
    negative_control_passes = sum(
        [
            exp5["severe_gap_growth_minus_none"] > 10.0,
            exp5["stuck_slip_gap_growth_minus_none"] > 10.0,
            exp10["anti_scorer_mean_N64"] < exp10["anti_scorer_mean_N1"],
            exp10["randomized_dynamics_oracle_gap_N64"] > 1.0,
        ]
    )
    protocol_rows = [
        {
            "gate": "claim ledger clean",
            "threshold": "partial=0, unsupported=0, failed=0",
            "value": claims["num_partial"] + claims["num_unsupported"] + claims.get("num_failed", 0),
            "pass": claims["num_partial"] == 0 and claims["num_unsupported"] == 0 and claims.get("num_failed", 0) == 0,
        },
        {
            "gate": "core exact success MAE",
            "threshold": "<0.003",
            "value": exp1["mean_success_mc_mae"],
            "pass": exp1["mean_success_mc_mae"] < 0.003,
        },
        {
            "gate": "core exact utility MAE",
            "threshold": "<0.02",
            "value": exp1["mean_utility_mc_mae"],
            "pass": exp1["mean_utility_mc_mae"] < 0.02,
        },
        {
            "gate": "high-N AUC counterexample",
            "threshold": ">0.9 N64 gap",
            "value": exp2["same_p_kappa_counterexample_gap_N64"],
            "pass": exp2["same_p_kappa_counterexample_gap_N64"] > 0.9,
        },
        {
            "gate": "negative controls",
            "threshold": "4/4 pass",
            "value": negative_control_passes,
            "pass": negative_control_passes == 4,
        },
        {
            "gate": "real benchmark rows",
            "threshold": "8 promoted rows",
            "value": len(benchmark_rows),
            "pass": len(benchmark_rows) == 8,
        },
        {
            "gate": "benchmark rollout pools",
            "threshold": ">=450 heldout pools",
            "value": benchmark_pools,
            "pass": benchmark_pools >= 450,
        },
        {
            "gate": "positive benchmark CI rows",
            "threshold": "8/8 positive lower bounds",
            "value": positive_ci_rows,
            "pass": positive_ci_rows == len(benchmark_rows),
        },
        {
            "gate": "RoboCasa coverage accounting",
            "threshold": ">=136 task IDs with any artifact",
            "value": robocasa_catalog["any_artifact_task_count"],
            "pass": robocasa_catalog["any_artifact_task_count"] >= 136,
        },
        {
            "gate": "artifact manifest breadth",
            "threshold": ">=462 scientific artifacts",
            "value": manifest["n_files"],
            "pass": manifest["n_files"] >= 462,
        },
    ]
    write_csv(OUT / "v4_protocol_gates.csv", protocol_rows, ["gate", "threshold", "value", "pass"])

    barh(
        FIG_OUT / "v4_exact_law_errors.pdf",
        [row["benchmark"] for row in benchmark_rows],
        [float(row["exact_mae"]) for row in benchmark_rows],
        "Exact-law utility MAE across rollout-pool artifacts",
        "utility MAE",
    )
    barh(
        FIG_OUT / "v4_benchmark_ci_lowers.pdf",
        [row["benchmark"] for row in benchmark_rows],
        [float(row["ci_lo"] or 0.0) for row in benchmark_rows],
        "Promoted scorer gain over random: CI lower bounds",
        "CI lower bound",
    )
    barh(
        FIG_OUT / "v4_failure_modes.pdf",
        [row["failure_mode"] for row in failure_rows],
        [float(row["value"]) for row in failure_rows],
        "Score-tail failure and mismatch magnitudes",
        "effect size",
    )
    barh(
        FIG_OUT / "v4_robocasa_coverage.pdf",
        [
            "registered",
            "rollout-pool",
            "micro-rollout",
            "any artifact",
        ],
        [
            robocasa_catalog["registry_count"],
            robocasa_catalog["verified_artifact_task_count"],
            robocasa_catalog["micro_rollout_task_count"],
            robocasa_catalog["any_artifact_task_count"],
        ],
        "RoboCasa coverage accounting",
        "task IDs",
    )
    barh(
        FIG_OUT / "v4_claim_artifact_counts.pdf",
        ["verified claims", "CSV", "JSON", "PNG", "NPZ"],
        [
            claims["num_verified"],
            manifest["counts_by_suffix"][".csv"],
            manifest["counts_by_suffix"][".json"],
            manifest["counts_by_suffix"][".png"],
            manifest["counts_by_suffix"][".npz"],
        ],
        "Claim and artifact coverage",
        "count",
    )
    barh(
        FIG_OUT / "v4_protocol_gates.pdf",
        [row["gate"] for row in protocol_rows],
        [1.0 if row["pass"] else 0.0 for row in protocol_rows],
        "Frozen protocol gates",
        "pass indicator",
    )

    for figure in FIG_OUT.glob("*.pdf"):
        shutil.copy2(figure, PAPER_FIG_OUT / figure.name)

    summary = {
        "claims_verified": claims["num_verified"],
        "claims_partial": claims["num_partial"],
        "claims_unsupported": claims["num_unsupported"],
        "artifact_files": manifest["n_files"],
        "artifact_total_mb": manifest["total_bytes"] / 1_000_000,
        "artifact_csv": manifest["counts_by_suffix"][".csv"],
        "artifact_json": manifest["counts_by_suffix"][".json"],
        "artifact_png": manifest["counts_by_suffix"][".png"],
        "artifact_npz": manifest["counts_by_suffix"][".npz"],
        "success_mae": exp1["mean_success_mc_mae"],
        "utility_mae": exp1["mean_utility_mc_mae"],
        "auc_identity_error": exp2["max_n2_identity_error"],
        "same_p_kappa_gap_n64": exp2["same_p_kappa_counterexample_gap_N64"],
        "severe_gap_growth": exp5["severe_gap_growth_minus_none"],
        "stuck_gap_growth": exp5["stuck_slip_gap_growth_minus_none"],
        "anti_scorer_n1": exp10["anti_scorer_mean_N1"],
        "anti_scorer_n64": exp10["anti_scorer_mean_N64"],
        "adaptive_gain": exp6["moment_law_improvement_over_uniform"],
        "fetch_exact_mae": fetch["exact_law_utility_mae"],
        "fetch_learned_ci_lo": ci(fetch, "learned_minus_random_N32"),
        "fetch_rgb_exact_mae": fetch_rgb["exact_law_utility_mae"],
        "fetch_rgb_ci_lo": ci(fetch_rgb, "visual_minus_random_N32"),
        "metaworld_exact_mae": metaworld["exact_law_utility_mae"],
        "robosuite_exact_mae": robosuite["exact_law_utility_mae"],
        "maniskill_exact_mae": maniskill["exact_law_utility_mae"],
        "robocasa97_tasks": len(robocasa97["env_ids"]),
        "robocasa97_pools": robocasa97["eval_rollout_pools"],
        "robocasa97_eval_samples": robocasa97["eval_samples"],
        "robocasa97_corr": robocasa97["model_metrics"]["utility_corr"],
        "robocasa97_ci_lo": ci(robocasa97, "promoted_learned_minus_random_N8"),
        "robocasa35_tasks": len(robocasa35["env_ids"]),
        "robocasa35_pools": robocasa35["eval_rollout_pools"],
        "robocasa35_corr": robocasa35["model_metrics"]["utility_corr"],
        "robocasa35_ci_lo": ci(robocasa35, "promoted_learned_minus_random_N4"),
        "robocasa_registered_tasks": robocasa_catalog["registry_count"],
        "robocasa_rollout_task_ids": robocasa_catalog["verified_artifact_task_count"],
        "robocasa_any_artifact_task_ids": robocasa_catalog["any_artifact_task_count"],
        "libero_pools": libero["eval_rollout_pools"],
        "libero_corr": libero["model_metrics"]["utility_corr"],
        "libero_ci_lo": ci(libero, "learned_energy_regularized_minus_random_N8"),
        "benchmark_rows": len(benchmark_rows),
        "benchmark_pools": benchmark_pools,
        "positive_ci_rows": positive_ci_rows,
        "max_exact_mae": max_exact_mae,
        "min_benchmark_ci_lo": min_benchmark_ci_lo,
        "negative_control_passes": negative_control_passes,
        "protocol_gate_rows": len(protocol_rows),
        "protocol_gate_passes": sum(1 for row in protocol_rows if row["pass"]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with MACROS.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by experiments/v4_frozen_evidence.py\n")
        handle.write(macro_line("VFourWAMClaimsVerified", summary["claims_verified"]))
        handle.write(macro_line("VFourWAMClaimsPartial", summary["claims_partial"]))
        handle.write(macro_line("VFourWAMClaimsUnsupported", summary["claims_unsupported"]))
        handle.write(macro_line("VFourWAMArtifactFiles", summary["artifact_files"]))
        handle.write(macro_line("VFourWAMArtifactTotalMB", summary["artifact_total_mb"], 1))
        handle.write(macro_line("VFourWAMArtifactCSV", summary["artifact_csv"]))
        handle.write(macro_line("VFourWAMArtifactJSON", summary["artifact_json"]))
        handle.write(macro_line("VFourWAMExactSuccessMAE", summary["success_mae"], 4))
        handle.write(macro_line("VFourWAMExactUtilityMAE", summary["utility_mae"], 4))
        handle.write(macro_line("VFourWAMAUCError", summary["auc_identity_error"], 1))
        handle.write(macro_line("VFourWAMSameAUCGap", summary["same_p_kappa_gap_n64"], 4))
        handle.write(macro_line("VFourWAMSevereGapGrowth", summary["severe_gap_growth"], 2))
        handle.write(macro_line("VFourWAMStuckGapGrowth", summary["stuck_gap_growth"], 2))
        handle.write(macro_line("VFourWAMAntiScorerNOne", summary["anti_scorer_n1"], 2))
        handle.write(macro_line("VFourWAMAntiScorerNSixtyFour", summary["anti_scorer_n64"], 2))
        handle.write(macro_line("VFourWAMAdaptiveGain", summary["adaptive_gain"], 4))
        handle.write(macro_line("VFourWAMFetchExactMAE", summary["fetch_exact_mae"], 4))
        handle.write(macro_line("VFourWAMFetchCILo", summary["fetch_learned_ci_lo"], 3))
        handle.write(macro_line("VFourWAMFetchRGBExactMAE", summary["fetch_rgb_exact_mae"], 4))
        handle.write(macro_line("VFourWAMFetchRGBCILo", summary["fetch_rgb_ci_lo"], 3))
        handle.write(macro_line("VFourWAMMetaWorldExactMAE", summary["metaworld_exact_mae"], 4))
        handle.write(macro_line("VFourWAMRoboSuiteExactMAE", summary["robosuite_exact_mae"], 4))
        handle.write(macro_line("VFourWAMManiSkillExactMAE", summary["maniskill_exact_mae"], 4))
        handle.write(macro_line("VFourWAMRoboCasaTasks", summary["robocasa97_tasks"]))
        handle.write(macro_line("VFourWAMRoboCasaPools", summary["robocasa97_pools"]))
        handle.write(macro_line("VFourWAMRoboCasaEvalSamples", summary["robocasa97_eval_samples"]))
        handle.write(macro_line("VFourWAMRoboCasaCorr", summary["robocasa97_corr"], 3))
        handle.write(macro_line("VFourWAMRoboCasaCILo", summary["robocasa97_ci_lo"], 3))
        handle.write(macro_line("VFourWAMRoboCasaResidualTasks", summary["robocasa35_tasks"]))
        handle.write(macro_line("VFourWAMRoboCasaResidualPools", summary["robocasa35_pools"]))
        handle.write(macro_line("VFourWAMRoboCasaResidualCorr", summary["robocasa35_corr"], 3))
        handle.write(macro_line("VFourWAMRoboCasaResidualCILo", summary["robocasa35_ci_lo"], 3))
        handle.write(macro_line("VFourWAMRoboCasaRegistered", summary["robocasa_registered_tasks"]))
        handle.write(macro_line("VFourWAMRoboCasaRolloutTaskIDs", summary["robocasa_rollout_task_ids"]))
        handle.write(macro_line("VFourWAMRoboCasaAnyTaskIDs", summary["robocasa_any_artifact_task_ids"]))
        handle.write(macro_line("VFourWAMLiberoPools", summary["libero_pools"]))
        handle.write(macro_line("VFourWAMLiberoCorr", summary["libero_corr"], 3))
        handle.write(macro_line("VFourWAMLiberoCILo", summary["libero_ci_lo"], 3))
        handle.write(macro_line("VFourWAMBenchmarkRows", summary["benchmark_rows"]))
        handle.write(macro_line("VFourWAMBenchmarkPools", summary["benchmark_pools"]))
        handle.write(macro_line("VFourWAMPositiveCIRows", summary["positive_ci_rows"]))
        handle.write(macro_line("VFourWAMMaxExactMAE", summary["max_exact_mae"], 4))
        handle.write(macro_line("VFourWAMMinBenchmarkCILo", summary["min_benchmark_ci_lo"], 3))
        handle.write(macro_line("VFourWAMNegativeControls", summary["negative_control_passes"]))
        handle.write(macro_line("VFourWAMProtocolGateRows", summary["protocol_gate_rows"]))
        handle.write(macro_line("VFourWAMProtocolGatePasses", summary["protocol_gate_passes"]))

    print(f"v4 cached evidence complete: {OUT}")
    print(f"claims={summary['claims_verified']} artifacts={summary['artifact_files']} benchmark_rows={len(benchmark_rows)}")


if __name__ == "__main__":
    main()
