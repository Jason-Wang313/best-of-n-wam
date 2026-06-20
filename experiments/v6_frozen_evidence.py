from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import v6_real_benchmark_evidence


RESULTS = ROOT / "results"
V6 = RESULTS / "v6"
OUT = RESULTS / "v6_frozen_evidence"
FIG_OUT = OUT / "figures"
PAPER_FIG_OUT = ROOT / "paper_figures" / "v6"
MACROS = ROOT / "v6_results_macros.tex"
PAPER_DIR = ROOT / "paper"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.{digits}f}\\%"


def macro_line(name: str, value: Any, digits: int = 3) -> str:
    rendered = fmt(value, digits) if isinstance(value, float) else str(value)
    return f"\\newcommand{{\\{name}}}{{{rendered}}}\n"


def copy_figure(path: Path) -> None:
    PAPER_FIG_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, PAPER_FIG_OUT / path.name)


def bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, metadata={"Creator": "experiments/v6_frozen_evidence.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    copy_figure(path)


def line_plot(path: Path, series: dict[str, list[tuple[float, float]]], title: str, ylabel: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, points in series.items():
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, metadata={"Creator": "experiments/v6_frozen_evidence.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    copy_figure(path)


def collect_summary(raw: dict[str, Any]) -> dict[str, Any]:
    audit = raw["real_benchmark_audit"]
    ablation = raw["selector_metric_ablation"]
    leakage = raw["leakage_protocol"]
    negative = raw["negative_controls"]
    theory = raw["finite_sample_theory"]
    calibration = raw["calibration_abstention"]
    return {
        "gate_passed": bool(raw["gate_passed"]),
        "family_count": audit["family_count"],
        "pool_count": audit["pool_count"],
        "decided_rate": audit["decided_rate"],
        "decision_accuracy": audit["decision_accuracy"],
        "false_allow_rate": audit["false_allow_rate"],
        "false_block_rate": audit["false_block_rate"],
        "request_label_rate": audit["decision_counts"].get("request_labels", 0) / max(1, audit["pool_count"]),
        "interval_coverage": audit["interval_coverage"],
        "audit_utility": audit["mean_audit_policy_utility"],
        "raw_high_n_utility": audit["mean_raw_high_n_utility"],
        "n1_utility": audit["mean_n1_utility"],
        "random_high_n_utility": audit["mean_random_high_n_utility"],
        "oracle_high_n_utility": audit["mean_oracle_high_n_utility"],
        "audit_minus_raw": audit["mean_audit_minus_raw_high_n"],
        "audit_rollout_units": audit["mean_audit_rollout_units"],
        "raw_rollout_units": audit["mean_raw_high_n_rollout_units"],
        "audit_label_units": audit["mean_audit_label_units"],
        "rollout_savings": ablation["audit_rollout_savings_vs_raw"],
        "audit_utility_per_rollout": ablation["audit_utility_per_rollout"],
        "raw_utility_per_rollout": ablation["raw_utility_per_rollout"],
        "prediction_hash_short": leakage["prediction_sha256"][:10],
        "split_hash_short": leakage["split_manifest_sha256"][:10],
        "negative_control_gap": negative["worst_nonoracle_gap_vs_raw"],
        "finite_labels_005_005": theory["epsilon_0_05_delta_0_05_labels"],
        "calibration_bin_count": calibration["bin_count"],
        "calibration_coverage": calibration["overall_interval_coverage"],
    }


def write_macros(summary: dict[str, Any]) -> None:
    with MACROS.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by experiments/v6_frozen_evidence.py\n")
        handle.write(macro_line("VSixWAMFamilies", summary["family_count"]))
        handle.write(macro_line("VSixWAMPools", summary["pool_count"]))
        handle.write(macro_line("VSixWAMDecidedRate", pct(summary["decided_rate"])))
        handle.write(macro_line("VSixWAMDecisionAccuracy", pct(summary["decision_accuracy"])))
        handle.write(macro_line("VSixWAMFalseAllow", pct(summary["false_allow_rate"])))
        handle.write(macro_line("VSixWAMFalseBlock", pct(summary["false_block_rate"])))
        handle.write(macro_line("VSixWAMRequestLabelRate", pct(summary["request_label_rate"])))
        handle.write(macro_line("VSixWAMIntervalCoverage", pct(summary["interval_coverage"])))
        handle.write(macro_line("VSixWAMAuditUtility", summary["audit_utility"], 3))
        handle.write(macro_line("VSixWAMRawHighNUtility", summary["raw_high_n_utility"], 3))
        handle.write(macro_line("VSixWAMNOneUtility", summary["n1_utility"], 3))
        handle.write(macro_line("VSixWAMOracleUtility", summary["oracle_high_n_utility"], 3))
        handle.write(macro_line("VSixWAMAuditMinusRaw", summary["audit_minus_raw"], 3))
        handle.write(macro_line("VSixWAMAuditRolloutUnits", summary["audit_rollout_units"], 1))
        handle.write(macro_line("VSixWAMRawRolloutUnits", summary["raw_rollout_units"], 1))
        handle.write(macro_line("VSixWAMRolloutSavings", summary["rollout_savings"], 1))
        handle.write(macro_line("VSixWAMAuditUtilityPerRollout", summary["audit_utility_per_rollout"], 3))
        handle.write(macro_line("VSixWAMRawUtilityPerRollout", summary["raw_utility_per_rollout"], 3))
        handle.write(macro_line("VSixWAMPredictionHash", summary["prediction_hash_short"]))
        handle.write(macro_line("VSixWAMSplitHash", summary["split_hash_short"]))
        handle.write(macro_line("VSixWAMNegativeControlGap", summary["negative_control_gap"], 3))
        handle.write(macro_line("VSixWAMFiniteLabelsEpsFiveDeltaFive", summary["finite_labels_005_005"]))


def write_summary_table(summary: dict[str, Any]) -> None:
    path = PAPER_DIR / "v6_summary_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = rf"""\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{p{{0.30\linewidth}}p{{0.26\linewidth}}p{{0.34\linewidth}}}}
\toprule
V6 check & Result & Claim discipline \\
\midrule
Real benchmark transfer & {summary['family_count']} families; {summary['pool_count']} pools & Existing simulated rollout-pool curves only. \\
Frozen decisions & accuracy {pct(summary['decision_accuracy'])}; decided {pct(summary['decided_rate'])} & Abstention/request-label is counted, not hidden. \\
Safety errors & false allow {pct(summary['false_allow_rate'])}; false block {pct(summary['false_block_rate'])} & Conservative audit; no real-robot claim. \\
Leakage protocol & split {summary['split_hash_short']}; predictions {summary['prediction_hash_short']} & Prediction hash is written before outcomes. \\
Compute accounting & {fmt(summary['rollout_savings'], 1)} fewer rollout units vs raw high-$N$ & Reports utility per rollout and label cost. \\
Finite-sample bound & {summary['finite_labels_005_005']} labels for $\epsilon=0.05,\delta=0.05$ & Explains why abstention is necessary. \\
\bottomrule
\end{{tabular}}
\caption{{V6 real-benchmark audit summary. V6 hardens the evidence on existing simulated benchmark rollout-pool curves while preserving the no-real-robot, no-GPU-scale, and no-broad-SOTA claim boundaries.}}
\label{{tab:v6-summary}}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")


def write_ablation_table(rows: list[dict[str, str]]) -> None:
    path = PAPER_DIR / "v6_ablation_table.tex"
    wanted = {
        "n1": "N=1",
        "raw_high_n": "raw high-N",
        "random_high_n": "random high-N",
        "audit_with_abstention": "audit+abstain",
        "pilot_sign_no_abstention": "pilot sign",
        "calibrated_sign_no_abstention": "calibrated sign",
        "oracle_upper_bound": "oracle",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Strategy & Utility & Rollout units & Utility / rollout \\\\",
        "\\midrule",
    ]
    for row in rows:
        strategy = row["strategy"]
        if strategy not in wanted:
            continue
        lines.append(
            f"{wanted[strategy]} & {fmt(row['mean_utility'], 3)} & "
            f"{fmt(row['mean_rollout_units'], 1)} & {fmt(row['utility_per_rollout_unit'], 3)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{V6 selector and compute ablation on existing real-benchmark rollout-pool curves. Oracle is diagnostic, not deployable.}",
            "\\label{tab:v6-ablation}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(raw: dict[str, Any]) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    family_rows = raw["cross_family_transfer"]["family_rows"]
    bar(
        FIG_OUT / "v6_cross_family_accuracy.pdf",
        [row["family"].replace("robocasa_", "rc_") for row in family_rows],
        [float(row["decision_accuracy"] or 0.0) for row in family_rows],
        "V6 leave-one-family-out decision accuracy",
        "accuracy on decided pools",
    )
    bar(
        FIG_OUT / "v6_cross_family_decided_rate.pdf",
        [row["family"].replace("robocasa_", "rc_") for row in family_rows],
        [float(row["decided_rate"]) for row in family_rows],
        "V6 abstention by heldout family",
        "decided rate",
    )
    ablation_rows = raw["selector_metric_ablation"]["rows"]
    selected = [row for row in ablation_rows if row["strategy"] in {"raw_high_n", "audit_with_abstention", "pilot_sign_no_abstention", "oracle_upper_bound"}]
    bar(
        FIG_OUT / "v6_compute_ablation.pdf",
        [row["strategy"].replace("_", " ") for row in selected],
        [float(row["utility_per_rollout_unit"]) for row in selected],
        "V6 utility per rollout unit",
        "utility / rollout",
    )
    robustness = read_csv(V6 / "robustness_grid.csv")
    series: dict[str, list[tuple[float, float]]] = {}
    for row in robustness:
        label = f"epsilon={row['epsilon']}"
        series.setdefault(label, []).append((float(row["radius_quantile"]), float(row["decision_accuracy"] or 0.0)))
    line_plot(
        FIG_OUT / "v6_robustness_grid.pdf",
        series,
        "V6 robustness over confidence radius",
        "decision accuracy",
        "radius quantile",
    )


def run(*, smoke: bool = False, output_root: Path = ROOT, source_root: Path | None = None) -> dict[str, Any]:
    global ROOT, RESULTS, V6, OUT, FIG_OUT, PAPER_FIG_OUT, MACROS, PAPER_DIR
    ROOT = output_root
    RESULTS = ROOT / "results"
    V6 = RESULTS / ("v6_smoke" if smoke else "v6")
    OUT = RESULTS / ("v6_frozen_evidence_smoke" if smoke else "v6_frozen_evidence")
    FIG_OUT = OUT / "figures"
    PAPER_FIG_OUT = ROOT / "paper_figures" / "v6"
    MACROS = ROOT / "v6_results_macros.tex"
    PAPER_DIR = ROOT / "paper"

    raw = v6_real_benchmark_evidence.run(output_root=output_root, source_root=source_root or output_root, smoke=smoke)
    summary = collect_summary(raw)
    write_macros(summary)
    write_summary_table(summary)
    ablation_rows = read_csv(V6 / "selector_metric_ablation.csv")
    write_ablation_table(ablation_rows)
    write_figures(raw)
    write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run(smoke=args.smoke)
    print(
        f"v6 frozen evidence complete ({'smoke' if args.smoke else 'canonical'}): "
        f"families={summary['family_count']} pools={summary['pool_count']} "
        f"accuracy={pct(summary['decision_accuracy'])} gate={summary['gate_passed']}"
    )


if __name__ == "__main__":
    main()
