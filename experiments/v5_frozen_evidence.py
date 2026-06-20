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

from experiments import v5_core_evidence, v5_prospective_evidence


RESULTS = ROOT / "results"
V5 = RESULTS / "v5"
OUT = RESULTS / "v5_frozen_evidence"
FIG_OUT = OUT / "figures"
PAPER_FIG_OUT = ROOT / "paper_figures" / "v5"
MACROS = ROOT / "v5_results_macros.tex"
PAPER_DIR = ROOT / "paper"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def macro_line(name: str, value: str | int | float, digits: int = 3) -> str:
    if isinstance(value, float):
        rendered = fmt(value, digits)
    else:
        rendered = str(value)
    return f"\\newcommand{{\\{name}}}{{{rendered}}}\n"


def copy_figure(path: Path) -> None:
    PAPER_FIG_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, PAPER_FIG_OUT / path.name)


def bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(labels, values, color="#2c7fb8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=24)
    fig.tight_layout()
    fig.savefig(path, metadata={"Creator": "experiments/v5_frozen_evidence.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    copy_figure(path)


def line_plot(path: Path, series: dict[str, list[tuple[float, float]]], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, points in series.items():
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("budget / labels")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, metadata={"Creator": "experiments/v5_frozen_evidence.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    copy_figure(path)


def policy_label(policy: str) -> str:
    return {
        "n1": "N=1",
        "raw_high_n": "raw high-N",
        "random_high_n": "random high-N",
        "audit_policy": "audit policy",
        "oracle_upper_bound": "oracle",
    }.get(policy, policy)


def write_summary_table(summary: dict[str, Any]) -> None:
    path = PAPER_DIR / "v5_summary_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = rf"""\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{p{{0.32\linewidth}}p{{0.22\linewidth}}p{{0.34\linewidth}}}}
\toprule
V5 check & Result & Claim discipline \\
\midrule
Exact-law edge cases & {summary['exact_case_count']} cases; max error {fmt(summary['exact_max_error'], 1)} & Finite tied-pool law remains source of truth. \\
AUC/correlation insufficiency & matched-summary high-$N$ gap {fmt(summary['auc_high_n_gap'], 3)} & AUC and correlation are not promoted as high-$N$ guarantees. \\
Finite-pool census & {summary['census_rows']} rows; {summary['census_nonmonotonic']} nonmonotonic & CPU enumeration gives complete coverage for the stated finite-pool class. \\
Blind prospective audit & {summary['prospective_pools']} heldout pools; accuracy {pct(summary['prospective_accuracy'])} & Predictions are hash-locked before outcomes. \\
Closed-loop validation & audit {fmt(summary['closed_loop_audit_return'], 3)} vs raw {fmt(summary['closed_loop_raw_return'], 3)} & Toy execution check, not robot success. \\
ScoreTailBench core & {summary['scoretailbench_pools']} pools & Reusable tiny artifact, not a leaderboard. \\
\bottomrule
\end{{tabular}}
\caption{{V5 frozen evidence summary. V5 adds prospective, exhaustive, and reproducibility checks while preserving the paper's no-real-robot and no-SOTA claim boundaries.}}
\label{{tab:v5-summary}}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")


def write_compute_table(summary: dict[str, Any]) -> None:
    path = PAPER_DIR / "v5_compute_frontier_table.tex"
    rows = summary["equal_compute_rows"]
    wanted = [row for row in rows if row["budget"] in {16, 64} and row["strategy"] in {"blind_more_rollouts", "calibrated_lcb", "audit_fewer_rollouts"}]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Budget & Strategy & Utility & CPU units \\\\",
        "\\midrule",
    ]
    label_map = {
        "blind_more_rollouts": "blind more rollouts",
        "calibrated_lcb": "calibrated LCB",
        "audit_fewer_rollouts": "audit fewer rollouts",
    }
    for row in wanted:
        lines.append(
            f"{row['budget']} & {label_map[row['strategy']]} & "
            f"{fmt(row['mean_selected_utility'], 3)} & {fmt(row['mean_cpu_units'], 1)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Equal-compute V5 frontier. CPU units count rollout evaluations plus label cost in the deterministic V5 audit harness; oracle rows are omitted from this deployable comparison.}",
            "\\label{tab:v5-compute-frontier}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_summary(core: dict[str, Any], prospective: dict[str, Any]) -> dict[str, Any]:
    closed = prospective["closed_loop_validation"]
    returns = closed["mean_return_by_policy"]
    budget_rows = prospective["equal_compute_frontier"]["summary_rows"]
    budget64 = {row["strategy"]: row for row in budget_rows if row["budget"] == 64}
    label_rows = prospective["label_budget_sample_complexity"]["budget_summaries"]
    census_counts = core["finite_pool_census"]["classification_counts"]
    pred_hash = prospective["prospective_audit"]["prediction_sha256"]
    return {
        "exact_case_count": core["exact_law_hardening"]["case_count"],
        "exact_max_error": core["exact_law_hardening"]["max_abs_error"],
        "auc_high_n_gap": core["auc_correlation_insufficiency"]["high_n_gap"],
        "census_rows": core["finite_pool_census"]["row_count"],
        "census_helps": census_counts.get("helps", 0),
        "census_harms": census_counts.get("harms", 0),
        "census_nonmonotonic": census_counts.get("nonmonotonic", 0),
        "impossibility_gap": core["impossibility_boundary"]["selected_real_utility_gap"],
        "scoretailbench_pools": core["scoretailbench"]["pool_count"],
        "prospective_pools": prospective["prospective_audit"]["heldout_pools"],
        "prospective_accuracy": prospective["prospective_audit"]["decision_accuracy"],
        "prospective_false_allow": prospective["prospective_audit"]["false_allow_rate"],
        "prospective_false_block": prospective["prospective_audit"]["false_block_rate"],
        "prospective_prediction_hash": pred_hash,
        "prospective_prediction_hash_short": pred_hash[:10],
        "label_budget_rows": label_rows,
        "selector_count": len(prospective["selector_gauntlet"]["selectors"]),
        "equal_compute_rows": budget_rows,
        "budget64_blind_utility": budget64["blind_more_rollouts"]["mean_selected_utility"],
        "budget64_calibrated_utility": budget64["calibrated_lcb"]["mean_selected_utility"],
        "budget64_audit_utility": budget64["audit_fewer_rollouts"]["mean_selected_utility"],
        "closed_loop_episodes": closed["episodes"],
        "closed_loop_audit_return": returns["audit_policy"],
        "closed_loop_raw_return": returns["raw_high_n"],
        "closed_loop_n1_return": returns["n1"],
        "closed_loop_oracle_return": returns["oracle_upper_bound"],
        "audit_minus_raw_ci_lo": closed["audit_minus_raw_ci"]["lo"],
        "audit_minus_raw_ci_mean": closed["audit_minus_raw_ci"]["mean"],
        "gate_passed": bool(core["gate_passed"] and prospective["gate_passed"]),
    }


def write_macros(summary: dict[str, Any]) -> None:
    with MACROS.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by experiments/v5_frozen_evidence.py\n")
        handle.write(macro_line("VFiveWAMExactHardeningCases", summary["exact_case_count"]))
        handle.write(macro_line("VFiveWAMExactMaxError", summary["exact_max_error"], 1))
        handle.write(macro_line("VFiveWAMAucHighNGap", summary["auc_high_n_gap"], 3))
        handle.write(macro_line("VFiveWAMCensusRows", summary["census_rows"]))
        handle.write(macro_line("VFiveWAMCensusHelps", summary["census_helps"]))
        handle.write(macro_line("VFiveWAMCensusHarms", summary["census_harms"]))
        handle.write(macro_line("VFiveWAMCensusNonmonotonic", summary["census_nonmonotonic"]))
        handle.write(macro_line("VFiveWAMImpossibilityGap", summary["impossibility_gap"], 3))
        handle.write(macro_line("VFiveWAMScoreTailBenchPools", summary["scoretailbench_pools"]))
        handle.write(macro_line("VFiveWAMProspectivePools", summary["prospective_pools"]))
        handle.write(macro_line("VFiveWAMProspectiveAccuracy", pct(summary["prospective_accuracy"])))
        handle.write(macro_line("VFiveWAMProspectiveFalseAllow", pct(summary["prospective_false_allow"])))
        handle.write(macro_line("VFiveWAMProspectiveFalseBlock", pct(summary["prospective_false_block"])))
        handle.write(macro_line("VFiveWAMPredictionHash", summary["prospective_prediction_hash_short"]))
        handle.write(macro_line("VFiveWAMSelectorCount", summary["selector_count"]))
        handle.write(macro_line("VFiveWAMBudgetSixtyFourBlind", summary["budget64_blind_utility"], 3))
        handle.write(macro_line("VFiveWAMBudgetSixtyFourCalibrated", summary["budget64_calibrated_utility"], 3))
        handle.write(macro_line("VFiveWAMBudgetSixtyFourAudit", summary["budget64_audit_utility"], 3))
        handle.write(macro_line("VFiveWAMClosedLoopEpisodes", summary["closed_loop_episodes"]))
        handle.write(macro_line("VFiveWAMClosedLoopAuditReturn", summary["closed_loop_audit_return"], 3))
        handle.write(macro_line("VFiveWAMClosedLoopRawReturn", summary["closed_loop_raw_return"], 3))
        handle.write(macro_line("VFiveWAMClosedLoopNOneReturn", summary["closed_loop_n1_return"], 3))
        handle.write(macro_line("VFiveWAMClosedLoopOracleReturn", summary["closed_loop_oracle_return"], 3))
        handle.write(macro_line("VFiveWAMAuditMinusRawCILo", summary["audit_minus_raw_ci_lo"], 3))


def write_figures(summary: dict[str, Any]) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    census_counts = {
        "helps": summary["census_helps"],
        "harms": summary["census_harms"],
        "nonmonotonic": summary["census_nonmonotonic"],
    }
    bar(
        FIG_OUT / "v5_census_regimes.pdf",
        list(census_counts),
        [float(v) for v in census_counts.values()],
        "V5 exhaustive finite-pool census",
        "pool configurations",
    )
    bar(
        FIG_OUT / "v5_closed_loop_returns.pdf",
        [policy_label(k) for k in ["n1", "raw_high_n", "random_high_n", "audit_policy", "oracle_upper_bound"]],
        [
            summary["closed_loop_n1_return"],
            summary["closed_loop_raw_return"],
            0.0,  # replaced below from frozen JSON if present
            summary["closed_loop_audit_return"],
            summary["closed_loop_oracle_return"],
        ],
        "V5 toy closed-loop validation",
        "mean return",
    )
    # Rewrite closed-loop figure with the random policy value included.
    prospective = load_json(V5 / "prospective_evidence_summary.json")
    returns = prospective["closed_loop_validation"]["mean_return_by_policy"]
    bar(
        FIG_OUT / "v5_closed_loop_returns.pdf",
        [policy_label(k) for k in returns.keys()],
        [float(v) for v in returns.values()],
        "V5 toy closed-loop validation",
        "mean return",
    )
    label_series = {}
    acc_points = []
    request_points = []
    for row in summary["label_budget_rows"]:
        budget = float(row["label_budget"])
        acc = row["decision_accuracy"]
        acc_points.append((budget, 0.0 if acc is None else float(acc)))
        request_count = float(row["decision_counts"].get("request_labels", 0))
        total = max(1.0, sum(float(v) for v in row["decision_counts"].values()))
        request_points.append((budget, request_count / total))
    label_series["decision accuracy"] = acc_points
    label_series["request-label rate"] = request_points
    line_plot(
        FIG_OUT / "v5_label_budget.pdf",
        label_series,
        "V5 label-budget sample complexity",
        "rate",
    )
    frontier_series: dict[str, list[tuple[float, float]]] = {}
    for row in summary["equal_compute_rows"]:
        if row["strategy"] not in {"blind_more_rollouts", "calibrated_lcb", "audit_fewer_rollouts"}:
            continue
        frontier_series.setdefault(row["strategy"].replace("_", " "), []).append(
            (float(row["budget"]), float(row["mean_selected_utility"]))
        )
    line_plot(
        FIG_OUT / "v5_equal_compute_frontier.pdf",
        frontier_series,
        "V5 equal-compute CPU frontier",
        "mean selected utility",
    )


def run(*, smoke: bool = False, output_root: Path = ROOT) -> dict[str, Any]:
    global ROOT, RESULTS, V5, OUT, FIG_OUT, PAPER_FIG_OUT, MACROS, PAPER_DIR
    ROOT = output_root
    RESULTS = ROOT / "results"
    V5 = RESULTS / ("v5_smoke" if smoke else "v5")
    OUT = RESULTS / ("v5_frozen_evidence_smoke" if smoke else "v5_frozen_evidence")
    FIG_OUT = OUT / "figures"
    PAPER_FIG_OUT = ROOT / "paper_figures" / "v5"
    MACROS = ROOT / "v5_results_macros.tex"
    PAPER_DIR = ROOT / "paper"

    core = v5_core_evidence.run(output_root, smoke=smoke)
    prospective = v5_prospective_evidence.run(smoke=smoke, output_root=output_root)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = collect_summary(core, prospective)
    write_macros(summary)
    write_summary_table(summary)
    write_compute_table(summary)
    write_figures(summary)
    write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summary = run(smoke=args.smoke)
    print(
        f"v5 frozen evidence complete ({'smoke' if args.smoke else 'canonical'}): "
        f"prospective_pools={summary['prospective_pools']} "
        f"accuracy={pct(summary['prospective_accuracy'])} "
        f"gate={summary['gate_passed']}"
    )


if __name__ == "__main__":
    main()
