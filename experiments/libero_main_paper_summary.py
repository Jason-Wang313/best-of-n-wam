from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "libero_full_suite_serial" / "summary.json"
SEED_METRICS = ROOT / "results" / "libero_full_suite_serial" / "benchmark_libero_full_suite_serial_seed_metrics.csv"
OUT_CSV = ROOT / "results" / "libero_full_suite_serial" / "libero_main_suite_summary.csv"
OUT_JSON = ROOT / "results" / "libero_full_suite_serial" / "libero_main_suite_summary.json"
PAPER_TABLE = ROOT / "paper" / "libero_main_suite_table.tex"

SUITE_LABELS = {
    "__all__": "All configured",
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}
SUITE_ORDER = ["__all__", "libero_spatial", "libero_object", "libero_goal", "libero_10"]
PROMOTED_SCORER_COLUMNS = {
    "learned_wam": "learned_wam",
    "learned_physics_score": "learned_physics",
    "learned_energy_regularized": "learned_energy_regularized",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def suite_name(task_key: str) -> str:
    return str(task_key).split("/", 1)[0]


def ci95(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "stderr": None, "ci95": None, "lo": None, "hi": None}
    mean = sum(clean) / len(clean)
    if len(clean) > 1:
        var = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
        std = math.sqrt(var)
        stderr = std / math.sqrt(len(clean))
    else:
        std = 0.0
        stderr = 0.0
    half_width = 1.96 * stderr
    return {
        "n": len(clean),
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ci95": half_width,
        "lo": mean - half_width,
        "hi": mean + half_width,
    }


def fmt3(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def build_suite_rows(summary: dict[str, Any], seed_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    max_n = int(summary.get("max_n") or 8)
    promoted_scorer = str(summary.get("promoted_scorer") or "best_learned")
    metric_stem = PROMOTED_SCORER_COLUMNS.get(promoted_scorer, "best_learned")
    metric_key = f"{metric_stem}_minus_random_N{max_n}"
    rows: list[dict[str, Any]] = []

    for suite in SUITE_ORDER:
        if suite == "__all__":
            suite_seed_rows = seed_rows
        else:
            suite_seed_rows = [row for row in seed_rows if suite_name(row["task_key"]) == suite]
        values = [float(row[metric_key]) for row in suite_seed_rows]
        ci = ci95(values)
        task_values = []
        for task_key in sorted({str(row["task_key"]) for row in suite_seed_rows}):
            task_metric_values = [float(row[metric_key]) for row in suite_seed_rows if str(row["task_key"]) == task_key]
            task_values.append((task_key, sum(task_metric_values) / len(task_metric_values)))
        worst_task, worst_value = min(task_values, key=lambda item: item[1]) if task_values else ("", None)
        rows.append(
            {
                "suite": suite,
                "label": SUITE_LABELS[suite],
                "promoted_scorer": promoted_scorer,
                "tasks": len({row["task_key"] for row in suite_seed_rows}),
                "rollout_pools": len(suite_seed_rows),
                f"mean_{metric_key}": ci["mean"],
                f"ci_lower_{metric_key}": ci["lo"],
                f"ci_upper_{metric_key}": ci["hi"],
                "worst_task": worst_task,
                f"worst_task_mean_{metric_key}": worst_value,
            }
        )
    return rows


def write_suite_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_paper_table(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    max_n = int(summary.get("max_n") or 8)
    promoted_scorer = str(summary.get("promoted_scorer") or "best_learned")
    metric_stem = PROMOTED_SCORER_COLUMNS.get(promoted_scorer, "best_learned")
    metric_key = f"{metric_stem}_minus_random_N{max_n}"
    mean_key = f"mean_{metric_key}"
    lower_key = f"ci_lower_{metric_key}"
    worst_key = f"worst_task_mean_{metric_key}"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        rf"LIBERO slice & Tasks & Pools & Mean $\Delta_{{{max_n}}}$ & CI lower / worst \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['tasks']} & {row['rollout_pools']} & "
            f"{fmt3(row[mean_key])} & {fmt3(row[lower_key])} / {fmt3(row[worst_key])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{\textbf{Main-paper LIBERO full-suite signal.} "
                rf"The completed serial run covers {int(summary.get('completed_task_count') or 0)}/"
                rf"{int(summary.get('task_count') or 0)} configured LIBERO tasks and "
                rf"{int(summary.get('eval_rollout_pools') or 0)} state-level rollout pools. "
                rf"$\Delta_{{{max_n}}}$ is the promoted learned scorer minus random selected real utility at $N={max_n}$; "
                r"the last column reports the suite CI lower bound and the worst task mean. "
                r"This is CPU rollout-pool evidence, not real-robot validation, not VLA-scale SOTA, "
                r"and not solved-policy success.}"
            ),
            r"\label{tab:libero-main-suite}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(summary_path: Path, seed_metrics_path: Path, out_csv: Path, out_json: Path, paper_table: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    seed_rows = read_csv_rows(seed_metrics_path)
    rows = build_suite_rows(summary, seed_rows)
    payload = {
        "experiment": "libero_main_paper_summary",
        "source_summary": str(summary_path),
        "source_seed_metrics": str(seed_metrics_path),
        "complete": bool(summary.get("complete")),
        "verified": bool(summary.get("verified")),
        "claim_boundaries": summary.get("claim_boundaries") or {},
        "suite_rows": rows,
    }
    write_suite_csv(out_csv, rows)
    write_json(out_json, payload)
    write_paper_table(paper_table, rows, summary)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive compact main-paper LIBERO suite summaries from completed serial artifacts.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-metrics", type=Path, default=SEED_METRICS)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--paper-table", type=Path, default=PAPER_TABLE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = generate(args.summary, args.seed_metrics, args.out_csv, args.out_json, args.paper_table)
    all_row = next(row for row in payload["suite_rows"] if row["suite"] == "__all__")
    print(
        "LIBERO main summary: "
        f"complete={payload['complete']} verified={payload['verified']} "
        f"tasks={all_row['tasks']} pools={all_row['rollout_pools']}"
    )


if __name__ == "__main__":
    main()
