from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wam_inference_value.benchmarks.registry import benchmark_statuses
from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json


def run() -> dict:
    ensure_result_dirs()
    statuses = [s.__dict__ for s in benchmark_statuses()]
    robocasa_smoke = {}
    smoke_path = results_dir() / "benchmark_robocasa_smoke.json"
    robocasa_learned = {}
    learned_path = results_dir() / "benchmark_robocasa_learned_wam.json"
    if smoke_path.exists():
        import json

        robocasa_smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if learned_path.exists():
        import json

        robocasa_learned = json.loads(learned_path.read_text(encoding="utf-8"))
    any_available = any(s["available"] for s in statuses)
    summary = {
        "experiment": "benchmark_smoke",
        "attempted": True,
        "any_available": any_available,
        "statuses": statuses,
        "verified_benchmark_claims": [],
    }
    write_json(results_dir() / "benchmark_smoke.json", summary)
    pd.DataFrame(statuses).to_csv(results_dir() / "tables" / "benchmark_status.csv", index=False)
    report = [
        "# Benchmark Blocker Report",
        "",
        "External benchmark integration was attempted.",
        "",
        "## Status",
    ]
    for item in statuses:
        report.append(f"- {item['name']}: available={item['available']} reason={item['reason']}")
    if any_available:
        remaining = [item["name"] for item in statuses if not item["available"]]
        report.extend(
            [
                "",
                "## Current Outcome",
                "",
                "At least one optional external benchmark path is available. Run `bash scripts/run_benchmark_full.sh` to generate benchmark artifacts.",
                "",
                "## Remaining Blockers",
                "",
                ("Remaining unavailable adapters: " + ", ".join(remaining) + ".") if remaining else "No registered benchmark adapter is currently blocked.",
            ]
        )
        if robocasa_smoke.get("verified"):
            report.extend(
                [
                    "",
                    "## Separate RoboCasa Smoke Artifact",
                    "",
                    f"RoboCasa is unavailable in the active Python environment but has a verified external smoke artifact: `{robocasa_smoke.get('env_id')}`, `{robocasa_smoke.get('n_rollouts_total')}` rollouts, exact-law utility MAE `{robocasa_smoke.get('exact_law_utility_mae')}`.",
                    "This is single task only, not full multi-task RoboCasa validation.",
                ]
            )
        if robocasa_learned.get("verified"):
            metrics = robocasa_learned.get("model_metrics") or {}
            ci = (robocasa_learned.get("confidence_intervals") or {}).get("learned_minus_random_N8") or {}
            report.extend(
                [
                    "",
                    "## Separate RoboCasa Learned-WAM Artifact",
                    "",
                    f"A lightweight ridge state/action-sequence WAM-lite was trained on `{robocasa_learned.get('train_samples')}` single-task RoboCasa rollouts and evaluated on `{robocasa_learned.get('eval_samples')}` heldout rollouts.",
                    f"Validation utility correlation is `{metrics.get('utility_corr')}`; learned-minus-random N8 CI lower bound is `{ci.get('lo')}`.",
                    "This supports only a single-task contact-rich sanity check, not a multi-task RoboCasa benchmark.",
                ]
            )
    else:
        report.extend(
            [
                "",
                "## What Was Attempted",
                "",
                "- `experiments/benchmark_smoke.py` checked import availability for the registered optional benchmark adapters.",
                "- ManiSkill was not importable as `mani_skill` or `mani_skill2` in the active Python environment.",
                "- LIBERO, RoboCasa, and Gym manipulation adapters reported unavailable.",
                "",
                "## Next Commands To Try In A Benchmark Environment",
                "",
                "```bash",
                "python -m pip install mani_skill",
                "python -m pip install \"gymnasium[mujoco]\"",
                "bash scripts/run_benchmark_smoke.sh",
                "bash scripts/run_benchmark_full.sh",
                "```",
                "",
                "## Claim Policy",
                "",
                "No benchmark validation claims are verified. README and paper claims must treat benchmark support as optional/future until artifacts exist.",
            ]
        )
    (ROOT / "reports" / "benchmark_blocker_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    print(f"benchmark smoke attempted: any_available={summary['any_available']}")


if __name__ == "__main__":
    main()
