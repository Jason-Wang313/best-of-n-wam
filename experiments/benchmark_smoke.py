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
    robocasa_multitask = {}
    multitask_path = results_dir() / "benchmark_robocasa_multitask_wam.json"
    robocasa_broad = {}
    broad_path = results_dir() / "benchmark_robocasa_broad_wam.json"
    robocasa_family12 = {}
    family12_path = results_dir() / "benchmark_robocasa_family12_wam.json"
    libero_wam = {}
    libero_path = results_dir() / "benchmark_libero_wam.json"
    libero_scripted = {}
    libero_scripted_path = results_dir() / "benchmark_libero_scripted_policy.json"
    if smoke_path.exists():
        import json

        robocasa_smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if learned_path.exists():
        import json

        robocasa_learned = json.loads(learned_path.read_text(encoding="utf-8"))
    if multitask_path.exists():
        import json

        robocasa_multitask = json.loads(multitask_path.read_text(encoding="utf-8"))
    if broad_path.exists():
        import json

        robocasa_broad = json.loads(broad_path.read_text(encoding="utf-8"))
    if family12_path.exists():
        import json

        robocasa_family12 = json.loads(family12_path.read_text(encoding="utf-8"))
    if libero_path.exists():
        import json

        libero_wam = json.loads(libero_path.read_text(encoding="utf-8"))
    if libero_scripted_path.exists():
        import json

        libero_scripted = json.loads(libero_scripted_path.read_text(encoding="utf-8"))
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
        if robocasa_multitask.get("verified"):
            metrics = robocasa_multitask.get("model_metrics") or {}
            ci = (robocasa_multitask.get("confidence_intervals") or {}).get("best_learned_minus_random_N8") or {}
            report.extend(
                [
                    "",
                    "## Separate RoboCasa Three-Task Learned-WAM Artifact",
                    "",
                    f"A task conditioned ridge state/action-sequence WAM-lite was trained across `{len(robocasa_multitask.get('env_ids') or [])}` RoboCasa task IDs with `{robocasa_multitask.get('train_samples')}` train rollouts and `{robocasa_multitask.get('eval_samples')}` heldout eval rollouts.",
                    f"Validation utility correlation is `{metrics.get('utility_corr')}`; promoted scorer `{robocasa_multitask.get('promoted_scorer')}` has learned-minus-random N8 CI lower bound `{ci.get('lo')}`.",
                    "This supports a three-task RoboCasa pick-place family artifact, not full RoboCasa-wide validation.",
                ]
            )
        if robocasa_broad.get("verified"):
            metrics = robocasa_broad.get("model_metrics") or {}
            ci = (robocasa_broad.get("confidence_intervals") or {}).get("best_learned_minus_random_N8") or {}
            report.extend(
                [
                    "",
                    "## Separate RoboCasa Broad Task Family Learned-WAM Artifact",
                    "",
                    f"A task conditioned ridge state/action-sequence WAM-lite was trained across `{len(robocasa_broad.get('env_ids') or [])}` non-pick-place RoboCasa task IDs with `{robocasa_broad.get('train_samples')}` train rollouts and `{robocasa_broad.get('eval_samples')}` heldout eval rollouts.",
                    f"Validation utility correlation is `{metrics.get('utility_corr')}`; promoted scorer `{robocasa_broad.get('promoted_scorer')}` has learned-minus-random N8 CI lower bound `{ci.get('lo')}`.",
                    "This supports broad RoboCasa rollout-pool dense-utility validation across atomic kitchen manipulation tasks, not full RoboCasa-wide validation or solved-policy performance.",
                ]
            )
        if robocasa_family12.get("verified"):
            metrics = robocasa_family12.get("model_metrics") or {}
            ci = (robocasa_family12.get("confidence_intervals") or {}).get("best_learned_minus_random_N8") or {}
            report.extend(
                [
                    "",
                    "## Separate RoboCasa 12-Task Family Learned-WAM Artifact",
                    "",
                    f"A task conditioned ridge state/action-sequence WAM-lite was trained across `{len(robocasa_family12.get('env_ids') or [])}` RoboCasa open/close/turn task IDs with `{robocasa_family12.get('train_samples')}` train rollouts and `{robocasa_family12.get('eval_samples')}` heldout eval rollouts.",
                    f"Validation utility correlation is `{metrics.get('utility_corr')}`; promoted scorer `{robocasa_family12.get('promoted_scorer')}` has learned-minus-random N8 CI lower bound `{ci.get('lo')}`.",
                    "This supports a wider RoboCasa task family rollout-pool dense-utility artifact, not full RoboCasa-wide validation or solved-policy performance.",
                ]
            )
        if libero_wam.get("verified"):
            metrics = libero_wam.get("model_metrics") or {}
            ci = (libero_wam.get("confidence_intervals") or {}).get("best_learned_minus_random_N8") or {}
            report.extend(
                [
                    "",
                    "## Separate LIBERO Three-Task Learned-WAM Artifact",
                    "",
                    f"A ridge state/action-sequence WAM-lite was trained across `{len(libero_wam.get('tasks') or [])}` LIBERO Spatial tasks with `{libero_wam.get('train_samples')}` train rollout samples and `{libero_wam.get('eval_samples')}` heldout eval rollout samples.",
                    f"Validation utility correlation is `{metrics.get('utility_corr')}`; promoted scorer `{libero_wam.get('promoted_scorer')}` has learned-minus-random N8 CI lower bound `{ci.get('lo')}`.",
                    "This supports LIBERO rollout-pool dense-utility validation, not solved-task LIBERO policy performance.",
                ]
            )
        if libero_scripted.get("verified"):
            ci = (libero_scripted.get("confidence_intervals") or {}).get("success_rate") or {}
            report.extend(
                [
                    "",
                    "## Separate LIBERO Object Sparse-Success Scripted Smoke",
                    "",
                    f"A hand scripted OSC pick-place controller was evaluated on `{libero_scripted.get('n_episodes')}` LIBERO Object episodes across `{libero_scripted.get('n_tasks')}` tasks and `{libero_scripted.get('n_seeds')}` seeds.",
                    f"It achieved `{libero_scripted.get('n_successes')}` sparse successes; success-rate bootstrap CI is [`{ci.get('lo')}`, `{ci.get('hi')}`].",
                    "This supports a narrow sparse-success simulator smoke, not learned policy performance or full LIBERO validation.",
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
