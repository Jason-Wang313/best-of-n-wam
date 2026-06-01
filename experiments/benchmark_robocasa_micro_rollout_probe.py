from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from wam_inference_value.benchmarks.robocasa_adapter import (
    RoboCasaAdapter,
    RoboCasaUnavailableError,
    is_robocasa_available,
)
from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json


DEFAULT_ENV_IDS = [
    "robocasa/PickPlaceCounterToStandMixer",
    "robocasa/PickPlaceCounterToToasterOven",
    "robocasa/PickPlaceDrawerToCounter",
    "robocasa/PickPlaceMicrowaveToCounter",
]


def _paths(output_tag: str) -> dict[str, Path]:
    prefix = f"benchmark_robocasa_micro_rollout_{output_tag}"
    return {
        "summary": results_dir() / f"{prefix}.json",
        "table": results_dir() / "tables" / f"{prefix}.csv",
        "report": ROOT / "reports" / f"robocasa_micro_rollout_{output_tag}_report.md",
    }


def _write_report(summary: dict[str, Any]) -> None:
    report_path = Path(summary.get("report_path") or _paths(str(summary.get("output_tag") or "extra"))["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary.get("available"):
        lines = [
            "# RoboCasa Micro-Rollout Probe",
            "",
            "- status: `unavailable`",
            f"- reason: `{summary.get('reason')}`",
        ]
    else:
        lines = [
            "# RoboCasa Micro-Rollout Probe",
            "",
            f"- status: `{'verified' if summary.get('verified') else 'attempted_not_promoted'}`",
            f"- candidate task IDs: `{summary.get('candidate_task_count')}`",
            f"- runnable task IDs: `{summary.get('runnable_task_count')}`",
            f"- nondegenerate task IDs: `{summary.get('nondegenerate_task_count')}`",
            f"- rollouts per task: `{summary.get('rollouts_per_task')}`",
            f"- horizon: `{summary.get('horizon')}`",
            f"- total wall-clock seconds: `{summary.get('wall_clock_seconds')}`",
            "",
            "## Runnable Task IDs",
            "",
        ]
        for env_id in summary.get("runnable_env_ids") or []:
            lines.append(f"- `{env_id}`")
        lines.extend(
            [
                "",
                "This is a reset/clone/short-rollout viability probe. It does not promote these task IDs to learned-WAM, exact-law, closed-loop, or solved-policy evidence; those require the heavier rollout-pool and CI artifacts.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    paths = _paths(args.output_tag)
    ok, reason = is_robocasa_available()
    if not ok:
        summary = {
            "experiment": paths["summary"].stem,
            "attempted": True,
            "available": False,
            "verified": False,
            "output_tag": args.output_tag,
            "reason": reason,
            "report_path": str(paths["report"]),
        }
        write_json(paths["summary"], summary)
        _write_report(summary)
        return summary

    rows: list[dict[str, Any]] = []
    started = time.time()
    for task_index, env_id in enumerate(args.env_ids):
        t0 = time.time()
        adapter: RoboCasaAdapter | None = None
        row: dict[str, Any] = {
            "env_id": env_id,
            "task_index": int(task_index),
            "seed": int(args.seed + task_index),
            "rollouts": int(args.rollouts),
            "horizon": int(args.horizon),
        }
        try:
            print(f"[robocasa-micro] probing {env_id} ({task_index + 1}/{len(args.env_ids)})", flush=True)
            adapter = RoboCasaAdapter(
                env_id=env_id,
                split=args.split,
                horizon=args.horizon,
                camera_width=args.camera_size,
                camera_height=args.camera_size,
            )
            initial_state = adapter.reset_task(seed=args.seed + task_index)
            initial_distance = float(adapter.object_distance())
            pool = adapter.sample_rollouts(
                initial_state=initial_state,
                n_rollouts=args.rollouts,
                horizon=args.horizon,
                seed=args.seed + 10_003 + task_index,
            )
            records = pool["records"]
            utilities = np.asarray([float(r["utility"]) for r in records], dtype=float)
            progress = np.asarray([float(r["progress"]) for r in records], dtype=float)
            final_distance = np.asarray([float(r["final_distance"]) for r in records], dtype=float)
            row.update(
                {
                    "available": True,
                    "reset_ok": True,
                    "rollout_ok": len(records) == args.rollouts,
                    "initial_distance": initial_distance,
                    "mean_final_distance": float(np.mean(final_distance)),
                    "mean_progress": float(np.mean(progress)),
                    "mean_utility": float(np.mean(utilities)),
                    "utility_std": float(np.std(utilities)),
                    "utility_min": float(np.min(utilities)),
                    "utility_max": float(np.max(utilities)),
                    "success_count": int(sum(bool(r["success"]) for r in records)),
                    "nondegenerate": bool(initial_distance > 0.0 and len(records) == args.rollouts and np.isfinite(utilities).all()),
                    "error": "",
                }
            )
        except (RoboCasaUnavailableError, Exception) as exc:  # pragma: no cover - optional dependency path
            row.update(
                {
                    "available": False,
                    "reset_ok": False,
                    "rollout_ok": False,
                    "initial_distance": np.nan,
                    "mean_final_distance": np.nan,
                    "mean_progress": np.nan,
                    "mean_utility": np.nan,
                    "utility_std": np.nan,
                    "utility_min": np.nan,
                    "utility_max": np.nan,
                    "success_count": 0,
                    "nondegenerate": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            row["seconds"] = float(time.time() - t0)
            if adapter is not None:
                adapter.close()
        rows.append(row)
        print(
            f"[robocasa-micro] {env_id}: nondegenerate={row.get('nondegenerate')} seconds={row['seconds']:.1f}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(paths["table"], index=False)
    runnable = table[table["rollout_ok"].astype(bool) & table["reset_ok"].astype(bool)] if not table.empty else table
    nondegenerate = table[table["nondegenerate"].astype(bool)] if not table.empty else table
    summary = {
        "experiment": paths["summary"].stem,
        "attempted": True,
        "available": True,
        "verified": bool(len(args.env_ids) >= args.min_tasks and len(nondegenerate) >= args.min_tasks),
        "output_tag": args.output_tag,
        "candidate_task_count": int(len(args.env_ids)),
        "runnable_task_count": int(len(runnable)),
        "nondegenerate_task_count": int(len(nondegenerate)),
        "runnable_env_ids": [str(e) for e in runnable["env_id"].tolist()],
        "nondegenerate_env_ids": [str(e) for e in nondegenerate["env_id"].tolist()],
        "rollouts_per_task": int(args.rollouts),
        "horizon": int(args.horizon),
        "split": args.split,
        "wall_clock_seconds": float(time.time() - started),
        "table_path": str(paths["table"]),
        "report_path": str(paths["report"]),
        "note": "RoboCasa micro-rollout viability only; not learned-WAM, exact-law, closed-loop, solved-policy, or full-suite validation.",
    }
    write_json(paths["summary"], summary)
    _write_report(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="*", default=DEFAULT_ENV_IDS)
    parser.add_argument("--output-tag", default="extra")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--camera-size", type=int, default=8)
    parser.add_argument("--min-tasks", type=int, default=4)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
