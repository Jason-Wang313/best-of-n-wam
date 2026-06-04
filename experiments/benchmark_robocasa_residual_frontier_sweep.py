from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json


def _category(env_id: str) -> str:
    name = env_id.split("/", 1)[-1]
    if name.startswith("PickPlace"):
        return "pick_place"
    if name.startswith("Open"):
        return "open"
    if name.startswith("Close"):
        return "close"
    if name.startswith("Turn"):
        return "turn"
    if name.startswith("Move"):
        return "move"
    if name.startswith("Manipulate"):
        return "manipulate"
    if any(token in name for token in ("Clean", "Clear", "Wash", "Rinse", "Dry")):
        return "cleaning"
    if any(token in name for token in ("Cook", "Heat", "Boil", "Bake", "Oven", "Microwave", "Toast")):
        return "cooking"
    return "long_horizon_or_compositional"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_ids() -> list[str]:
    registry_path = results_dir() / "tables" / "benchmark_robocasa_catalog_registry.csv"
    if registry_path.exists():
        df = pd.read_csv(registry_path)
        return sorted(str(env_id) for env_id in df["env_id"].tolist())

    import gymnasium as gym
    import robocasa  # noqa: F401 - registers environments

    return sorted(spec.id for spec in gym.envs.registry.values() if spec.id.startswith("robocasa/"))


def _candidate_ids(args: argparse.Namespace) -> list[str]:
    if args.env_ids:
        candidates = [str(env_id) for env_id in args.env_ids]
    else:
        catalog = _load_json(results_dir() / "benchmark_robocasa_catalog_probe.json")
        covered = set(str(env_id) for env_id in catalog.get("any_artifact_env_ids") or [])
        candidates = [env_id for env_id in _registry_ids() if env_id not in covered]
    if args.categories:
        categories = set(args.categories)
        candidates = [env_id for env_id in candidates if _category(env_id) in categories]
    if args.max_tasks is not None:
        candidates = candidates[: int(args.max_tasks)]
    return candidates


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), int(size))]


def _probe_path(tag: str) -> Path:
    return results_dir() / f"benchmark_robocasa_micro_rollout_{tag}.json"


def _timeout_summary(tag: str, env_ids: list[str], timeout_seconds: float, seconds: float) -> dict[str, Any]:
    return {
        "experiment": f"benchmark_robocasa_micro_rollout_{tag}",
        "attempted": True,
        "available": True,
        "verified": False,
        "output_tag": tag,
        "candidate_task_count": len(env_ids),
        "runnable_task_count": 0,
        "nondegenerate_task_count": 0,
        "runnable_env_ids": [],
        "nondegenerate_env_ids": [],
        "timed_out": True,
        "timeout_seconds": float(timeout_seconds),
        "wall_clock_seconds": float(seconds),
        "env_ids": env_ids,
        "note": "Chunk timed out during RoboCasa micro-rollout probing; no validation claim is promoted.",
    }


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.kill()


def _run_chunk(tag: str, env_ids: list[str], args: argparse.Namespace) -> dict[str, Any]:
    path = _probe_path(tag)
    if args.skip_existing and path.exists():
        payload = _load_json(path)
        payload["reused_existing"] = True
        return payload

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"robocasa_micro_{tag}.out.log"
    stderr_path = log_dir / f"robocasa_micro_{tag}.err.log"
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "benchmark_robocasa_micro_rollout_probe.py"),
        "--output-tag",
        tag,
        "--rollouts",
        str(args.rollouts),
        "--horizon",
        str(args.horizon),
        "--min-tasks",
        "1",
        "--env-ids",
        *env_ids,
    ]
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr)
        try:
            proc.wait(timeout=float(args.timeout_seconds))
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            summary = _timeout_summary(tag, env_ids, float(args.timeout_seconds), time.time() - started)
            write_json(path, summary)
            return summary

    payload = _load_json(path)
    if payload:
        payload["stdout_path"] = str(stdout_path)
        payload["stderr_path"] = str(stderr_path)
        return payload
    return {
        "experiment": f"benchmark_robocasa_micro_rollout_{tag}",
        "attempted": True,
        "available": False,
        "verified": False,
        "output_tag": tag,
        "candidate_task_count": len(env_ids),
        "runnable_task_count": 0,
        "nondegenerate_task_count": 0,
        "runnable_env_ids": [],
        "nondegenerate_env_ids": [],
        "reason": "probe process exited without writing a summary JSON",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _write_report(summary: dict[str, Any]) -> None:
    report_path = ROOT / "reports" / "robocasa_residual_frontier_sweep_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RoboCasa Residual Frontier Sweep",
        "",
        f"- status: `{'verified' if summary.get('verified') else 'attempted_not_promoted'}`",
        f"- candidate task IDs: `{summary.get('candidate_task_count')}`",
        f"- completed chunks: `{summary.get('completed_chunk_count')}`",
        f"- timed-out chunks: `{summary.get('timed_out_chunk_count')}`",
        f"- runnable task IDs: `{summary.get('runnable_task_count')}`",
        f"- nondegenerate task IDs: `{summary.get('nondegenerate_task_count')}`",
        f"- timeout seconds per chunk: `{summary.get('timeout_seconds')}`",
        "",
        "## Nondegenerate Task IDs",
        "",
    ]
    for env_id in summary.get("nondegenerate_env_ids") or []:
        lines.append(f"- `{env_id}`")
    lines.extend(
        [
            "",
            "This is a resumable micro-rollout frontier search. It is not learned-WAM, exact-law, closed-loop, solved-policy, or full-suite validation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    candidates = _candidate_ids(args)
    chunk_rows: list[dict[str, Any]] = []
    runnable: set[str] = set()
    nondegenerate: set[str] = set()
    started = time.time()
    chunks = _chunks(candidates, args.chunk_size)
    for chunk_index, env_ids in enumerate(chunks):
        tag = f"{args.output_tag_prefix}_{chunk_index:03d}"
        print(f"[robocasa-residual-sweep] probing chunk {chunk_index + 1}/{len(chunks)}: {tag}", flush=True)
        payload = _run_chunk(tag, env_ids, args)
        runnable.update(str(env_id) for env_id in payload.get("runnable_env_ids") or [])
        nondegenerate.update(str(env_id) for env_id in payload.get("nondegenerate_env_ids") or [])
        chunk_rows.append(
            {
                "chunk_index": int(chunk_index),
                "tag": tag,
                "env_ids": ";".join(env_ids),
                "candidate_task_count": int(len(env_ids)),
                "verified": bool(payload.get("verified", False)),
                "timed_out": bool(payload.get("timed_out", False)),
                "runnable_task_count": int(payload.get("runnable_task_count") or 0),
                "nondegenerate_task_count": int(payload.get("nondegenerate_task_count") or 0),
                "summary_path": str(_probe_path(tag)),
            }
        )

    chunk_df = pd.DataFrame(chunk_rows)
    table_path = results_dir() / "tables" / "benchmark_robocasa_residual_frontier_sweep_chunks.csv"
    chunk_df.to_csv(table_path, index=False)
    summary = {
        "experiment": "benchmark_robocasa_residual_frontier_sweep",
        "attempted": True,
        "available": True,
        "verified": bool(len(nondegenerate) > 0),
        "candidate_task_count": int(len(candidates)),
        "chunk_count": int(len(chunks)),
        "completed_chunk_count": int(sum(not row["timed_out"] for row in chunk_rows)),
        "timed_out_chunk_count": int(sum(bool(row["timed_out"]) for row in chunk_rows)),
        "runnable_task_count": int(len(runnable)),
        "nondegenerate_task_count": int(len(nondegenerate)),
        "runnable_env_ids": sorted(runnable),
        "nondegenerate_env_ids": sorted(nondegenerate),
        "categories": sorted(set(_category(env_id) for env_id in candidates)),
        "rollouts_per_task": int(args.rollouts),
        "horizon": int(args.horizon),
        "chunk_size": int(args.chunk_size),
        "timeout_seconds": float(args.timeout_seconds),
        "wall_clock_seconds": float(time.time() - started),
        "table_path": str(table_path),
        "report_path": str(ROOT / "reports" / "robocasa_residual_frontier_sweep_report.md"),
        "note": "Resumable RoboCasa residual catalog micro-rollout search; not promoted to learned-WAM benchmark evidence.",
    }
    write_json(results_dir() / "benchmark_robocasa_residual_frontier_sweep.json", summary)
    _write_report(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="*", default=None)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-tag-prefix", default="residual_sweep")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
