from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from benchmark_libero_learned_action_head import (
    parse_task_ids,
    phase_targets,
    sanitize,
    scripted_action,
    task_index,
)
from libero_object_grasp_tuning import ALL_LIBERO_OBJECT_TASKS
from wam_inference_value.benchmarks.libero_adapter import LIBEROAdapter, LIBEROUnavailableError, is_libero_available
from wam_inference_value.stats import bootstrap_ci


RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for path in [RESULTS, RESULTS / "tables", RESULTS / "models", REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")


def _obs_vector(adapter: LIBEROAdapter, key: str, size: int) -> np.ndarray:
    arr = adapter._obs_vector(key)
    if arr is None:
        return np.zeros(size, dtype=float)
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.size < size:
        out = np.pad(out, (0, size - out.size))
    return out[:size]


def state_feature(
    adapter: LIBEROAdapter,
    task_id: str,
    task_ids: list[str],
    prev_action: np.ndarray | None = None,
    step: int = 0,
    max_steps: int = 1,
) -> np.ndarray:
    """Low-dimensional autonomous policy feature.

    This intentionally excludes scripted phase indices and commanded target
    points. It uses simulator state observations, task ID, a finite-horizon
    step clock, and optional previous action as policy memory.
    """

    objects = list(getattr(adapter.env, "obj_of_interest", []) or [])
    obj = adapter._position(objects[0]) if objects else None
    goal = adapter._position(objects[1]) if len(objects) > 1 else None
    eef = adapter._eef_pos()
    obj_pos = np.zeros(3, dtype=float) if obj is None else np.asarray(obj, dtype=float).reshape(3)
    goal_pos = np.zeros(3, dtype=float) if goal is None else np.asarray(goal, dtype=float).reshape(3)
    gripper_q = _obs_vector(adapter, "robot0_gripper_qpos", 2)
    gripper_v = _obs_vector(adapter, "robot0_gripper_qvel", 2)
    eef_quat = _obs_vector(adapter, "robot0_eef_quat", 4)
    prev = np.zeros(adapter.action_dim, dtype=float) if prev_action is None else np.asarray(prev_action, dtype=float).reshape(-1)
    if prev.size < adapter.action_dim:
        prev = np.pad(prev, (0, adapter.action_dim - prev.size))
    prev = prev[: adapter.action_dim]
    tasks = np.zeros(len(task_ids), dtype=float)
    tasks[task_ids.index(task_id)] = 1.0
    scalars = np.asarray(
        [
            float(np.linalg.norm(obj_pos - eef)),
            float(np.linalg.norm(obj_pos - goal_pos)),
            float(np.linalg.norm(eef - goal_pos)),
            float(adapter.task_distance()),
            float(adapter.evaluate_success()),
            float(step) / float(max(max_steps, 1)),
            np.sin(2.0 * np.pi * float(step) / float(max(max_steps, 1))),
            np.cos(2.0 * np.pi * float(step) / float(max(max_steps, 1))),
        ],
        dtype=float,
    )
    return np.concatenate(
        [
            eef,
            eef_quat,
            obj_pos,
            goal_pos,
            obj_pos - eef,
            goal_pos - eef,
            obj_pos - goal_pos,
            gripper_q,
            gripper_v,
            prev,
            scalars,
            tasks,
        ]
    )


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def knn_predict(
    x_train_z: np.ndarray,
    y_train: np.ndarray,
    x_z: np.ndarray,
    *,
    k: int,
    temperature: float,
) -> np.ndarray:
    diff = x_train_z - x_z.reshape(1, -1)
    dist2 = np.sum(diff * diff, axis=1)
    k_eff = min(int(k), len(dist2))
    idx = np.argpartition(dist2, k_eff - 1)[:k_eff]
    local = dist2[idx]
    scale = max(float(temperature), 1e-8)
    weights = np.exp(-(local - float(np.min(local))) / scale)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 1e-12:
        weights = np.ones_like(weights)
    return np.average(y_train[idx], axis=0, weights=weights)


def collect_scripted_episode(
    adapter: LIBEROAdapter,
    args: argparse.Namespace,
    task_id: str,
    task_ids: list[str],
) -> dict[str, Any]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    prev_action = np.zeros(adapter.action_dim, dtype=float)
    initial_distance = float(adapter.task_distance())
    total_reward = 0.0
    energy = 0.0
    steps = 0
    phases = phase_targets(adapter, args)
    if not phases:
        return {"success": False, "features": features, "actions": actions, "failure_reason": "phase targets unavailable"}
    for _, target, gripper, n_steps in phases:
        for _ in range(int(n_steps)):
            if getattr(adapter, "last_done", False) or adapter.evaluate_success():
                break
            feature = state_feature(adapter, task_id, task_ids, prev_action, steps, args.eval_steps)
            action = scripted_action(adapter, target, gripper, args.servo_gain)
            features.append(feature)
            actions.append(action)
            try:
                _, reward, done, truncated, _ = adapter.step(action)
            except ValueError as exc:
                return {"success": bool(adapter.evaluate_success()), "features": features, "actions": actions, "failure_reason": str(exc)}
            total_reward += float(reward)
            energy += float(np.sum(action * action))
            steps += 1
            prev_action = action
            if done or truncated:
                break
        if getattr(adapter, "last_done", False) or adapter.evaluate_success():
            break
    final_distance = float(adapter.task_distance())
    return {
        "success": bool(adapter.evaluate_success()),
        "features": features,
        "actions": actions,
        "failure_reason": None,
        "total_reward": float(total_reward),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "progress": float(initial_distance - final_distance),
        "energy": float(energy),
        "steps": int(steps),
    }


def run_bc_episode(
    adapter: LIBEROAdapter,
    args: argparse.Namespace,
    task_id: str,
    task_ids: list[str],
    x_train_z: np.ndarray,
    y_train: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    prev_action = np.zeros(adapter.action_dim, dtype=float)
    initial_distance = float(adapter.task_distance())
    total_reward = 0.0
    energy = 0.0
    steps = 0
    failure_reason = None
    for _ in range(int(args.eval_steps)):
        if getattr(adapter, "last_done", False) or adapter.evaluate_success():
            break
        feature = state_feature(adapter, task_id, task_ids, prev_action, steps, args.eval_steps)
        z = (feature - mean) / scale
        action = knn_predict(x_train_z, y_train, z, k=args.knn_k, temperature=args.knn_temperature)
        action = np.clip(action, adapter.action_low, adapter.action_high)
        try:
            _, reward, done, truncated, _ = adapter.step(action)
        except ValueError as exc:
            failure_reason = str(exc)
            break
        total_reward += float(reward)
        energy += float(np.sum(action * action))
        steps += 1
        prev_action = action
        if done or truncated:
            break
    final_distance = float(adapter.task_distance())
    return {
        "success": bool(adapter.evaluate_success()),
        "total_reward": float(total_reward),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "progress": float(initial_distance - final_distance),
        "energy": float(energy),
        "steps": int(steps),
        "failure_reason": failure_reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "task_id",
        "task_name",
        "seed",
        "success",
        "total_reward",
        "initial_distance",
        "final_distance",
        "progress",
        "energy",
        "steps",
        "failure_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(summary: dict[str, Any]) -> None:
    ci = (summary.get("confidence_intervals") or {}).get("eval_success_rate") or {}
    lines = [
        "# LIBERO Autonomous BC Policy Report",
        "",
        "This optional artifact evaluates a low-dimensional behavior-cloned kNN policy on LIBERO Object tasks. The policy receives simulator state features, task ID, previous action memory, and a finite-horizon step clock; it does not receive scripted phase indices or commanded target points.",
        "",
        "## Summary",
        "",
        f"- Available: `{summary.get('available')}`.",
        f"- Verified: `{summary.get('verified')}`.",
        f"- Train action examples: `{summary.get('train_examples')}`.",
        f"- Eval episodes: `{summary.get('eval_episodes')}`.",
        f"- Eval successes: `{summary.get('eval_successes')}`.",
        f"- Eval success rate: `{ci.get('mean')}` with bootstrap CI [`{ci.get('lo')}`, `{ci.get('hi')}`].",
        "",
        "## Claim Boundary",
        "",
        "- This is low-dimensional simulator-state behavior cloning, not image-based or language-conditioned LIBERO.",
        "- It does not use scripted phase labels or target-point commands at evaluation time.",
        "- It is time-conditioned; this is stronger than a scripted target/action-head smoke but still not a broad robust autonomous LIBERO policy.",
        "- The default artifact evaluates all ten LIBERO Object tasks, not all LIBERO suites.",
        "- Demonstrations come from the hand-coded object-tuned scripted controller.",
    ]
    (REPORTS / "libero_autonomous_bc_policy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def unavailable_summary(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "benchmark_libero_autonomous_bc_policy",
        "available": False,
        "attempted": True,
        "verified": False,
        "reason": reason,
        "tasks": parse_task_ids(args.tasks, args.suite),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", nargs="+", default=ALL_LIBERO_OBJECT_TASKS)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[200, 201, 202])
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--eval-steps", type=int, default=280)
    parser.add_argument("--knn-k", type=int, default=7)
    parser.add_argument("--knn-temperature", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    parser.add_argument("--min-success-ci-lo", type=float, default=0.6)
    parser.add_argument("--fail-on-low-success", action="store_true")
    parser.add_argument("--safe-lift", type=float, default=0.25)
    parser.add_argument("--approach-z-offset", type=float, default=0.055)
    parser.add_argument("--grasp-z-offset", type=float, default=0.035)
    parser.add_argument("--place-z-offset", type=float, default=0.11)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--servo-gain", type=float, default=8.0)
    parser.add_argument("--above-steps", type=int, default=35)
    parser.add_argument("--descend-steps", type=int, default=25)
    parser.add_argument("--close-steps", type=int, default=35)
    parser.add_argument("--lift-steps", type=int, default=45)
    parser.add_argument("--move-steps", type=int, default=60)
    parser.add_argument("--place-steps", type=int, default=25)
    parser.add_argument("--open-steps", type=int, default=35)
    parser.add_argument("--retreat-steps", type=int, default=20)
    parser.add_argument("--object-grasp-tuning", dest="object_grasp_tuning", action="store_true", default=True)
    parser.add_argument("--disable-object-grasp-tuning", dest="object_grasp_tuning", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dirs()
    ok, reason = is_libero_available()
    if not ok:
        summary = unavailable_summary(reason, args)
        write_json(RESULTS / "benchmark_libero_autonomous_bc_policy.json", summary)
        write_report(summary)
        print(reason)
        return

    task_ids = parse_task_ids(args.tasks, args.suite)
    rows: list[dict[str, Any]] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    adapter: LIBEROAdapter | None = None
    try:
        adapter = LIBEROAdapter(
            suite=args.suite,
            task_index=task_index(task_ids[0]),
            horizon=args.horizon,
            controller=args.controller,
            use_camera_obs=False,
            has_offscreen_renderer=False,
        )
        for tid in task_ids:
            for seed in args.train_seeds:
                adapter.reset(int(seed), task_id=tid)
                out = collect_scripted_episode(adapter, args, tid, task_ids)
                if out["features"] and out["actions"]:
                    x_parts.append(np.vstack(out["features"]))
                    y_parts.append(np.vstack(out["actions"]))
                rows.append(
                    {
                        "split": "train_scripted",
                        "task_id": tid,
                        "task_name": str(getattr(adapter.task, "name", tid)),
                        "seed": int(seed),
                        **{k: out.get(k) for k in ["success", "total_reward", "initial_distance", "final_distance", "progress", "energy", "steps", "failure_reason"]},
                    }
                )
                print(f"train {tid} seed={seed} success={out.get('success')} examples={len(out['features'])}")
        if not x_parts:
            raise RuntimeError("no train examples collected")
        x = np.vstack(x_parts)
        y = np.vstack(y_parts)
        mean, scale = standardize_fit(x)
        x_z = (x - mean) / scale
        model_path = RESULTS / "models" / "benchmark_libero_autonomous_bc_policy.npz"
        np.savez(
            model_path,
            x_train=x.astype(np.float32),
            y_train=y.astype(np.float32),
            mean=mean,
            scale=scale,
            task_ids=np.asarray(task_ids, dtype=object),
            train_seeds=np.asarray(args.train_seeds, dtype=int),
            eval_seeds=np.asarray(args.eval_seeds, dtype=int),
        )
        for tid in task_ids:
            for seed in args.eval_seeds:
                adapter.reset(int(seed), task_id=tid)
                out = run_bc_episode(adapter, args, tid, task_ids, x_z, y, mean, scale)
                rows.append(
                    {
                        "split": "eval_autonomous_bc",
                        "task_id": tid,
                        "task_name": str(getattr(adapter.task, "name", tid)),
                        "seed": int(seed),
                        **{k: out.get(k) for k in ["success", "total_reward", "initial_distance", "final_distance", "progress", "energy", "steps", "failure_reason"]},
                    }
                )
                print(f"eval {tid} seed={seed} success={out.get('success')} progress={out.get('progress')}")
    except (LIBEROUnavailableError, RuntimeError, ValueError) as exc:
        summary = unavailable_summary(f"{type(exc).__name__}: {exc}", args)
        summary["rows"] = rows
        write_json(RESULTS / "benchmark_libero_autonomous_bc_policy.json", summary)
        write_csv(RESULTS / "tables" / "benchmark_libero_autonomous_bc_policy_episodes.csv", rows)
        write_report(summary)
        if args.fail_on_low_success:
            raise
        return
    finally:
        if adapter is not None:
            adapter.close()

    eval_rows = [r for r in rows if r.get("split") == "eval_autonomous_bc"]
    successes = [float(r.get("success", False)) for r in eval_rows]
    ci = bootstrap_ci(successes, seed=args.seed, n_boot=args.bootstrap_samples)
    train_successes = [float(r.get("success", False)) for r in rows if r.get("split") == "train_scripted"]
    verified = (
        len(eval_rows) >= len(task_ids) * len(args.eval_seeds)
        and (ci.get("mean") or 0.0) >= float(args.min_success_rate)
        and (ci.get("lo") or 0.0) >= float(args.min_success_ci_lo)
    )
    summary = {
        "experiment": "benchmark_libero_autonomous_bc_policy",
        "available": True,
        "attempted": True,
        "verified": bool(verified),
        "tasks": task_ids,
        "train_seeds": [int(s) for s in args.train_seeds],
        "eval_seeds": [int(s) for s in args.eval_seeds],
        "train_episodes": int(len(train_successes)),
        "train_successes": int(sum(train_successes)),
        "train_examples": int(len(x)),
        "eval_episodes": int(len(eval_rows)),
        "eval_successes": int(sum(successes)),
        "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
        "confidence_intervals": {"eval_success_rate": ci},
        "policy": {
            "type": "low_dim_knn_behavior_cloning",
            "uses_phase_index": False,
            "uses_target_point_command": False,
            "uses_task_id": True,
            "uses_previous_action": True,
            "uses_step_clock": True,
            "knn_k": int(args.knn_k),
            "knn_temperature": float(args.knn_temperature),
        },
        "object_grasp_tuning": bool(getattr(args, "object_grasp_tuning", True)),
        "model_path": str(model_path.relative_to(ROOT)),
        "artifact_paths": {
            "json": "results/benchmark_libero_autonomous_bc_policy.json",
            "episodes_csv": "results/tables/benchmark_libero_autonomous_bc_policy_episodes.csv",
            "report": "reports/libero_autonomous_bc_policy_report.md",
        },
        "note": "Low-dimensional time-conditioned simulator-state BC policy without scripted phase labels or target-point commands; not image/language LIBERO.",
    }
    write_json(RESULTS / "benchmark_libero_autonomous_bc_policy.json", summary)
    write_csv(RESULTS / "tables" / "benchmark_libero_autonomous_bc_policy_episodes.csv", rows)
    write_report(summary)
    print(json.dumps(sanitize(summary), indent=2))
    if args.fail_on_low_success and not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
