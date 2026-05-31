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

from wam_inference_value.benchmarks.libero_adapter import LIBEROAdapter, LIBEROUnavailableError, is_libero_available
from wam_inference_value.stats import bootstrap_ci


RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for path in [RESULTS, RESULTS / "tables", RESULTS / "models", REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")


def parse_task_ids(raw_tasks: list[str], default_suite: str) -> list[str]:
    out: list[str] = []
    for raw in raw_tasks:
        item = str(raw).strip()
        if not item:
            continue
        out.append(item if "/" in item else f"{default_suite}/{int(item)}")
    return out


def task_index(task_id: str) -> int:
    return int(task_id.split("/", 1)[1])


def _obs_vector(adapter: LIBEROAdapter, key: str) -> np.ndarray:
    arr = adapter._obs_vector(key)
    return np.zeros(0, dtype=float) if arr is None else np.asarray(arr, dtype=float).reshape(-1)


def build_feature(
    adapter: LIBEROAdapter,
    target_point: np.ndarray,
    gripper_command: float,
    phase_index: int,
    n_phases: int,
    task_id: str,
    task_ids: list[str],
) -> np.ndarray:
    objects = list(getattr(adapter.env, "obj_of_interest", []) or [])
    obj = adapter._position(objects[0]) if objects else None
    goal = adapter._position(objects[1]) if len(objects) > 1 else None
    eef = adapter._eef_pos()
    target = np.asarray(target_point, dtype=float).reshape(3)
    obj_pos = np.zeros(3, dtype=float) if obj is None else np.asarray(obj, dtype=float).reshape(3)
    goal_pos = np.zeros(3, dtype=float) if goal is None else np.asarray(goal, dtype=float).reshape(3)
    phase = np.zeros(int(n_phases), dtype=float)
    phase[int(phase_index)] = 1.0
    tasks = np.zeros(len(task_ids), dtype=float)
    tasks[task_ids.index(task_id)] = 1.0
    gripper_q = _obs_vector(adapter, "robot0_gripper_qpos")
    if gripper_q.size == 0:
        gripper_q = np.zeros(2, dtype=float)
    elif gripper_q.size < 2:
        gripper_q = np.pad(gripper_q, (0, 2 - gripper_q.size))
    return np.concatenate(
        [
            eef,
            obj_pos,
            goal_pos,
            target,
            target - eef,
            obj_pos - eef,
            obj_pos - goal_pos,
            np.asarray([float(gripper_command), adapter.task_distance(), float(adapter.evaluate_success())], dtype=float),
            gripper_q[:2],
            phase,
            tasks,
        ]
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    z = np.column_stack([np.ones(len(z)), z])
    reg = float(alpha) * np.eye(z.shape[1])
    reg[0, 0] = 0.0
    weights = np.linalg.solve(z.T @ z + reg, z.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(1, -1)
    z = (arr - model["mean"]) / model["scale"]
    z = np.column_stack([np.ones(len(z)), z])
    return (z @ model["weights"])[0]


def scripted_action(adapter: LIBEROAdapter, target: np.ndarray, gripper: float, gain: float) -> np.ndarray:
    eef = adapter._eef_pos()
    delta = (np.asarray(target, dtype=float).reshape(3) - eef) * float(gain)
    action = np.zeros(adapter.action_dim, dtype=float)
    action[: min(3, adapter.action_dim)] = np.clip(delta[: min(3, adapter.action_dim)], -1.0, 1.0)
    if adapter.action_dim:
        action[-1] = float(gripper)
    return np.clip(action, adapter.action_low, adapter.action_high)


def phase_targets(adapter: LIBEROAdapter, args: argparse.Namespace) -> list[tuple[str, np.ndarray, float, int]]:
    objects = list(getattr(adapter.env, "obj_of_interest", []) or [])
    if len(objects) < 2:
        return []
    obj = adapter._position(objects[0])
    target = adapter._position(objects[1])
    if obj is None or target is None:
        return []
    grasp_xy = np.asarray([obj[0] + args.grasp_offset_x, obj[1] + args.grasp_offset_y], dtype=float)
    zsafe = max(float(obj[2]), float(target[2])) + float(args.safe_lift)
    return [
        ("open_above_object", np.asarray([grasp_xy[0], grasp_xy[1], zsafe], dtype=float), -1.0, args.above_steps),
        (
            "descend_open",
            np.asarray([grasp_xy[0], grasp_xy[1], float(obj[2]) + args.approach_z_offset], dtype=float),
            -1.0,
            args.descend_steps,
        ),
        (
            "close_gripper",
            np.asarray([grasp_xy[0], grasp_xy[1], float(obj[2]) + args.grasp_z_offset], dtype=float),
            1.0,
            args.close_steps,
        ),
        ("lift_object", np.asarray([grasp_xy[0], grasp_xy[1], zsafe], dtype=float), 1.0, args.lift_steps),
        ("move_to_target", np.asarray([target[0], target[1], zsafe], dtype=float), 1.0, args.move_steps),
        (
            "lower_to_target",
            np.asarray([target[0], target[1], float(target[2]) + args.place_z_offset], dtype=float),
            1.0,
            args.place_steps,
        ),
        (
            "open_gripper",
            np.asarray([target[0], target[1], float(target[2]) + args.place_z_offset], dtype=float),
            -1.0,
            args.open_steps,
        ),
        ("retreat", np.asarray([target[0], target[1], zsafe], dtype=float), -1.0, args.retreat_steps),
    ]


def run_episode(
    adapter: LIBEROAdapter,
    args: argparse.Namespace,
    task_id: str,
    task_ids: list[str],
    *,
    model: dict[str, np.ndarray] | None,
    collect: bool,
) -> dict[str, Any]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    initial_distance = float(adapter.task_distance())
    total_reward = 0.0
    energy = 0.0
    steps = 0
    phases = phase_targets(adapter, args)
    if not phases:
        return {"success": False, "features": features, "actions": actions, "failure_reason": "phase targets unavailable"}
    for phase_index, (_, target, gripper, n_steps) in enumerate(phases):
        for _ in range(int(n_steps)):
            if getattr(adapter, "last_done", False) or adapter.evaluate_success():
                break
            feat = build_feature(adapter, target, gripper, phase_index, len(phases), task_id, task_ids)
            if model is None:
                action = scripted_action(adapter, target, gripper, args.servo_gain)
            else:
                action = np.clip(predict(model, feat), adapter.action_low, adapter.action_high)
            if collect:
                features.append(feat)
                actions.append(scripted_action(adapter, target, gripper, args.servo_gain))
            try:
                _, reward, done, truncated, _ = adapter.step(action)
            except ValueError as exc:
                return {
                    "success": bool(adapter.evaluate_success()),
                    "features": features,
                    "actions": actions,
                    "failure_reason": f"step failed after termination: {exc}",
                    "total_reward": total_reward,
                    "energy": energy,
                    "steps": steps,
                }
            total_reward += float(reward)
            energy += float(np.sum(action * action))
            steps += 1
            if done or truncated:
                break
        if getattr(adapter, "last_done", False) or adapter.evaluate_success():
            break
    final_distance = float(adapter.task_distance())
    progress = float(initial_distance - final_distance)
    return {
        "success": bool(adapter.evaluate_success()),
        "features": features,
        "actions": actions,
        "failure_reason": None,
        "total_reward": float(total_reward),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "progress": progress,
        "energy": float(energy),
        "steps": int(steps),
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


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ci = (summary.get("confidence_intervals") or {}).get("eval_success_rate") or {}
    lines = [
        "# LIBERO Learned Action-Head Smoke Report",
        "",
        "This optional artifact imitates the successful LIBERO Object scripted controller with a learned ridge action head. The phase schedule and target points are still scripted, so this is a narrow learned-control smoke, not a learned autonomous LIBERO policy.",
        "",
        "## Summary",
        "",
        f"- Available: `{summary.get('available')}`.",
        f"- Verified: `{summary.get('verified')}`.",
        f"- Train episodes: `{summary.get('train_episodes')}`.",
        f"- Train action examples: `{summary.get('train_examples')}`.",
        f"- Eval episodes: `{summary.get('eval_episodes')}`.",
        f"- Eval successes: `{summary.get('eval_successes')}`.",
        f"- Eval success rate: `{ci.get('mean')}` with bootstrap CI [`{ci.get('lo')}`, `{ci.get('hi')}`].",
        f"- Action MAE on collected train examples: `{summary.get('train_action_mae')}`.",
        "",
        "## Limitations",
        "",
        "- The learned component is only the continuous action head.",
        "- High-level phase ordering and target-point construction are still scripted.",
        "- The artifact evaluates the scripted-success LIBERO Object subset, not all LIBERO suites.",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "libero_learned_action_head_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def unavailable_summary(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "benchmark_libero_learned_action_head",
        "available": False,
        "attempted": True,
        "verified": False,
        "reason": reason,
        "note": "Optional LIBERO learned action-head smoke; unavailable in this interpreter.",
        "tasks": parse_task_ids(args.tasks, args.suite),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", nargs="+", default=["0", "2", "3", "4", "7", "9"])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[100, 101])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[200, 201, 202])
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=919)
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dirs()
    ok, reason = is_libero_available()
    if not ok:
        summary = unavailable_summary(reason, args)
        write_json(RESULTS / "benchmark_libero_learned_action_head.json", summary)
        write_report(summary, [])
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
                out = run_episode(adapter, args, tid, task_ids, model=None, collect=True)
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
        model = fit_ridge(x, y, args.ridge_alpha)
        pred_train = np.asarray([predict(model, feat) for feat in x], dtype=float)
        action_mae = float(np.mean(np.abs(pred_train - y)))
        action_rmse = float(np.sqrt(np.mean((pred_train - y) ** 2)))
        model_path = RESULTS / "models" / "benchmark_libero_learned_action_head.npz"
        np.savez(
            model_path,
            mean=model["mean"],
            scale=model["scale"],
            weights=model["weights"],
            task_ids=np.asarray(task_ids, dtype=object),
            train_seeds=np.asarray(args.train_seeds, dtype=int),
            eval_seeds=np.asarray(args.eval_seeds, dtype=int),
        )
        for tid in task_ids:
            for seed in args.eval_seeds:
                adapter.reset(int(seed), task_id=tid)
                out = run_episode(adapter, args, tid, task_ids, model=model, collect=False)
                rows.append(
                    {
                        "split": "eval_learned_action_head",
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
        write_json(RESULTS / "benchmark_libero_learned_action_head.json", summary)
        write_csv(RESULTS / "tables" / "benchmark_libero_learned_action_head_episodes.csv", rows)
        write_report(summary, rows)
        if args.fail_on_low_success:
            raise
        return
    finally:
        if adapter is not None:
            adapter.close()

    eval_rows = [r for r in rows if r.get("split") == "eval_learned_action_head"]
    successes = [float(r.get("success", False)) for r in eval_rows]
    ci = bootstrap_ci(successes, seed=args.seed, n_boot=args.bootstrap_samples)
    train_successes = [float(r.get("success", False)) for r in rows if r.get("split") == "train_scripted"]
    verified = (
        len(eval_rows) >= len(task_ids) * len(args.eval_seeds)
        and (ci.get("mean") or 0.0) >= float(args.min_success_rate)
        and (ci.get("lo") or 0.0) >= float(args.min_success_ci_lo)
    )
    summary = {
        "experiment": "benchmark_libero_learned_action_head",
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
        "train_action_mae": action_mae,
        "train_action_rmse": action_rmse,
        "model_path": str(model_path.relative_to(ROOT)),
        "artifact_paths": {
            "json": "results/benchmark_libero_learned_action_head.json",
            "episodes_csv": "results/tables/benchmark_libero_learned_action_head_episodes.csv",
            "report": "reports/libero_learned_action_head_report.md",
        },
        "note": "Learned ridge action head with scripted phases and target points; not autonomous learned LIBERO policy performance.",
    }
    write_json(RESULTS / "benchmark_libero_learned_action_head.json", summary)
    write_csv(RESULTS / "tables" / "benchmark_libero_learned_action_head_episodes.csv", rows)
    write_report(summary, rows)
    print(json.dumps(sanitize(summary), indent=2))
    if args.fail_on_low_success and not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
