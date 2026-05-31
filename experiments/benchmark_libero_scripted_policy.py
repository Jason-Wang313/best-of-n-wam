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
    for path in [RESULTS, RESULTS / "tables", REPORTS]:
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
        if "/" in item:
            out.append(item)
        else:
            out.append(f"{default_suite}/{int(item)}")
    return out


def normalize(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        return [0.5 for _ in arr]
    return [float(v) for v in (arr - lo) / (hi - lo)]


def servo_to(
    adapter: LIBEROAdapter,
    target: np.ndarray | list[float],
    gripper: float,
    *,
    steps: int,
    gain: float,
) -> tuple[float, float, int, str | None]:
    total_reward = 0.0
    energy = 0.0
    steps_done = 0
    target_arr = np.asarray(target, dtype=float).reshape(-1)
    if target_arr.size < 3:
        return 0.0, 0.0, 0, "invalid servo target"
    for _ in range(int(steps)):
        if getattr(adapter, "last_done", False) or adapter.evaluate_success():
            break
        eef = adapter._eef_pos()
        delta = (target_arr[:3] - eef) * float(gain)
        action = np.zeros(adapter.action_dim, dtype=float)
        action[: min(3, adapter.action_dim)] = np.clip(delta[: min(3, adapter.action_dim)], -1.0, 1.0)
        if adapter.action_dim > 0:
            action[-1] = float(gripper)
        energy += float(np.sum(action * action))
        steps_done += 1
        try:
            _, reward, done, truncated, _ = adapter.step(action)
        except ValueError as exc:
            return total_reward, energy, steps_done, f"step failed after termination: {exc}"
        total_reward += float(reward)
        if done or truncated:
            break
    return total_reward, energy, steps_done, None


def run_pick_place_script(adapter: LIBEROAdapter, args: argparse.Namespace) -> dict[str, Any]:
    objects = list(getattr(adapter.env, "obj_of_interest", []) or [])
    if len(objects) < 2:
        return {
            "success": False,
            "total_reward": 0.0,
            "energy": 0.0,
            "steps": 0,
            "failure_reason": "LIBERO task did not expose object and target names",
            "objects": objects,
        }
    obj_name, target_name = objects[0], objects[1]
    obj = adapter._position(obj_name)
    target = adapter._position(target_name)
    if obj is None or target is None:
        return {
            "success": False,
            "total_reward": 0.0,
            "energy": 0.0,
            "steps": 0,
            "failure_reason": "object or target position unavailable",
            "objects": objects,
        }

    grasp_xy = np.asarray([obj[0] + args.grasp_offset_x, obj[1] + args.grasp_offset_y], dtype=float)
    zsafe = max(float(obj[2]), float(target[2])) + float(args.safe_lift)
    total_reward = 0.0
    energy = 0.0
    steps = 0
    errors: list[str] = []

    phase_specs: list[tuple[str, np.ndarray, float, int]] = [
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
    ]

    for name, target_pos, gripper, n_steps in phase_specs:
        reward, phase_energy, phase_steps, error = servo_to(
            adapter,
            target_pos,
            gripper,
            steps=n_steps,
            gain=args.servo_gain,
        )
        total_reward += reward
        energy += phase_energy
        steps += phase_steps
        if error:
            errors.append(f"{name}: {error}")
            break
        if getattr(adapter, "last_done", False) or adapter.evaluate_success():
            break

    if not errors and not adapter.evaluate_success():
        target = adapter._position(target_name)
        if target is None:
            errors.append("target position unavailable after lift")
        else:
            place_specs: list[tuple[str, np.ndarray, float, int]] = [
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
            for name, target_pos, gripper, n_steps in place_specs:
                reward, phase_energy, phase_steps, error = servo_to(
                    adapter,
                    target_pos,
                    gripper,
                    steps=n_steps,
                    gain=args.servo_gain,
                )
                total_reward += reward
                energy += phase_energy
                steps += phase_steps
                if error:
                    errors.append(f"{name}: {error}")
                    break
                if getattr(adapter, "last_done", False) or adapter.evaluate_success():
                    break

    return {
        "success": bool(adapter.evaluate_real_success()),
        "total_reward": float(total_reward),
        "energy": float(energy),
        "steps": int(steps),
        "failure_reason": "; ".join(errors) if errors else None,
        "objects": objects,
        "object_name": obj_name,
        "target_name": target_name,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "task_name",
        "seed",
        "success",
        "total_reward",
        "initial_distance",
        "final_distance",
        "progress",
        "energy",
        "utility",
        "normalized_utility",
        "steps",
        "object_name",
        "target_name",
        "failure_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    succeeded = [r for r in rows if r.get("success")]
    failed = [r for r in rows if not r.get("success")]
    ci = (summary.get("confidence_intervals") or {}).get("success_rate") or {}
    lines = [
        "# LIBERO Scripted Policy Smoke Report",
        "",
        "This optional artifact runs a hand scripted OSC pick-place controller against LIBERO's real sparse success predicate. It is separate from the LIBERO rollout-pool WAM-lite artifact and should not be described as a learned policy or full LIBERO benchmark result.",
        "",
        "## Summary",
        "",
        f"- Available: `{summary.get('available')}`.",
        f"- Attempted: `{summary.get('attempted')}`.",
        f"- Verified: `{summary.get('verified')}`.",
        f"- Suite: `{summary.get('suite')}`.",
        f"- Episodes: `{summary.get('n_episodes')}` across `{summary.get('n_tasks')}` tasks and `{summary.get('n_seeds')}` seeds.",
        f"- Success rate: `{ci.get('mean')}` with bootstrap CI [`{ci.get('lo')}`, `{ci.get('hi')}`].",
        f"- Successful episodes: `{summary.get('n_successes')}`.",
        "",
        "## Successful Episodes",
        "",
    ]
    if succeeded:
        for row in succeeded:
            lines.append(
                f"- `{row.get('task_id')}` seed `{row.get('seed')}`: `{row.get('task_name')}`; "
                f"reward `{row.get('total_reward')}`, progress `{row.get('progress')}`."
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Failed Episodes", ""])
    if failed:
        for row in failed:
            lines.append(
                f"- `{row.get('task_id')}` seed `{row.get('seed')}`: `{row.get('task_name')}`; "
                f"final distance `{row.get('final_distance')}`, progress `{row.get('progress')}`."
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is a scripted sparse-success smoke, not a learned WAM policy and not a demonstration that the project solves LIBERO.",
            "- The controller uses object and target positions exposed by the simulator, so it is diagnostic benchmark evidence rather than deployable perception.",
            "- Failed tasks remain reported in the CSV/JSON artifact; the claim is limited to the measured success subset.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unavailable_summary(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "benchmark_libero_scripted_policy",
        "available": False,
        "attempted": True,
        "verified": False,
        "reason": reason,
        "suite": args.suite,
        "tasks": parse_task_ids(args.tasks, args.suite),
        "note": "Optional LIBERO sparse-success scripted policy smoke; unavailable in this interpreter.",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", nargs="+", default=[str(i) for i in range(10)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100])
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--camera-width", type=int, default=64)
    parser.add_argument("--camera-height", type=int, default=64)
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
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--energy-penalty", type=float, default=0.001)
    parser.add_argument("--min-success-rate", type=float, default=0.5)
    parser.add_argument("--min-episodes", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fail-on-low-success", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dirs()

    ok, reason = is_libero_available()
    if not ok:
        summary = unavailable_summary(reason, args)
        write_json(RESULTS / "benchmark_libero_scripted_policy.json", summary)
        write_report(summary, [], REPORTS / "libero_scripted_policy_report.md")
        print(reason)
        return

    task_ids = parse_task_ids(args.tasks, args.suite)
    rows: list[dict[str, Any]] = []
    adapter: LIBEROAdapter | None = None
    try:
        first_suite, first_idx = task_ids[0].split("/", 1)
        adapter = LIBEROAdapter(
            suite=first_suite,
            task_index=int(first_idx) if first_idx.isdigit() else 0,
            horizon=args.horizon,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            controller=args.controller,
            use_camera_obs=False,
            has_offscreen_renderer=False,
        )
        for task_id in task_ids:
            for seed in args.seeds:
                state = adapter.reset(int(seed), task_id=task_id)
                initial_distance = float(adapter.task_distance())
                outcome = run_pick_place_script(adapter, args)
                final_distance = float(adapter.task_distance())
                progress = float(initial_distance - final_distance)
                utility = float(
                    progress
                    + args.success_bonus * float(outcome["success"])
                    + float(outcome["total_reward"])
                    - args.energy_penalty * float(outcome["energy"])
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "task_name": str(getattr(adapter.task, "name", task_id)),
                        "seed": int(seed),
                        "success": bool(outcome["success"]),
                        "total_reward": float(outcome["total_reward"]),
                        "initial_distance": initial_distance,
                        "final_distance": final_distance,
                        "progress": progress,
                        "energy": float(outcome["energy"]),
                        "utility": utility,
                        "steps": int(outcome["steps"]),
                        "object_name": outcome.get("object_name"),
                        "target_name": outcome.get("target_name"),
                        "objects": outcome.get("objects"),
                        "failure_reason": outcome.get("failure_reason"),
                        "initial_state_dim": int(np.asarray(state).size),
                    }
                )
                print(
                    f"{task_id} seed={seed} success={bool(outcome['success'])} "
                    f"reward={float(outcome['total_reward']):.3f} progress={progress:.3f}"
                )
    except (LIBEROUnavailableError, RuntimeError, ValueError) as exc:
        summary = unavailable_summary(f"{type(exc).__name__}: {exc}", args)
        summary["partial_rows"] = rows
        write_json(RESULTS / "benchmark_libero_scripted_policy.json", summary)
        write_csv(RESULTS / "tables" / "benchmark_libero_scripted_policy_episodes.csv", rows)
        write_report(summary, rows, REPORTS / "libero_scripted_policy_report.md")
        if args.fail_on_low_success:
            raise
        return
    finally:
        if adapter is not None:
            adapter.close()

    normalized = normalize([float(r["utility"]) for r in rows])
    for row, norm in zip(rows, normalized):
        row["normalized_utility"] = norm

    successes = [float(r["success"]) for r in rows]
    rewards = [float(r["total_reward"]) for r in rows]
    utilities = [float(r["utility"]) for r in rows]
    progress_values = [float(r["progress"]) for r in rows]
    success_ci = bootstrap_ci(successes, seed=args.seed, n_boot=args.bootstrap_samples)
    reward_ci = bootstrap_ci(rewards, seed=args.seed + 1, n_boot=args.bootstrap_samples)
    utility_ci = bootstrap_ci(utilities, seed=args.seed + 2, n_boot=args.bootstrap_samples)
    progress_ci = bootstrap_ci(progress_values, seed=args.seed + 3, n_boot=args.bootstrap_samples)
    by_task: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        task_rows = [r for r in rows if r["task_id"] == task_id]
        task_successes = [float(r["success"]) for r in task_rows]
        by_task[task_id] = {
            "task_name": task_rows[0]["task_name"] if task_rows else task_id,
            "n": len(task_rows),
            "success_rate": float(np.mean(task_successes)) if task_successes else 0.0,
            "n_successes": int(np.sum(task_successes)) if task_successes else 0,
        }

    verified = (
        len(rows) >= int(args.min_episodes)
        and success_ci.get("mean") is not None
        and float(success_ci["mean"]) >= float(args.min_success_rate)
        and int(sum(successes)) > 0
    )
    summary = {
        "experiment": "benchmark_libero_scripted_policy",
        "available": True,
        "attempted": True,
        "verified": bool(verified),
        "suite": args.suite,
        "tasks": task_ids,
        "n_tasks": len(task_ids),
        "seeds": [int(s) for s in args.seeds],
        "n_seeds": len(args.seeds),
        "n_episodes": len(rows),
        "n_successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "confidence_intervals": {
            "success_rate": success_ci,
            "total_reward": reward_ci,
            "utility": utility_ci,
            "progress": progress_ci,
        },
        "by_task": by_task,
        "controller": {
            "type": "scripted_osc_pick_place",
            "controller": args.controller,
            "horizon": args.horizon,
            "safe_lift": args.safe_lift,
            "approach_z_offset": args.approach_z_offset,
            "grasp_z_offset": args.grasp_z_offset,
            "place_z_offset": args.place_z_offset,
            "grasp_offset": [args.grasp_offset_x, args.grasp_offset_y],
            "servo_gain": args.servo_gain,
        },
        "artifact_paths": {
            "json": "results/benchmark_libero_scripted_policy.json",
            "episodes_csv": "results/tables/benchmark_libero_scripted_policy_episodes.csv",
            "report": "reports/libero_scripted_policy_report.md",
        },
        "note": "Sparse-success scripted LIBERO smoke. This is not learned WAM policy performance and not full LIBERO validation.",
    }

    write_json(RESULTS / "benchmark_libero_scripted_policy.json", summary)
    write_csv(RESULTS / "tables" / "benchmark_libero_scripted_policy_episodes.csv", rows)
    write_report(summary, rows, REPORTS / "libero_scripted_policy_report.md")
    print(json.dumps(sanitize(summary), indent=2))
    if args.fail_on_low_success and not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
