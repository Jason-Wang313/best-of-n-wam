from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wam_inference_value.benchmarks.libero_adapter import LIBEROAdapter, LIBEROUnavailableError, is_libero_available


TARGETS = ("utility", "progress", "final_distance", "energy", "success")
DEFAULT_TASKS = ["libero_spatial/0"]
DEFAULT_N_VALUES = [1, 2, 4, 8]
RESULTS = ROOT / "results"


def results_dir() -> Path:
    return RESULTS


def ensure_result_dirs() -> None:
    for path in [RESULTS, RESULTS / "tables", RESULTS / "models", ROOT / "reports"]:
        path.mkdir(parents=True, exist_ok=True)


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def ci95(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "stderr": None, "ci95": None, "lo": None, "hi": None}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    stderr = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    half_width = float(1.96 * stderr)
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ci95": half_width,
        "lo": mean - half_width,
        "hi": mean + half_width,
    }


def normalized_utility(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo <= 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


@dataclass
class RidgeWAM:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    target_names: tuple[str, ...]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = (x - self.mean) / self.scale
        z = np.column_stack([np.ones(len(z)), z])
        return z @ self.weights


@dataclass
class TaskSpec:
    suite: str
    task_index: int

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task_index}"


@dataclass
class TaskData:
    task_key: str
    task_name: str
    task_index: int
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    eval_x: np.ndarray
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]


def parse_task_spec(raw: str) -> TaskSpec:
    if "/" not in raw:
        return TaskSpec(suite="libero_spatial", task_index=int(raw))
    suite, idx = raw.split("/", 1)
    return TaskSpec(suite=suite, task_index=int(idx))


def _state_summary(adapter: LIBEROAdapter, state: np.ndarray) -> np.ndarray:
    adapter.set_state(state)
    feat = np.asarray(adapter.feature_state(), dtype=float).reshape(-1)
    if feat.size == 0:
        return np.zeros(8, dtype=float)
    q = np.quantile(feat, [0.0, 0.1, 0.5, 0.9, 1.0])
    return np.asarray(
        [
            adapter.object_distance(),
            float(np.mean(feat)),
            float(np.std(feat)),
            float(np.linalg.norm(feat) / np.sqrt(max(1, feat.size))),
            *[float(v) for v in q],
        ],
        dtype=float,
    )


def _action_features(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=float)
    flat = actions.reshape(actions.shape[0], -1)
    energy = np.sum(flat * flat, axis=1, keepdims=True)
    abs_sum = np.sum(np.abs(flat), axis=1, keepdims=True)
    mean = np.mean(actions, axis=1)
    std = np.std(actions, axis=1)
    first = actions[:, 0, :]
    last = actions[:, -1, :]
    max_abs = np.max(np.abs(actions), axis=1)
    return np.concatenate([flat, energy, abs_sum, mean, std, first, last, max_abs], axis=1)


def _one_hot(task_index: int, n_tasks: int, rows: int) -> np.ndarray:
    out = np.zeros((int(rows), int(n_tasks)), dtype=float)
    out[:, int(task_index)] = 1.0
    return out


def _features(adapter: LIBEROAdapter, state: np.ndarray, actions: np.ndarray, task_index: int, n_tasks: int) -> np.ndarray:
    state_part = _state_summary(adapter, state)
    action_part = _action_features(actions)
    state_tile = np.repeat(state_part.reshape(1, -1), len(action_part), axis=0)
    return np.concatenate([state_tile, action_part, _one_hot(task_index, n_tasks, len(action_part))], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeWAM:
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
    return RidgeWAM(mean=mean, scale=scale, weights=weights, target_names=TARGETS)


def save_model(model: RidgeWAM, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=model.mean,
        scale=model.scale,
        weights=model.weights,
        target_names=np.asarray(model.target_names, dtype=object),
        metadata=np.asarray([metadata], dtype=object),
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _model_metrics(pred: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    learned_physics = pred[:, 1] + args.success_bonus * pred[:, 4] - args.energy_penalty * pred[:, 3]
    return {
        "utility_mae": float(np.mean(np.abs(pred[:, 0] - y[:, 0]))),
        "utility_corr": _corr(pred[:, 0], y[:, 0]),
        "learned_physics_score_corr": _corr(learned_physics, y[:, 0]),
        "progress_mae": float(np.mean(np.abs(pred[:, 1] - y[:, 1]))),
        "progress_corr": _corr(pred[:, 1], y[:, 1]),
        "final_distance_mae": float(np.mean(np.abs(pred[:, 2] - y[:, 2]))),
        "energy_mae": float(np.mean(np.abs(pred[:, 3] - y[:, 3]))),
        "success_mae": float(np.mean(np.abs(pred[:, 4] - y[:, 4]))),
    }


def _rows_from_records(
    records: list[dict[str, Any]],
    *,
    task_key: str,
    task_name: str,
    task_index: int,
    split: str,
    state_id: int,
    seed: int,
    feature_dim: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rollout_id, record in enumerate(records):
        row = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in record.items()}
        row.update(
            {
                "task_key": task_key,
                "task_name": task_name,
                "task_index": int(task_index),
                "split": split,
                "state_id": int(state_id),
                "seed": int(seed),
                "rollout_id": int(rollout_id),
                "feature_dim": int(feature_dim),
            }
        )
        rows.append(row)
    return rows


def _collect_split(
    adapter: LIBEROAdapter,
    *,
    task_key: str,
    task_name: str,
    task_index: int,
    n_tasks: int,
    states: int,
    rollouts: int,
    horizon: int,
    seed: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for state_id in range(int(states)):
        state_seed = int(seed + 10_007 * state_id)
        state = adapter.reset_task(seed=state_seed)
        pool = adapter.sample_rollouts(
            initial_state=state,
            n_rollouts=rollouts,
            horizon=horizon,
            seed=state_seed + 17,
        )
        actions = np.asarray(pool["actions"], dtype=float)
        records = pool["records"]
        x = _features(adapter, state, actions, task_index, n_tasks)
        y = np.asarray([[float(r[name]) for name in TARGETS] for r in records], dtype=float)
        xs.append(x)
        ys.append(y)
        rows.extend(
            _rows_from_records(
                records,
                task_key=task_key,
                task_name=task_name,
                task_index=task_index,
                split=split,
                state_id=state_id,
                seed=state_seed,
                feature_dim=x.shape[1],
            )
        )
    return np.vstack(xs), np.vstack(ys), rows


def collect_task_data(spec: TaskSpec, task_index: int, n_tasks: int, args: argparse.Namespace) -> TaskData:
    adapter = LIBEROAdapter(
        suite=spec.suite,
        task_index=spec.task_index,
        horizon=args.horizon,
        camera_width=args.camera_size,
        camera_height=args.camera_size,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        action_scale=args.action_scale,
        gripper_scale=args.gripper_scale,
        target_weight=args.target_weight,
        eef_weight=args.eef_weight,
        success_bonus=args.success_bonus,
        reward_weight=args.reward_weight,
        energy_penalty=args.energy_penalty,
    )
    try:
        offset = 100_003 * (task_index + 1)
        train_x, train_y, train_rows = _collect_split(
            adapter,
            task_key=spec.key,
            task_name=adapter.task_name,
            task_index=task_index,
            n_tasks=n_tasks,
            states=args.train_states,
            rollouts=args.train_rollouts,
            horizon=args.horizon,
            seed=args.seed + offset,
            split="train",
        )
        val_x, val_y, val_rows = _collect_split(
            adapter,
            task_key=spec.key,
            task_name=adapter.task_name,
            task_index=task_index,
            n_tasks=n_tasks,
            states=args.val_states,
            rollouts=args.val_rollouts,
            horizon=args.horizon,
            seed=args.seed + 500_000 + offset,
            split="validation",
        )
        eval_x, _, eval_rows = _collect_split(
            adapter,
            task_key=spec.key,
            task_name=adapter.task_name,
            task_index=task_index,
            n_tasks=n_tasks,
            states=args.eval_states,
            rollouts=args.eval_rollouts,
            horizon=args.horizon,
            seed=args.seed + 900_000 + offset,
            split="eval",
        )
    finally:
        adapter.close()
    return TaskData(
        task_key=spec.key,
        task_name=adapter.task_name,
        task_index=int(task_index),
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        eval_x=eval_x,
        train_rows=train_rows,
        val_rows=val_rows,
        eval_rows=eval_rows,
    )


def _write_report(summary: dict[str, Any]) -> None:
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not summary.get("available"):
        lines = [
            "# LIBERO WAM Report",
            "",
            "- status: unavailable",
            f"- reason: {summary.get('reason')}",
        ]
    else:
        ci = (summary.get("confidence_intervals") or {}).get(f"best_learned_minus_random_N{max(summary.get('n_values') or [8])}") or {}
        metrics = summary.get("model_metrics") or {}
        lines = [
            "# LIBERO WAM Report",
            "",
            f"- status: `{'verified' if summary.get('verified') else 'attempted_not_promoted'}`",
            f"- tasks: `{summary.get('tasks')}`",
            f"- train samples: `{summary.get('train_samples')}`",
            f"- validation samples: `{summary.get('validation_samples')}`",
            f"- eval samples: `{summary.get('eval_samples')}`",
            f"- eval rollout pools: `{summary.get('eval_rollout_pools')}`",
            f"- exact-law utility MAE: `{summary.get('exact_law_utility_mae')}`",
            f"- validation utility correlation: `{metrics.get('utility_corr')}`",
            f"- validation learned-physics correlation: `{metrics.get('learned_physics_score_corr')}`",
            f"- promoted learned scorer: `{summary.get('promoted_scorer')}`",
            f"- promoted scorer minus random CI: `{ci}`",
            "",
            "This is an optional LIBERO state/action-sequence WAM-lite artifact. The dense utility is task local progress plus sparse success/reward, so it should be cited as LIBERO rollout-pool validation rather than solved-task policy performance.",
        ]
    (report_dir / "libero_wam_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unavailable(reason: str) -> dict[str, Any]:
    ensure_result_dirs()
    summary = {
        "experiment": "benchmark_libero_wam",
        "attempted": True,
        "available": False,
        "verified": False,
        "reason": reason,
    }
    write_json(results_dir() / "benchmark_libero_wam.json", summary)
    _write_report(summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    ok, reason = is_libero_available()
    if not ok:
        return _unavailable(reason)

    import pandas as pd
    from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite

    specs = [parse_task_spec(raw) for raw in args.tasks]
    n_tasks = len(specs)
    n_values = [int(n) for n in args.n_values]
    max_n = max(n_values)

    task_data: list[TaskData] = []
    unavailable: list[dict[str, str]] = []
    for idx, spec in enumerate(specs):
        try:
            print(f"[libero] collecting {spec.key} ({idx + 1}/{n_tasks})", flush=True)
            task_data.append(collect_task_data(spec, idx, n_tasks, args))
            print(f"[libero] finished {spec.key}", flush=True)
        except LIBEROUnavailableError as exc:
            unavailable.append({"task": spec.key, "reason": str(exc)})
        except Exception as exc:  # pragma: no cover - optional benchmark failures are artifacted
            unavailable.append({"task": spec.key, "reason": f"{type(exc).__name__}: {exc}"})
    if len(task_data) < args.min_tasks:
        summary = {
            "experiment": "benchmark_libero_wam",
            "attempted": True,
            "available": bool(task_data),
            "verified": False,
            "tasks": [d.task_key for d in task_data],
            "unavailable": unavailable,
            "reason": f"only {len(task_data)} task(s) ran; min_tasks={args.min_tasks}",
        }
        write_json(results_dir() / "benchmark_libero_wam.json", summary)
        _write_report(summary)
        return summary

    train_x = np.vstack([d.train_x for d in task_data])
    train_y = np.vstack([d.train_y for d in task_data])
    val_x = np.vstack([d.val_x for d in task_data])
    val_y = np.vstack([d.val_y for d in task_data])
    model = fit_ridge(train_x, train_y, alpha=args.ridge_alpha)
    val_pred = model.predict(val_x)
    model_metrics = _model_metrics(val_pred, val_y, args)
    model_path = results_dir() / "models" / "benchmark_libero_ridge_wam.npz"
    save_model(
        model,
        model_path,
        {
            "tasks": [d.task_key for d in task_data],
            "model_type": "libero_ridge_state_action_sequence_wam",
            "horizon": int(args.horizon),
            "train_states_per_task": int(args.train_states),
            "train_rollouts": int(args.train_rollouts),
        },
    )

    train_val_rows: list[dict[str, Any]] = []
    eval_detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    for data in task_data:
        train_val_rows.extend(data.train_rows)
        train_val_rows.extend(data.val_rows)
        pred = model.predict(data.eval_x)
        for row, p in zip(data.eval_rows, pred):
            out = dict(row)
            out.update(
                {
                    "predicted_utility": float(p[0]),
                    "predicted_progress": float(p[1]),
                    "predicted_final_distance": float(p[2]),
                    "predicted_energy": float(p[3]),
                    "predicted_success": float(p[4]),
                    "learned_physics_score": float(p[1] + args.success_bonus * p[4] - args.energy_penalty * p[3]),
                }
            )
            eval_detail_rows.append(out)

    eval_detail = pd.DataFrame(eval_detail_rows)
    for (task_key, seed, state_id), sub in eval_detail.groupby(["task_key", "seed", "state_id"], dropna=False):
        real_utility = sub["utility"].to_numpy(dtype=float)
        norm_utility = normalized_utility(real_utility)
        success = sub["success"].astype(float).to_numpy()
        energy = sub["energy"].to_numpy(dtype=float)
        progress = sub["progress"].to_numpy(dtype=float)
        final_distance = sub["final_distance"].to_numpy(dtype=float)
        reward = sub["total_reward"].to_numpy(dtype=float)
        pred_utility = sub["predicted_utility"].to_numpy(dtype=float)
        pred_physics = sub["learned_physics_score"].to_numpy(dtype=float)
        pred_progress = sub["predicted_progress"].to_numpy(dtype=float)
        pred_success = sub["predicted_success"].to_numpy(dtype=float)
        rng = np.random.default_rng(int(seed) + 37 * int(state_id))
        scorers = {
            "random": rng.normal(size=len(sub)),
            "learned_wam": pred_utility,
            "learned_physics_score": pred_physics,
            "learned_energy_regularized": pred_utility - args.learned_energy_regularizer * energy,
            "predicted_progress": pred_progress,
            "predicted_success": pred_success,
            "low_energy": -energy,
            "distance_progress": progress - final_distance,
            "benchmark_reward": reward,
            "oracle_real_utility": real_utility,
        }
        for scorer, scores in scorers.items():
            raw_curve = utility_best_of_n_finite(scores, real_utility, n_values)
            norm_curve = utility_best_of_n_finite(scores, norm_utility, n_values)
            succ_curve = utility_best_of_n_finite(scores, success, n_values)
            for n in n_values:
                curve_rows.append(
                    {
                        "task_key": task_key,
                        "seed": int(seed),
                        "state_id": int(state_id),
                        "scorer": scorer,
                        "N": int(n),
                        "real_utility": float(raw_curve[n]),
                        "normalized_real_utility": float(norm_curve[n]),
                        "success": float(succ_curve[n]),
                    }
                )
                if scorer in {"learned_wam", "learned_physics_score", "learned_energy_regularized", "oracle_real_utility"}:
                    mc = simulate_best_of_n(scores, real_utility, n, args.mc_trials, int(seed) + 100 * n + 13 * int(state_id))
                    exact_rows.append(
                        {
                            "task_key": task_key,
                            "seed": int(seed),
                            "state_id": int(state_id),
                            "scorer": scorer,
                            "N": int(n),
                            "utility_exact": float(raw_curve[n]),
                            "utility_mc": float(mc),
                            "utility_abs_error": float(abs(raw_curve[n] - mc)),
                        }
                    )

    curves = pd.DataFrame(curve_rows)
    exact = pd.DataFrame(exact_rows)
    train_val = pd.DataFrame(train_val_rows)
    curves_path = results_dir() / "tables" / "benchmark_libero_curves.csv"
    exact_path = results_dir() / "tables" / "benchmark_libero_exact_law.csv"
    train_val_path = results_dir() / "tables" / "benchmark_libero_train_validation.csv"
    eval_path = results_dir() / "tables" / "benchmark_libero_eval_rollouts.csv"
    seed_path = results_dir() / "tables" / "benchmark_libero_seed_metrics.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    train_val.to_csv(train_val_path, index=False)
    eval_detail.to_csv(eval_path, index=False)

    seed_metrics = []
    high_n = curves[curves["N"] == max_n].copy()
    for (task_key, seed, state_id), sub in high_n.groupby(["task_key", "seed", "state_id"], dropna=False):
        by_scorer = sub.groupby("scorer")["normalized_real_utility"].mean()
        learned_deltas = {
            "learned_wam": float(by_scorer.get("learned_wam", np.nan) - by_scorer.get("random", np.nan)),
            "learned_physics_score": float(by_scorer.get("learned_physics_score", np.nan) - by_scorer.get("random", np.nan)),
            "learned_energy_regularized": float(by_scorer.get("learned_energy_regularized", np.nan) - by_scorer.get("random", np.nan)),
        }
        best_name = max(learned_deltas, key=lambda k: learned_deltas[k] if np.isfinite(learned_deltas[k]) else -np.inf)
        seed_metrics.append(
            {
                "task_key": task_key,
                "seed": int(seed),
                "state_id": int(state_id),
                f"learned_wam_minus_random_N{max_n}": learned_deltas["learned_wam"],
                f"learned_physics_minus_random_N{max_n}": learned_deltas["learned_physics_score"],
                f"learned_energy_regularized_minus_random_N{max_n}": learned_deltas["learned_energy_regularized"],
                f"best_learned_minus_random_N{max_n}": learned_deltas[best_name],
                "best_learned_scorer": best_name,
                f"oracle_minus_random_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get("random", np.nan)),
                f"oracle_minus_best_learned_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get(best_name, np.nan)),
                f"benchmark_reward_minus_random_N{max_n}": float(by_scorer.get("benchmark_reward", np.nan) - by_scorer.get("random", np.nan)),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_df.to_csv(seed_path, index=False)
    confidence_intervals = {
        key: ci95(seed_df[key].to_numpy())
        for key in seed_df.columns
        if key not in {"task_key", "seed", "state_id", "best_learned_scorer"}
    }
    learned_ci_keys = {
        "learned_wam": f"learned_wam_minus_random_N{max_n}",
        "learned_physics_score": f"learned_physics_minus_random_N{max_n}",
        "learned_energy_regularized": f"learned_energy_regularized_minus_random_N{max_n}",
    }
    promoted_scorer = max(
        learned_ci_keys,
        key=lambda scorer: (confidence_intervals.get(learned_ci_keys[scorer]) or {}).get("lo")
        if (confidence_intervals.get(learned_ci_keys[scorer]) or {}).get("lo") is not None
        else -np.inf,
    )
    confidence_intervals[f"best_learned_minus_random_N{max_n}"] = confidence_intervals[learned_ci_keys[promoted_scorer]]
    exact_mae = float(exact["utility_abs_error"].mean()) if not exact.empty else None
    promoted_ci = confidence_intervals.get(f"best_learned_minus_random_N{max_n}") or {}
    verified = (
        len(task_data) >= args.min_tasks
        and exact_mae is not None
        and exact_mae < args.max_exact_mae
        and promoted_ci.get("n", 0) >= args.min_eval_pools
        and promoted_ci.get("lo") is not None
        and promoted_ci["lo"] > 0.0
        and model_metrics["utility_corr"] > 0.0
    )
    summary = {
        "experiment": "benchmark_libero_wam",
        "attempted": True,
        "available": True,
        "verified": bool(verified),
        "tasks": [d.task_key for d in task_data],
        "task_names": [d.task_name for d in task_data],
        "unavailable": unavailable,
        "model_path": str(model_path),
        "model_type": "libero_ridge_state_action_sequence_wam",
        "train_states_per_task": int(args.train_states),
        "train_rollouts": int(args.train_rollouts),
        "train_samples": int(len(train_x)),
        "validation_states_per_task": int(args.val_states),
        "validation_rollouts": int(args.val_rollouts),
        "validation_samples": int(len(val_x)),
        "eval_states_per_task": int(args.eval_states),
        "eval_rollouts": int(args.eval_rollouts),
        "eval_samples": int(len(eval_detail)),
        "eval_rollout_pools": int(len(seed_df)),
        "horizon": int(args.horizon),
        "n_values": n_values,
        "model_metrics": model_metrics,
        "exact_law_utility_mae": exact_mae,
        "confidence_intervals": confidence_intervals,
        "promoted_scorer": promoted_scorer,
        "curves_path": str(curves_path),
        "exact_path": str(exact_path),
        "data_path": str(train_val_path),
        "eval_path": str(eval_path),
        "seed_metrics_path": str(seed_path),
        "note": "Optional LIBERO rollout-pool WAM-lite validation with dense progress utility; not a solved-task policy-performance claim.",
    }
    write_json(results_dir() / "benchmark_libero_wam.json", summary)
    _write_report(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--train-states", type=int, default=4)
    parser.add_argument("--train-rollouts", type=int, default=16)
    parser.add_argument("--val-states", type=int, default=2)
    parser.add_argument("--val-rollouts", type=int, default=16)
    parser.add_argument("--eval-states", type=int, default=5)
    parser.add_argument("--eval-rollouts", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--mc-trials", type=int, default=1500)
    parser.add_argument("--n-values", nargs="*", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--action-scale", type=float, default=0.65)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--eef-weight", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--energy-penalty", type=float, default=0.01)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--learned-energy-regularizer", type=float, default=0.03)
    parser.add_argument("--max-exact-mae", type=float, default=0.03)
    parser.add_argument("--min-tasks", type=int, default=1)
    parser.add_argument("--min-eval-pools", type=int, default=5)
    args = parser.parse_args()
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
