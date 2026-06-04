from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd

from benchmark_robocasa_learned_wam import TARGETS, _features, fit_ridge, save_model
from wam_inference_value.benchmarks.robocasa_adapter import (
    RoboCasaAdapter,
    RoboCasaUnavailableError,
    is_robocasa_available,
)
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


DEFAULT_ENV_IDS = [
    "robocasa/PickPlaceCounterToCabinet",
    "robocasa/PickPlaceCounterToDrawer",
    "robocasa/PickPlaceCounterToMicrowave",
]
DEFAULT_N_VALUES = [1, 2, 4, 8]


def _artifact_paths(output_tag: str = "") -> dict[str, Path]:
    if output_tag:
        table_prefix = f"benchmark_robocasa_{output_tag}"
        return {
            "summary": results_dir() / f"{table_prefix}_wam.json",
            "model": results_dir() / "models" / f"{table_prefix}_ridge_wam.npz",
            "curves": results_dir() / "tables" / f"{table_prefix}_curves.csv",
            "exact": results_dir() / "tables" / f"{table_prefix}_exact_law.csv",
            "train_val": results_dir() / "tables" / f"{table_prefix}_train_validation.csv",
            "eval": results_dir() / "tables" / f"{table_prefix}_eval_rollouts.csv",
            "task_metrics": results_dir() / "tables" / f"{table_prefix}_task_metrics.csv",
            "seed_metrics": results_dir() / "tables" / f"{table_prefix}_seed_metrics.csv",
            "report": ROOT / "reports" / f"robocasa_{output_tag}_wam_report.md",
        }
    return {
        "summary": results_dir() / "benchmark_robocasa_multitask_wam.json",
        "model": results_dir() / "models" / "benchmark_robocasa_multitask_ridge_wam.npz",
        "curves": results_dir() / "tables" / "benchmark_robocasa_multitask_curves.csv",
        "exact": results_dir() / "tables" / "benchmark_robocasa_multitask_exact_law.csv",
        "train_val": results_dir() / "tables" / "benchmark_robocasa_multitask_train_validation.csv",
        "eval": results_dir() / "tables" / "benchmark_robocasa_multitask_eval_rollouts.csv",
        "task_metrics": results_dir() / "tables" / "benchmark_robocasa_multitask_task_metrics.csv",
        "seed_metrics": results_dir() / "tables" / "benchmark_robocasa_multitask_seed_metrics.csv",
        "report": ROOT / "reports" / "robocasa_multitask_wam_report.md",
    }


@dataclass
class TaskData:
    env_id: str
    task_index: int
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    eval_x: np.ndarray
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]


def _task_one_hot(task_index: int, n_tasks: int, rows: int) -> np.ndarray:
    one_hot = np.zeros((int(rows), int(n_tasks)), dtype=float)
    one_hot[:, int(task_index)] = 1.0
    return one_hot


def _task_features(adapter: RoboCasaAdapter, state: np.ndarray, actions: np.ndarray, task_index: int, n_tasks: int) -> np.ndarray:
    base = _features(adapter, state, actions)
    return np.concatenate([base, _task_one_hot(task_index, n_tasks, len(base))], axis=1)


def _rows_from_records(
    records: list[dict[str, Any]],
    *,
    env_id: str,
    task_index: int,
    split: str,
    state_id: int,
    seed: int,
    x: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rollout_id, record in enumerate(records):
        row = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in record.items()}
        row.update(
            {
                "env_id": env_id,
                "task_index": int(task_index),
                "split": split,
                "state_id": int(state_id),
                "seed": int(seed),
                "rollout_id": int(rollout_id),
                "feature_dim": int(x.shape[1]),
            }
        )
        rows.append(row)
    return rows


def _collect_split(
    adapter: RoboCasaAdapter,
    *,
    env_id: str,
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
        x = _task_features(adapter, state, actions, task_index, n_tasks)
        y = np.asarray([[float(r[name]) for name in TARGETS] for r in records], dtype=float)
        xs.append(x)
        ys.append(y)
        rows.extend(_rows_from_records(records, env_id=env_id, task_index=task_index, split=split, state_id=state_id, seed=state_seed, x=x))
    return np.vstack(xs), np.vstack(ys), rows


def collect_task_data(env_id: str, task_index: int, n_tasks: int, args: argparse.Namespace) -> TaskData:
    adapter = RoboCasaAdapter(
        env_id=env_id,
        split=args.split,
        horizon=args.horizon,
        camera_width=args.camera_size,
        camera_height=args.camera_size,
        success_bonus=args.success_bonus,
        energy_penalty=args.energy_penalty,
    )
    try:
        offset = 100_003 * (task_index + 1)
        train_x, train_y, train_rows = _collect_split(
            adapter,
            env_id=env_id,
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
            env_id=env_id,
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
            env_id=env_id,
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
        env_id=env_id,
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


def _write_report(summary: dict[str, Any]) -> None:
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not summary.get("available"):
        lines = [
            "# RoboCasa Multi-Task WAM Report",
            "",
            "- status: unavailable",
            f"- reason: {summary.get('reason')}",
        ]
    else:
        n_values = [int(n) for n in (summary.get("n_values") or [])]
        n_max = max(n_values) if n_values else 8
        ci_key = f"best_learned_minus_random_N{n_max}"
        ci = (summary.get("confidence_intervals") or {}).get(ci_key) or (
            summary.get("confidence_intervals") or {}
        ).get("best_learned_minus_random_N8") or {}
        metrics = summary.get("model_metrics") or {}
        lines = [
            "# RoboCasa Multi-Task WAM Report",
            "",
            f"- status: `{'verified' if summary.get('verified') else 'attempted_not_promoted'}`",
            f"- tasks: `{summary.get('env_ids')}`",
            f"- train samples: `{summary.get('train_samples')}`",
            f"- validation samples: `{summary.get('validation_samples')}`",
            f"- eval samples: `{summary.get('eval_samples')}`",
            f"- eval rollout pools: `{summary.get('eval_rollout_pools')}`",
            f"- exact-law utility MAE: `{summary.get('exact_law_utility_mae')}`",
            f"- validation utility correlation: `{metrics.get('utility_corr')}`",
            f"- validation learned-physics correlation: `{metrics.get('learned_physics_score_corr')}`",
            f"- promoted learned scorer: `{summary.get('promoted_scorer')}`",
            f"- promoted scorer minus random N{n_max} CI: `{ci}`",
            "",
            "This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.",
        ]
    report_path = Path(summary.get("report_path") or _artifact_paths(str(summary.get("output_tag") or ""))["report"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unavailable(reason: str, output_tag: str = "") -> dict[str, Any]:
    ensure_result_dirs()
    paths = _artifact_paths(output_tag)
    summary = {
        "experiment": paths["summary"].stem,
        "attempted": True,
        "available": False,
        "verified": False,
        "output_tag": output_tag,
        "reason": reason,
        "report_path": str(paths["report"]),
    }
    write_json(paths["summary"], summary)
    _write_report(summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    ok, reason = is_robocasa_available()
    if not ok:
        return _unavailable(reason, getattr(args, "output_tag", ""))
    paths = _artifact_paths(str(getattr(args, "output_tag", "") or ""))

    env_ids = [str(e) for e in args.env_ids]
    n_tasks = len(env_ids)
    if n_tasks < 2:
        raise ValueError("multi-task RoboCasa run needs at least two env ids")
    n_values = [int(n) for n in args.n_values]
    max_n = max(n_values)

    task_data: list[TaskData] = []
    unavailable: list[dict[str, str]] = []
    for task_index, env_id in enumerate(env_ids):
        try:
            print(f"[robocasa-multitask] collecting {env_id} ({task_index + 1}/{n_tasks})", flush=True)
            task_data.append(collect_task_data(env_id, task_index, n_tasks, args))
            print(f"[robocasa-multitask] finished {env_id}", flush=True)
        except RoboCasaUnavailableError as exc:
            unavailable.append({"env_id": env_id, "reason": str(exc)})
        except Exception as exc:  # pragma: no cover - optional benchmark failures are artifacted
            unavailable.append({"env_id": env_id, "reason": f"{type(exc).__name__}: {exc}"})
    if len(task_data) < args.min_tasks:
        summary = {
            "experiment": paths["summary"].stem,
            "attempted": True,
            "available": bool(task_data),
            "verified": False,
            "env_ids": [d.env_id for d in task_data],
            "unavailable": unavailable,
            "reason": f"only {len(task_data)} task(s) ran; min_tasks={args.min_tasks}",
        }
        summary["output_tag"] = str(getattr(args, "output_tag", "") or "")
        summary["report_path"] = str(paths["report"])
        write_json(paths["summary"], summary)
        _write_report(summary)
        return summary

    train_x = np.vstack([d.train_x for d in task_data])
    train_y = np.vstack([d.train_y for d in task_data])
    val_x = np.vstack([d.val_x for d in task_data])
    val_y = np.vstack([d.val_y for d in task_data])
    model = fit_ridge(train_x, train_y, alpha=args.ridge_alpha)
    val_pred = model.predict(val_x)
    model_metrics = _model_metrics(val_pred, val_y, args)

    model_path = paths["model"]
    save_model(
        model,
        model_path,
        {
            "env_ids": [d.env_id for d in task_data],
            "model_type": "task_conditioned_ridge_state_action_sequence_wam",
            "train_states_per_task": int(args.train_states),
            "train_rollouts": int(args.train_rollouts),
            "horizon": int(args.horizon),
        },
    )

    train_val_rows: list[dict[str, Any]] = []
    for data in task_data:
        train_val_rows.extend(data.train_rows)
        train_val_rows.extend(data.val_rows)

    eval_detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    for data in task_data:
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
        task_eval = pd.DataFrame(eval_detail_rows).query("env_id == @data.env_id")
        for (seed, state_id), sub in task_eval.groupby(["seed", "state_id"], dropna=False):
            real_utility = sub["utility"].to_numpy(dtype=float)
            norm_utility = normalized_utility(real_utility)
            success = sub["success"].astype(float).to_numpy()
            energy = sub["energy"].to_numpy(dtype=float)
            progress = sub["progress"].to_numpy(dtype=float)
            final_distance = sub["final_distance"].to_numpy(dtype=float)
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
                "oracle_real_utility": real_utility,
            }
            for scorer, scores in scorers.items():
                raw_curve = utility_best_of_n_finite(scores, real_utility, n_values)
                norm_curve = utility_best_of_n_finite(scores, norm_utility, n_values)
                succ_curve = utility_best_of_n_finite(scores, success, n_values)
                for n in n_values:
                    curve_rows.append(
                        {
                            "env_id": data.env_id,
                            "task_index": int(data.task_index),
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
                                "env_id": data.env_id,
                                "task_index": int(data.task_index),
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
    eval_detail = pd.DataFrame(eval_detail_rows)
    curves_path = paths["curves"]
    exact_path = paths["exact"]
    train_val_path = paths["train_val"]
    eval_path = paths["eval"]
    task_metrics_path = paths["task_metrics"]
    seed_metrics_path = paths["seed_metrics"]
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    train_val.to_csv(train_val_path, index=False)
    eval_detail.to_csv(eval_path, index=False)

    task_metrics = []
    start = 0
    for data in task_data:
        stop = start + len(data.val_x)
        task_metrics.append({"env_id": data.env_id, **_model_metrics(val_pred[start:stop], data.val_y, args)})
        start = stop
    pd.DataFrame(task_metrics).to_csv(task_metrics_path, index=False)

    seed_metrics = []
    high_n = curves[curves["N"] == max_n].copy()
    for (env_id, seed, state_id), sub in high_n.groupby(["env_id", "seed", "state_id"], dropna=False):
        by_scorer = sub.groupby("scorer")["normalized_real_utility"].mean()
        learned_deltas = {
            "learned_wam": float(by_scorer.get("learned_wam", np.nan) - by_scorer.get("random", np.nan)),
            "learned_physics_score": float(by_scorer.get("learned_physics_score", np.nan) - by_scorer.get("random", np.nan)),
            "learned_energy_regularized": float(by_scorer.get("learned_energy_regularized", np.nan) - by_scorer.get("random", np.nan)),
        }
        best_name = max(learned_deltas, key=lambda k: learned_deltas[k] if np.isfinite(learned_deltas[k]) else -np.inf)
        seed_metrics.append(
            {
                "env_id": env_id,
                "seed": int(seed),
                "state_id": int(state_id),
                f"learned_wam_minus_random_N{max_n}": learned_deltas["learned_wam"],
                f"learned_physics_minus_random_N{max_n}": learned_deltas["learned_physics_score"],
                f"learned_energy_regularized_minus_random_N{max_n}": learned_deltas["learned_energy_regularized"],
                f"best_learned_minus_random_N{max_n}": learned_deltas[best_name],
                "best_learned_scorer": best_name,
                f"oracle_minus_random_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get("random", np.nan)),
                f"oracle_minus_best_learned_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get(best_name, np.nan)),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_df.to_csv(seed_metrics_path, index=False)
    confidence_intervals = {
        key: ci95(seed_df[key].to_numpy())
        for key in seed_df.columns
        if key not in {"env_id", "seed", "state_id", "best_learned_scorer"}
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
        and model_metrics["learned_physics_score_corr"] > 0.0
    )
    summary = {
        "experiment": paths["summary"].stem,
        "attempted": True,
        "available": True,
        "verified": bool(verified),
        "output_tag": str(getattr(args, "output_tag", "") or ""),
        "env_ids": [d.env_id for d in task_data],
        "unavailable": unavailable,
        "split": args.split,
        "model_path": str(model_path),
        "model_type": "task_conditioned_ridge_state_action_sequence_wam",
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
        "task_metrics": task_metrics,
        "exact_law_utility_mae": exact_mae,
        "confidence_intervals": confidence_intervals,
        "promoted_scorer": promoted_scorer,
        "learned_energy_regularizer": float(args.learned_energy_regularizer),
        "curves_path": str(curves_path),
        "exact_path": str(exact_path),
        "data_path": str(train_val_path),
        "eval_path": str(eval_path),
        "task_metrics_path": str(task_metrics_path),
        "seed_metrics_path": str(seed_metrics_path),
        "report_path": str(paths["report"]),
        "note": "Optional task conditioned RoboCasa learned WAM-lite across multiple kitchen task IDs; promoted only when a learned scorer beats random with positive heldout CI.",
    }
    write_json(paths["summary"], summary)
    _write_report(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="*", default=DEFAULT_ENV_IDS)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--train-states", type=int, default=2)
    parser.add_argument("--train-rollouts", type=int, default=8)
    parser.add_argument("--val-states", type=int, default=1)
    parser.add_argument("--val-rollouts", type=int, default=8)
    parser.add_argument("--eval-states", type=int, default=2)
    parser.add_argument("--eval-rollouts", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--camera-size", type=int, default=16)
    parser.add_argument("--mc-trials", type=int, default=1500)
    parser.add_argument("--n-values", nargs="*", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--energy-penalty", type=float, default=0.01)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--learned-energy-regularizer", type=float, default=0.03)
    parser.add_argument("--max-exact-mae", type=float, default=0.02)
    parser.add_argument("--min-tasks", type=int, default=3)
    parser.add_argument("--min-eval-pools", type=int, default=6)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
