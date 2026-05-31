from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
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
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8]
TARGETS = ("utility", "progress", "final_distance", "energy", "success")


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


def _state_summary(adapter: RoboCasaAdapter, state: np.ndarray) -> np.ndarray:
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


def _features(adapter: RoboCasaAdapter, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state_part = _state_summary(adapter, state)
    action_part = _action_features(actions)
    state_tile = np.repeat(state_part.reshape(1, -1), len(action_part), axis=0)
    return np.concatenate([state_tile, action_part], axis=1)


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


def collect_dataset(
    adapter: RoboCasaAdapter,
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
        x = _features(adapter, state, actions)
        y = np.asarray([[float(r[name]) for name in TARGETS] for r in records], dtype=float)
        xs.append(x)
        ys.append(y)
        for rollout_id, r in enumerate(records):
            row = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in r.items()}
            row.update(
                {
                    "split": split,
                    "state_id": int(state_id),
                    "seed": state_seed,
                    "feature_rows": int(x.shape[0]),
                    "feature_dim": int(x.shape[1]),
                    "action_energy_feature": float(np.sum(actions[rollout_id] ** 2)),
                }
            )
            rows.append(row)
    return np.vstack(xs), np.vstack(ys), rows


def _write_report(summary: dict[str, Any]) -> None:
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not summary.get("available"):
        lines = [
            "# RoboCasa Learned WAM Report",
            "",
            "- status: unavailable",
            f"- reason: {summary.get('reason')}",
        ]
    else:
        ci = (summary.get("confidence_intervals") or {}).get("learned_minus_random_N8") or {}
        lines = [
            "# RoboCasa Learned WAM Report",
            "",
            f"- status: `{'verified' if summary.get('verified') else 'attempted_not_promoted'}`",
            f"- env: `{summary.get('env_id')}`",
            f"- train samples: `{summary.get('train_samples')}`",
            f"- validation samples: `{summary.get('validation_samples')}`",
            f"- eval rollout pools: `{summary.get('eval_states')}`",
            f"- exact-law utility MAE: `{summary.get('exact_law_utility_mae')}`",
            f"- validation utility correlation: `{(summary.get('model_metrics') or {}).get('utility_corr')}`",
            f"- learned minus random N8 CI: `{ci}`",
            "",
            "This is a lightweight state/action-sequence RoboCasa WAM-lite artifact. It is promoted only if the heldout learned scorer beats random with a positive CI.",
        ]
    (report_dir / "robocasa_learned_wam_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unavailable(reason: str) -> dict[str, Any]:
    ensure_result_dirs()
    summary = {
        "experiment": "benchmark_robocasa_learned_wam",
        "attempted": True,
        "available": False,
        "verified": False,
        "reason": reason,
    }
    write_json(results_dir() / "benchmark_robocasa_learned_wam.json", summary)
    _write_report(summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    ok, reason = is_robocasa_available()
    if not ok:
        return _unavailable(reason)

    adapter: RoboCasaAdapter | None = None
    try:
        adapter = RoboCasaAdapter(
            env_id=args.env_id,
            split=args.split,
            horizon=args.horizon,
            camera_width=args.camera_size,
            camera_height=args.camera_size,
            success_bonus=args.success_bonus,
            energy_penalty=args.energy_penalty,
        )
        train_x, train_y, train_rows = collect_dataset(
            adapter,
            states=args.train_states,
            rollouts=args.train_rollouts,
            horizon=args.horizon,
            seed=args.seed,
            split="train",
        )
        val_x, val_y, val_rows = collect_dataset(
            adapter,
            states=args.val_states,
            rollouts=args.val_rollouts,
            horizon=args.horizon,
            seed=args.seed + 500_000,
            split="validation",
        )
        model = fit_ridge(train_x, train_y, alpha=args.ridge_alpha)
        val_pred = model.predict(val_x)
        utility_corr = (
            float(np.corrcoef(val_pred[:, 0], val_y[:, 0])[0, 1])
            if np.std(val_pred[:, 0]) > 1e-12 and np.std(val_y[:, 0]) > 1e-12
            else 0.0
        )
        progress_corr = (
            float(np.corrcoef(val_pred[:, 1], val_y[:, 1])[0, 1])
            if np.std(val_pred[:, 1]) > 1e-12 and np.std(val_y[:, 1]) > 1e-12
            else 0.0
        )
        model_metrics = {
            "utility_mae": float(np.mean(np.abs(val_pred[:, 0] - val_y[:, 0]))),
            "utility_corr": utility_corr,
            "progress_mae": float(np.mean(np.abs(val_pred[:, 1] - val_y[:, 1]))),
            "progress_corr": progress_corr,
            "final_distance_mae": float(np.mean(np.abs(val_pred[:, 2] - val_y[:, 2]))),
            "energy_mae": float(np.mean(np.abs(val_pred[:, 3] - val_y[:, 3]))),
            "success_mae": float(np.mean(np.abs(val_pred[:, 4] - val_y[:, 4]))),
        }

        model_path = results_dir() / "models" / "benchmark_robocasa_ridge_wam.npz"
        save_model(
            model,
            model_path,
            {
                "env_id": args.env_id,
                "model_type": "ridge_state_action_sequence_wam",
                "train_states": int(args.train_states),
                "train_rollouts": int(args.train_rollouts),
                "horizon": int(args.horizon),
            },
        )

        rows: list[dict[str, Any]] = []
        exact_rows: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        n_values = [int(n) for n in args.n_values]
        for eval_state_id in range(args.eval_states):
            state_seed = int(args.seed + 900_000 + 10_007 * eval_state_id)
            state = adapter.reset_task(seed=state_seed)
            pool = adapter.sample_rollouts(
                initial_state=state,
                n_rollouts=args.eval_rollouts,
                horizon=args.horizon,
                seed=state_seed + 17,
            )
            actions = np.asarray(pool["actions"], dtype=float)
            x = _features(adapter, state, actions)
            pred = model.predict(x)
            records = pool["records"]
            real_utility = np.asarray([r["utility"] for r in records], dtype=float)
            norm_utility = normalized_utility(real_utility)
            energy = np.asarray([r["energy"] for r in records], dtype=float)
            progress = np.asarray([r["progress"] for r in records], dtype=float)
            final_distance = np.asarray([r["final_distance"] for r in records], dtype=float)
            rng = np.random.default_rng(state_seed + 31)
            scorers = {
                "random": rng.normal(size=len(records)),
                "learned_wam": pred[:, 0],
                "predicted_progress": pred[:, 1],
                "low_energy": -energy,
                "distance_progress": progress - final_distance,
                "oracle_real_utility": real_utility,
            }
            for rollout_id, r in enumerate(records):
                row = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in r.items()}
                row.update(
                    {
                        "eval_state_id": int(eval_state_id),
                        "seed": state_seed,
                        "predicted_utility": float(pred[rollout_id, 0]),
                        "predicted_progress": float(pred[rollout_id, 1]),
                    }
                )
                eval_rows.append(row)
            for scorer, scores in scorers.items():
                raw_curve = utility_best_of_n_finite(scores, real_utility, n_values)
                norm_curve = utility_best_of_n_finite(scores, norm_utility, n_values)
                for n in n_values:
                    rows.append(
                        {
                            "env_id": args.env_id,
                            "seed": state_seed,
                            "eval_state_id": int(eval_state_id),
                            "scorer": scorer,
                            "N": int(n),
                            "real_utility": float(raw_curve[n]),
                            "normalized_real_utility": float(norm_curve[n]),
                        }
                    )
                    if scorer in {"learned_wam", "oracle_real_utility"}:
                        mc = simulate_best_of_n(scores, real_utility, n, args.mc_trials, state_seed + 100 * n)
                        exact_rows.append(
                            {
                                "env_id": args.env_id,
                                "seed": state_seed,
                                "eval_state_id": int(eval_state_id),
                                "scorer": scorer,
                                "N": int(n),
                                "utility_exact": float(raw_curve[n]),
                                "utility_mc": float(mc),
                                "utility_abs_error": float(abs(raw_curve[n] - mc)),
                            }
                        )
    except RoboCasaUnavailableError as exc:
        return _unavailable(str(exc))
    finally:
        if adapter is not None:
            adapter.close()

    curves = pd.DataFrame(rows)
    exact = pd.DataFrame(exact_rows)
    data_rows = pd.DataFrame(train_rows + val_rows)
    eval_detail = pd.DataFrame(eval_rows)
    curves_path = results_dir() / "tables" / "benchmark_robocasa_learned_curves.csv"
    exact_path = results_dir() / "tables" / "benchmark_robocasa_learned_exact_law.csv"
    data_path = results_dir() / "tables" / "benchmark_robocasa_learned_train_validation.csv"
    eval_path = results_dir() / "tables" / "benchmark_robocasa_learned_eval_rollouts.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    data_rows.to_csv(data_path, index=False)
    eval_detail.to_csv(eval_path, index=False)

    max_n = max(int(n) for n in args.n_values)
    seed_metrics = []
    for seed, sub in curves[curves["N"] == max_n].groupby("seed"):
        by_scorer = sub.groupby("scorer")["normalized_real_utility"].mean()
        seed_metrics.append(
            {
                "seed": int(seed),
                f"learned_minus_random_N{max_n}": float(by_scorer.get("learned_wam", np.nan) - by_scorer.get("random", np.nan)),
                f"oracle_minus_random_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get("random", np.nan)),
                f"oracle_minus_learned_N{max_n}": float(by_scorer.get("oracle_real_utility", np.nan) - by_scorer.get("learned_wam", np.nan)),
                f"low_energy_minus_random_N{max_n}": float(by_scorer.get("low_energy", np.nan) - by_scorer.get("random", np.nan)),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_path = results_dir() / "tables" / "benchmark_robocasa_learned_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)
    confidence_intervals = {
        key: ci95(seed_df[key].to_numpy())
        for key in seed_df.columns
        if key != "seed"
    }
    exact_mae = float(exact["utility_abs_error"].mean()) if not exact.empty else None
    learned_ci = confidence_intervals.get(f"learned_minus_random_N{max_n}") or {}
    verified = (
        exact_mae is not None
        and exact_mae < args.max_exact_mae
        and learned_ci.get("lo") is not None
        and learned_ci["lo"] > 0.0
        and model_metrics["utility_corr"] > 0.0
    )
    summary = {
        "experiment": "benchmark_robocasa_learned_wam",
        "attempted": True,
        "available": True,
        "verified": bool(verified),
        "env_id": args.env_id,
        "split": args.split,
        "model_path": str(model_path),
        "model_type": "ridge_state_action_sequence_wam",
        "train_states": int(args.train_states),
        "train_rollouts": int(args.train_rollouts),
        "train_samples": int(len(train_x)),
        "validation_states": int(args.val_states),
        "validation_rollouts": int(args.val_rollouts),
        "validation_samples": int(len(val_x)),
        "eval_states": int(args.eval_states),
        "eval_rollouts": int(args.eval_rollouts),
        "eval_samples": int(len(eval_detail)),
        "horizon": int(args.horizon),
        "n_values": [int(n) for n in args.n_values],
        "model_metrics": model_metrics,
        "exact_law_utility_mae": exact_mae,
        "confidence_intervals": confidence_intervals,
        "curves_path": str(curves_path),
        "exact_path": str(exact_path),
        "data_path": str(data_path),
        "eval_path": str(eval_path),
        "seed_metrics_path": str(seed_path),
        "note": "Lightweight optional RoboCasa learned WAM-lite; promoted only when learned-vs-random CI is positive.",
    }
    write_json(results_dir() / "benchmark_robocasa_learned_wam.json", summary)
    _write_report(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="robocasa/PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--train-states", type=int, default=5)
    parser.add_argument("--train-rollouts", type=int, default=16)
    parser.add_argument("--val-states", type=int, default=2)
    parser.add_argument("--val-rollouts", type=int, default=16)
    parser.add_argument("--eval-states", type=int, default=5)
    parser.add_argument("--eval-rollouts", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--camera-size", type=int, default=16)
    parser.add_argument("--mc-trials", type=int, default=2500)
    parser.add_argument("--n-values", nargs="*", type=int, default=N_VALUES)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--energy-penalty", type=float, default=0.01)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-exact-mae", type=float, default=0.01)
    args = parser.parse_args()
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
