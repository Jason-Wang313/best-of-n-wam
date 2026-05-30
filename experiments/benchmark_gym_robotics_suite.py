from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wam_inference_value.benchmarks.gym_manip_rollouts import benchmark_feature
from wam_inference_value.benchmarks.gym_robotics_adapter import GymRoboticsAdapter, is_gym_robotics_available
from wam_inference_value.benchmarks.gym_robotics_rollouts import run_closed_loop, sample_rollout_pool
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32]


def fit_model(x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=120,
        max_features=0.45,
        min_samples_leaf=2,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def save_model(model, path: Path, metadata: dict) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def make_dataset(adapter: GymRoboticsAdapter, states: int, rollouts: int, horizon: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for state_id in range(int(states)):
        state = adapter.reset(seed + 7919 * state_id)
        pool = sample_rollout_pool(adapter, state, rollouts, horizon, seed + 100_003 * (state_id + 1))
        actions = pool["actions"]
        records = pool["records"]
        x = benchmark_feature(adapter.feature_state(state), actions, horizon)
        utility = np.asarray([r["utility"] for r in records], dtype=float)
        success = np.asarray([float(r["success"]) for r in records], dtype=float)
        final_distance = np.asarray([r["final_distance"] for r in records], dtype=float)
        xs.append(x)
        ys.append(np.column_stack([utility, success, final_distance]))
    return np.vstack(xs), np.vstack(ys)


def evaluate_env(env_id: str, args: argparse.Namespace) -> dict:
    adapter = GymRoboticsAdapter(env_id=env_id, render_mode="rgb_array", horizon=args.horizon)
    try:
        env_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(env_id)) % 10_000
        train_x, train_y = make_dataset(adapter, args.train_states, args.train_rollouts, args.horizon, args.seed + env_offset)
        val_x, val_y = make_dataset(adapter, args.val_states, args.val_rollouts, args.horizon, args.seed + 20_000 + env_offset)
        model = fit_model(train_x, train_y, args.seed)
        val_pred = np.asarray(model.predict(val_x), dtype=float)
        utility_mae = float(np.mean(np.abs(val_pred[:, 0] - val_y[:, 0])))
        utility_corr = float(np.corrcoef(val_pred[:, 0], val_y[:, 0])[0, 1]) if np.std(val_pred[:, 0]) > 1e-12 and np.std(val_y[:, 0]) > 1e-12 else 0.0
        success_mae = float(np.mean(np.abs(val_pred[:, 1] - val_y[:, 1])))
        distance_mae = float(np.mean(np.abs(val_pred[:, 2] - val_y[:, 2])))

        model_path = results_dir() / "models" / f"benchmark_gym_robotics_{env_id}_wam.joblib"
        save_model(model, model_path, {"env_id": env_id, "model_type": "extra_trees_state_action_sequence_wam"})

        rows = []
        exact_rows = []
        closed_rows = []
        frame_path = results_dir() / "figures" / f"benchmark_gym_robotics_{env_id}_frame.png"
        wrote_frame = False
        for seed in args.seeds:
            for state_id in range(args.states):
                state = adapter.reset(seed + 1231 * state_id)
                if not wrote_frame:
                    iio.imwrite(frame_path, adapter.env.render())
                    wrote_frame = True
                pool = sample_rollout_pool(adapter, state, args.rollouts, args.horizon, seed + 65_537 * (state_id + 1))
                actions = pool["actions"]
                records = pool["records"]
                x = benchmark_feature(adapter.feature_state(state), actions, args.horizon)
                pred = np.asarray(model.predict(x), dtype=float)
                real_utility = np.asarray([r["utility"] for r in records], dtype=float)
                success = np.asarray([float(r["success"]) for r in records], dtype=float)
                energy = np.asarray([r["energy"] for r in records], dtype=float)
                norm_utility = normalized_utility(real_utility)
                score_sets = {
                    "random": np.random.default_rng(seed + state_id).normal(size=len(real_utility)),
                    "learned_wam": pred[:, 0],
                    "predicted_success": pred[:, 1],
                    "low_energy": -energy,
                    "oracle_real_utility": real_utility,
                }
                for scorer, scores in score_sets.items():
                    raw_curve = utility_best_of_n_finite(scores, real_utility, N_VALUES)
                    norm_curve = utility_best_of_n_finite(scores, norm_utility, N_VALUES)
                    succ_curve = utility_best_of_n_finite(scores, success, N_VALUES)
                    for n in N_VALUES:
                        rows.append(
                            {
                                "benchmark": env_id,
                                "seed": int(seed),
                                "state_id": int(state_id),
                                "scorer": scorer,
                                "N": int(n),
                                "real_utility": float(raw_curve[n]),
                                "normalized_real_utility": float(norm_curve[n]),
                                "success": float(succ_curve[n]),
                            }
                        )
                for n in N_VALUES:
                    exact = utility_best_of_n_finite(pred[:, 0], real_utility, [n])[n]
                    mc = simulate_best_of_n(pred[:, 0], real_utility, n, args.mc_trials, seed + 17 * n + state_id)
                    exact_rows.append(
                        {
                            "benchmark": env_id,
                            "seed": int(seed),
                            "state_id": int(state_id),
                            "N": int(n),
                            "utility_abs_error": float(abs(exact - mc)),
                        }
                    )
            if args.closed_loop:
                for n in (1, 8, 32):
                    for scorer in ("random", "learned", "oracle"):
                        rec = run_closed_loop(adapter, model, scorer, n=n, seed=seed + 4242, steps=args.closed_loop_steps, candidate_horizon=args.closed_loop_horizon, feature_horizon=args.horizon)
                        rec.update({"benchmark": env_id, "seed": int(seed), "scorer": scorer, "N": int(n)})
                        closed_rows.append(rec)

        return {
            "benchmark": env_id,
            "available": True,
            "model_path": str(model_path),
            "frame_path": str(frame_path),
            "model_metrics": {
                "utility_mae": utility_mae,
                "utility_corr": utility_corr,
                "success_mae": success_mae,
                "distance_mae": distance_mae,
                "train_samples": int(len(train_x)),
                "validation_samples": int(len(val_x)),
            },
            "rows": rows,
            "exact_rows": exact_rows,
            "closed_rows": closed_rows,
        }
    finally:
        adapter.close()


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    unavailable = []
    for env_id in args.env_ids:
        ok, reason = is_gym_robotics_available(env_id)
        if not ok:
            unavailable.append({"env_id": env_id, "reason": reason})
    if unavailable and len(unavailable) == len(args.env_ids):
        summary = {"experiment": "benchmark_gym_robotics_suite", "attempted": True, "available": False, "unavailable": unavailable}
        write_json(results_dir() / "benchmark_gym_robotics_suite.json", summary)
        return summary

    env_summaries = []
    all_rows = []
    all_exact = []
    all_closed = []
    for env_id in args.env_ids:
        ok, reason = is_gym_robotics_available(env_id)
        if not ok:
            env_summaries.append({"benchmark": env_id, "available": False, "reason": reason})
            continue
        env_summary = evaluate_env(env_id, args)
        env_summaries.append({k: v for k, v in env_summary.items() if k not in {"rows", "exact_rows", "closed_rows"}})
        all_rows.extend(env_summary["rows"])
        all_exact.extend(env_summary["exact_rows"])
        all_closed.extend(env_summary["closed_rows"])

    curves = pd.DataFrame(all_rows)
    exact = pd.DataFrame(all_exact)
    closed = pd.DataFrame(all_closed)
    curves_path = results_dir() / "tables" / "benchmark_gym_robotics_curves.csv"
    exact_path = results_dir() / "tables" / "benchmark_gym_robotics_exact_law.csv"
    closed_path = results_dir() / "tables" / "benchmark_gym_robotics_closed_loop.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    if not closed.empty:
        closed.to_csv(closed_path, index=False)

    agg = curves.groupby(["benchmark", "scorer", "N"], dropna=False)[["success", "real_utility", "normalized_real_utility"]].mean().reset_index()
    agg_path = results_dir() / "tables" / "benchmark_gym_robotics_curves_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    seed_agg = curves.groupby(["seed", "scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
    seed_metrics = []
    for seed, sub in seed_agg.groupby("seed"):
        n32 = sub[sub["N"] == 32].set_index("scorer")["normalized_real_utility"]
        seed_metrics.append(
            {
                "seed": int(seed),
                "oracle_minus_random_N32": float(n32["oracle_real_utility"] - n32["random"]),
                "learned_minus_random_N32": float(n32["learned_wam"] - n32["random"]),
                "oracle_minus_learned_N32": float(n32["oracle_real_utility"] - n32["learned_wam"]),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_path = results_dir() / "tables" / "benchmark_gym_robotics_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)

    closed_ci = {}
    if not closed.empty:
        closed_seed = closed.groupby(["seed", "scorer", "N"], dropna=False)["utility"].mean().reset_index()
        vals = []
        for seed, sub in closed_seed.groupby("seed"):
            n32 = sub[sub["N"] == 32].set_index("scorer")["utility"]
            vals.append(float(n32["learned"] - n32["random"]))
        closed_ci["closed_loop_learned_minus_random_N32"] = ci95(vals)

    plt.figure(figsize=(7.5, 4.8))
    plot_df = agg.groupby(["scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
    for scorer, sub in plot_df.groupby("scorer"):
        if scorer in {"random", "learned_wam", "low_energy", "oracle_real_utility"}:
            plt.plot(sub["N"], sub["normalized_real_utility"], marker="o", label=scorer)
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("normalized real utility")
    plt.title("Gymnasium Robotics Fetch benchmark")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "benchmark_gym_robotics_curves.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    exact_mae = float(exact["utility_abs_error"].mean()) if not exact.empty else None
    cis = {
        "oracle_minus_random_N32": ci95(seed_df["oracle_minus_random_N32"].to_numpy()),
        "learned_minus_random_N32": ci95(seed_df["learned_minus_random_N32"].to_numpy()),
        "oracle_minus_learned_N32": ci95(seed_df["oracle_minus_learned_N32"].to_numpy()),
    } | closed_ci
    summary = {
        "experiment": "benchmark_gym_robotics_suite",
        "attempted": True,
        "available": True,
        "verified": bool(exact_mae is not None and exact_mae < args.max_exact_mae and cis["oracle_minus_random_N32"]["lo"] > 0.0),
        "benchmark": "Gymnasium Robotics Fetch",
        "env_ids": [s["benchmark"] for s in env_summaries if s.get("available")],
        "unavailable": unavailable,
        "n_rollout_pools": int(curves[["benchmark", "seed", "state_id"]].drop_duplicates().shape[0]),
        "n_rollouts": int(args.rollouts),
        "exact_law_utility_mae": exact_mae,
        "model_metrics": [s for s in env_summaries if s.get("available")],
        "confidence_intervals": cis,
        "artifacts": {
            "curves": str(curves_path),
            "aggregate": str(agg_path),
            "exact_law": str(exact_path),
            "seed_metrics": str(seed_path),
            "closed_loop": str(closed_path) if not closed.empty else None,
            "figure": str(fig_path),
        },
    }
    write_json(results_dir() / "benchmark_gym_robotics_suite.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="*", default=["FetchReach-v4", "FetchPush-v4", "FetchPickAndPlace-v4"])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--train-states", type=int, default=16)
    parser.add_argument("--train-rollouts", type=int, default=48)
    parser.add_argument("--val-states", type=int, default=6)
    parser.add_argument("--val-rollouts", type=int, default=48)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=48)
    parser.add_argument("--mc-trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=25001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[25001, 25002, 25003, 25004, 25005])
    parser.add_argument("--closed-loop", action="store_true", default=True)
    parser.add_argument("--closed-loop-steps", type=int, default=8)
    parser.add_argument("--closed-loop-horizon", type=int, default=6)
    parser.add_argument("--max-exact-mae", type=float, default=0.06)
    args = parser.parse_args()
    summary = run(args)
    print(
        "benchmark Gymnasium Robotics complete: "
        f"available={summary.get('available')}, envs={summary.get('env_ids')}, "
        f"exact_mae={summary.get('exact_law_utility_mae')}"
    )


if __name__ == "__main__":
    main()
