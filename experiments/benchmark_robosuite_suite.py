from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wam_inference_value.benchmarks.gym_manip_rollouts import benchmark_feature
from wam_inference_value.benchmarks.robosuite_adapter import RoboSuiteAdapter, is_robosuite_available
from wam_inference_value.benchmarks.robosuite_rollouts import run_closed_loop, sample_rollout_pool
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32]


def fit_model(x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=180,
        max_features=0.50,
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


def make_dataset(adapter: RoboSuiteAdapter, states: int, rollouts: int, horizon: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
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
        terminal_reward = np.asarray([r["terminal_reward"] for r in records], dtype=float)
        total_reward = np.asarray([r["total_reward"] for r in records], dtype=float)
        xs.append(x)
        ys.append(np.column_stack([utility, success, final_distance, terminal_reward, total_reward]))
    return np.vstack(xs), np.vstack(ys)


def evaluate_task(env_name: str, args: argparse.Namespace) -> dict:
    adapter = RoboSuiteAdapter(env_name=env_name, robot=args.robot, horizon=args.horizon)
    try:
        offset = sum((i + 1) * ord(ch) for i, ch in enumerate(env_name)) % 10_000
        train_x, train_y = make_dataset(adapter, args.train_states, args.train_rollouts, args.horizon, args.seed + offset)
        val_x, val_y = make_dataset(adapter, args.val_states, args.val_rollouts, args.horizon, args.seed + 20_000 + offset)
        model = fit_model(train_x, train_y, args.seed + offset)
        val_pred = np.asarray(model.predict(val_x), dtype=float)
        utility_mae = float(np.mean(np.abs(val_pred[:, 0] - val_y[:, 0])))
        utility_corr = (
            float(np.corrcoef(val_pred[:, 0], val_y[:, 0])[0, 1])
            if np.std(val_pred[:, 0]) > 1e-12 and np.std(val_y[:, 0]) > 1e-12
            else 0.0
        )
        success_mae = float(np.mean(np.abs(val_pred[:, 1] - val_y[:, 1])))
        distance_mae = float(np.mean(np.abs(val_pred[:, 2] - val_y[:, 2])))
        reward_mae = float(np.mean(np.abs(val_pred[:, 3] - val_y[:, 3])))
        total_reward_mae = float(np.mean(np.abs(val_pred[:, 4] - val_y[:, 4])))

        safe_name = env_name.replace("-", "_")
        model_path = results_dir() / "models" / f"benchmark_robosuite_{safe_name}_wam.joblib"
        save_model(model, model_path, {"env_name": env_name, "model_type": "extra_trees_state_action_sequence_wam"})

        rows = []
        exact_rows = []
        closed_rows = []
        for seed in args.seeds:
            for state_id in range(args.states):
                state = adapter.reset(seed + 1231 * state_id)
                pool = sample_rollout_pool(adapter, state, args.rollouts, args.horizon, seed + 65_537 * (state_id + 1) + offset)
                actions = pool["actions"]
                records = pool["records"]
                x = benchmark_feature(adapter.feature_state(state), actions, args.horizon)
                pred = np.asarray(model.predict(x), dtype=float)
                real_utility = np.asarray([r["utility"] for r in records], dtype=float)
                success = np.asarray([float(r["success"]) for r in records], dtype=float)
                energy = np.asarray([r["energy"] for r in records], dtype=float)
                total_reward = np.asarray([r["total_reward"] for r in records], dtype=float)
                terminal_reward = np.asarray([r["terminal_reward"] for r in records], dtype=float)
                progress = np.asarray([r["progress"] + r["dense_progress"] for r in records], dtype=float)
                norm_utility = normalized_utility(real_utility)
                score_sets = {
                    "random": np.random.default_rng(seed + state_id + offset).normal(size=len(real_utility)),
                    "learned_wam": pred[:, 0],
                    "predicted_success": pred[:, 1],
                    "benchmark_reward": total_reward + terminal_reward,
                    "progress": progress,
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
                                "benchmark": env_name,
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
                    mc = simulate_best_of_n(pred[:, 0], real_utility, n, args.mc_trials, seed + 17 * n + state_id + offset)
                    exact_rows.append(
                        {
                            "benchmark": env_name,
                            "seed": int(seed),
                            "state_id": int(state_id),
                            "N": int(n),
                            "utility_abs_error": float(abs(exact - mc)),
                        }
                    )
            if args.closed_loop:
                for n in (1, 8):
                    for scorer in ("random", "learned", "reward", "oracle"):
                        rec = run_closed_loop(
                            adapter,
                            model,
                            scorer,
                            n=n,
                            seed=seed + 4242 + offset,
                            steps=args.closed_loop_steps,
                            candidate_horizon=args.closed_loop_horizon,
                            feature_horizon=args.horizon,
                        )
                        rec.update({"benchmark": env_name, "seed": int(seed), "scorer": scorer, "N": int(n)})
                        closed_rows.append(rec)

        return {
            "benchmark": env_name,
            "available": True,
            "model_path": str(model_path),
            "model_metrics": {
                "utility_mae": utility_mae,
                "utility_corr": utility_corr,
                "success_mae": success_mae,
                "distance_mae": distance_mae,
                "reward_mae": reward_mae,
                "total_reward_mae": total_reward_mae,
                "train_samples": int(len(train_x)),
                "validation_samples": int(len(val_x)),
            },
            "rows": rows,
            "exact_rows": exact_rows,
            "closed_rows": closed_rows,
        }
    finally:
        adapter.close()


def _write_blocker_report(summary: dict) -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RoboSuite Blocker Report",
        "",
        f"- attempted: `{summary.get('attempted')}`",
        f"- available: `{summary.get('available')}`",
        f"- unavailable: `{summary.get('unavailable')}`",
        "",
        "RoboSuite benchmark claims remain unavailable unless `results/benchmark_robosuite_suite.json` reports `available: true`.",
    ]
    (reports_dir / "robosuite_blocker_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    unavailable = []
    for env_name in args.env_names:
        ok, reason = is_robosuite_available(env_name, args.robot)
        if not ok:
            unavailable.append({"env_name": env_name, "reason": reason})
    if unavailable and len(unavailable) == len(args.env_names):
        summary = {"experiment": "benchmark_robosuite_suite", "attempted": True, "available": False, "unavailable": unavailable}
        write_json(results_dir() / "benchmark_robosuite_suite.json", summary)
        _write_blocker_report(summary)
        return summary

    task_summaries = []
    rows = []
    exact_rows = []
    closed_rows = []
    for env_name in args.env_names:
        ok, reason = is_robosuite_available(env_name, args.robot)
        if not ok:
            task_summaries.append({"benchmark": env_name, "available": False, "reason": reason})
            continue
        task_summary = evaluate_task(env_name, args)
        task_summaries.append({k: v for k, v in task_summary.items() if k not in {"rows", "exact_rows", "closed_rows"}})
        rows.extend(task_summary["rows"])
        exact_rows.extend(task_summary["exact_rows"])
        closed_rows.extend(task_summary["closed_rows"])

    curves = pd.DataFrame(rows)
    exact = pd.DataFrame(exact_rows)
    closed = pd.DataFrame(closed_rows)
    curves_path = results_dir() / "tables" / "benchmark_robosuite_curves.csv"
    exact_path = results_dir() / "tables" / "benchmark_robosuite_exact_law.csv"
    closed_path = results_dir() / "tables" / "benchmark_robosuite_closed_loop.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    if not closed.empty:
        closed.to_csv(closed_path, index=False)

    agg = curves.groupby(["benchmark", "scorer", "N"], dropna=False)[["success", "real_utility", "normalized_real_utility"]].mean().reset_index()
    agg_path = results_dir() / "tables" / "benchmark_robosuite_curves_aggregate.csv"
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
                "reward_minus_random_N32": float(n32["benchmark_reward"] - n32["random"]),
                "progress_minus_random_N32": float(n32["progress"] - n32["random"]),
                "oracle_minus_learned_N32": float(n32["oracle_real_utility"] - n32["learned_wam"]),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_path = results_dir() / "tables" / "benchmark_robosuite_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)

    closed_ci = {}
    if not closed.empty:
        closed_seed = closed.groupby(["seed", "scorer", "N"], dropna=False)["utility"].mean().reset_index()
        learned_vals = []
        reward_vals = []
        for seed, sub in closed_seed.groupby("seed"):
            n8 = sub[sub["N"] == 8].set_index("scorer")["utility"]
            learned_vals.append(float(n8["learned"] - n8["random"]))
            reward_vals.append(float(n8["reward"] - n8["random"]))
        closed_ci["closed_loop_learned_minus_random_N8"] = ci95(learned_vals)
        closed_ci["closed_loop_reward_minus_random_N8"] = ci95(reward_vals)

    plt.figure(figsize=(7.5, 4.8))
    plot_df = agg.groupby(["scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
    for scorer, sub in plot_df.groupby("scorer"):
        if scorer in {"random", "learned_wam", "benchmark_reward", "progress", "oracle_real_utility"}:
            plt.plot(sub["N"], sub["normalized_real_utility"], marker="o", label=scorer)
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("normalized utility")
    plt.title("RoboSuite benchmark inference curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "benchmark_robosuite_curves.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    summary = {
        "experiment": "benchmark_robosuite_suite",
        "attempted": True,
        "available": True,
        "benchmark": "RoboSuite",
        "env_names": args.env_names,
        "robot": args.robot,
        "controller": "BASIC",
        "n_tasks_verified": int(sum(1 for s in task_summaries if s.get("available"))),
        "n_rollout_pools": int(len(args.seeds) * args.states * sum(1 for s in task_summaries if s.get("available"))),
        "n_rollouts": int(args.rollouts),
        "seeds": [int(s) for s in args.seeds],
        "N_values": N_VALUES,
        "model_metrics": task_summaries,
        "exact_law_utility_mae": float(exact["utility_abs_error"].mean()),
        "score_comparison": {
            "learned_minus_random_N32": float(seed_df["learned_minus_random_N32"].mean()),
            "reward_minus_random_N32": float(seed_df["reward_minus_random_N32"].mean()),
            "progress_minus_random_N32": float(seed_df["progress_minus_random_N32"].mean()),
            "oracle_minus_random_N32": float(seed_df["oracle_minus_random_N32"].mean()),
            "oracle_minus_learned_N32": float(seed_df["oracle_minus_learned_N32"].mean()),
        },
        "closed_loop": {
            "learned_minus_random_N8": closed_ci.get("closed_loop_learned_minus_random_N8", {}).get("mean"),
            "reward_minus_random_N8": closed_ci.get("closed_loop_reward_minus_random_N8", {}).get("mean"),
        },
        "confidence_intervals": {
            "learned_minus_random_N32": ci95(seed_df["learned_minus_random_N32"].to_numpy()),
            "reward_minus_random_N32": ci95(seed_df["reward_minus_random_N32"].to_numpy()),
            "progress_minus_random_N32": ci95(seed_df["progress_minus_random_N32"].to_numpy()),
            "oracle_minus_random_N32": ci95(seed_df["oracle_minus_random_N32"].to_numpy()),
            "oracle_minus_learned_N32": ci95(seed_df["oracle_minus_learned_N32"].to_numpy()),
            **closed_ci,
        },
        "artifacts": {
            "curves": str(curves_path),
            "curves_aggregate": str(agg_path),
            "exact_law": str(exact_path),
            "closed_loop": str(closed_path) if not closed.empty else None,
            "seed_metrics": str(seed_path),
            "figure": str(fig_path),
        },
    }
    write_json(results_dir() / "benchmark_robosuite_suite.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", dest="env_names", nargs="*", default=["Lift", "Stack", "Door"])
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=40)
    parser.add_argument("--train-states", type=int, default=8)
    parser.add_argument("--train-rollouts", type=int, default=48)
    parser.add_argument("--val-states", type=int, default=3)
    parser.add_argument("--val-rollouts", type=int, default=40)
    parser.add_argument("--mc-trials", type=int, default=700)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--closed-loop-steps", type=int, default=5)
    parser.add_argument("--closed-loop-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=16001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[16001, 16002, 16003, 16004, 16005])
    args = parser.parse_args()
    summary = run(args)
    print(
        "benchmark RoboSuite complete: "
        f"available={summary.get('available')}, envs={summary.get('env_names')}, "
        f"exact_mae={summary.get('exact_law_utility_mae')}"
    )


if __name__ == "__main__":
    main()
