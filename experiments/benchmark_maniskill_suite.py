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

from wam_inference_value.benchmarks.maniskill_adapter import ManiSkillAdapter, is_maniskill_available
from wam_inference_value.benchmarks.maniskill_rollouts import benchmark_feature, run_closed_loop, sample_rollout_pool
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.models import HorizonWAM, WAMDataset
from wam_inference_value.theorem import binary_best_of_n_finite, simulate_best_of_n, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32]


def dataset_from_rollouts(adapter: ManiSkillAdapter, n_states: int, n_rollouts: int, horizon: int, seed: int, split: str) -> WAMDataset:
    xs = []
    ys = []
    for state_id in range(int(n_states)):
        state = adapter.reset(seed + 4099 * state_id)
        feature_state = adapter.feature_state(state)
        pool = sample_rollout_pool(adapter, state, n_rollouts, horizon, seed + 100_003 * (state_id + 1))
        actions = pool["actions"]
        records = pool["records"]
        x = benchmark_feature(feature_state, actions, horizon)
        final_states = np.asarray([r["final_state"] for r in records], dtype=float)
        utilities = np.asarray([r["utility"] for r in records], dtype=float)
        ys.append(np.column_stack([final_states - state[None, :], utilities]))
        xs.append(x)
    return WAMDataset(np.vstack(xs), np.vstack(ys), {"benchmark": adapter.env_id, "split": split, "seed": int(seed)})


def eval_model(model: HorizonWAM, dataset: WAMDataset, split: str, env_id: str) -> dict:
    pred = model.predict(dataset.x)
    err = pred - dataset.y
    util = dataset.y[:, -1]
    pred_util = pred[:, -1]
    corr = float(np.corrcoef(util, pred_util)[0, 1]) if np.std(util) > 1e-12 and np.std(pred_util) > 1e-12 else 0.0
    return {
        "benchmark": env_id,
        "split": split,
        "n_samples": int(len(dataset.x)),
        "state_delta_mae": float(np.mean(np.abs(err[:, :-1]))),
        "utility_mae": float(np.mean(np.abs(err[:, -1]))),
        "utility_corr": corr,
    }


def run_task(args: argparse.Namespace, env_id: str, task_i: int) -> dict:
    adapter = ManiSkillAdapter(
        env_id=env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        horizon=args.horizon,
        success_bonus=args.success_bonus,
        energy_penalty=args.energy_penalty,
    )
    try:
        train = dataset_from_rollouts(adapter, args.train_states, args.train_rollouts, args.horizon, args.seed + 10_000 * task_i, "train")
        val = dataset_from_rollouts(adapter, args.val_states, args.val_rollouts, args.horizon, args.seed + 20_000 + 10_000 * task_i, "validation")
        model = HorizonWAM().fit(train)
        model_path = results_dir() / "models" / f"benchmark_maniskill_{env_id}_horizon_wam.npz"
        model.save(model_path, {"benchmark": env_id, "seed": args.seed})
        model_metrics = [eval_model(model, train, "train", env_id), eval_model(model, val, "validation", env_id)]

        curve_rows = []
        exact_rows = []
        gap_rows = []
        rollout_pool_count = 0
        for seed in args.seeds:
            for state_id in range(args.states):
                state = adapter.reset(seed + 811 * state_id + 17 * task_i)
                feature_state = adapter.feature_state(state)
                pool = sample_rollout_pool(adapter, state, args.rollouts, args.horizon, seed + 65_537 * (state_id + 1) + task_i)
                rollout_pool_count += 1
                actions = pool["actions"]
                records = pool["records"]
                real_utility = np.asarray([r["utility"] for r in records], dtype=float)
                success = np.asarray([float(r["success"]) for r in records], dtype=float)
                energy = np.asarray([r["energy"] for r in records], dtype=float)
                total_reward = np.asarray([r["total_reward"] for r in records], dtype=float)
                x = benchmark_feature(feature_state, actions, args.horizon)
                pred_utility = model.predict(x)[:, -1]
                rng = np.random.default_rng(seed + state_id + 123 * task_i)
                scorers = {
                    "random": rng.normal(size=len(real_utility)),
                    "dense_reward": total_reward,
                    "low_energy": -energy,
                    "learned_predicted_utility": pred_utility,
                    "oracle_real_utility": real_utility,
                }
                for scorer, scores in scorers.items():
                    u_curve = utility_best_of_n_finite(scores, real_utility, N_VALUES)
                    s_curve = binary_best_of_n_finite(scores, success, N_VALUES)
                    pred_curve = utility_best_of_n_finite(scores, pred_utility, N_VALUES)
                    for n in N_VALUES:
                        curve_rows.append(
                            {
                                "benchmark": env_id,
                                "seed": int(seed),
                                "state_id": int(state_id),
                                "scorer": scorer,
                                "N": int(n),
                                "success": s_curve[n],
                                "real_utility": u_curve[n],
                                "predicted_utility": pred_curve[n],
                            }
                        )
                    if scorer == "learned_predicted_utility":
                        for n in [1, 8, 32]:
                            gap_rows.append(
                                {
                                    "benchmark": env_id,
                                    "seed": int(seed),
                                    "state_id": int(state_id),
                                    "N": int(n),
                                    "predicted_minus_real_utility": pred_curve[n] - u_curve[n],
                                }
                            )
                learned_scores = scorers["learned_predicted_utility"]
                for n in N_VALUES:
                    exact_u = utility_best_of_n_finite(learned_scores, real_utility, [n])[n]
                    mc_u = simulate_best_of_n(learned_scores, real_utility, n, args.mc_trials, seed + 17 * n + state_id + task_i)
                    exact_rows.append({"benchmark": env_id, "seed": int(seed), "state_id": int(state_id), "N": int(n), "utility_abs_error": abs(exact_u - mc_u)})

        closed_rows = []
        if args.closed_loop:
            for seed in args.seeds:
                for scorer in ["random", "learned", "oracle"]:
                    for n in [1, 8]:
                        rec = run_closed_loop(
                            adapter,
                            model,
                            scorer,
                            n,
                            seed + 123_457 + 1000 * task_i,
                            steps=args.closed_loop_steps,
                            candidate_horizon=args.closed_loop_horizon,
                            feature_horizon=args.horizon,
                        )
                        rec.update({"benchmark": env_id, "seed": int(seed), "scorer": scorer, "N": int(n)})
                        closed_rows.append(rec)
        return {
            "benchmark": env_id,
            "model_path": str(model_path),
            "model_metrics": model_metrics,
            "curve_rows": curve_rows,
            "exact_rows": exact_rows,
            "gap_rows": gap_rows,
            "closed_rows": closed_rows,
            "rollout_pool_count": int(rollout_pool_count),
        }
    finally:
        adapter.close()


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    if not is_maniskill_available():
        summary = {"experiment": "benchmark_maniskill_suite", "attempted": True, "available": False, "reason": "ManiSkill import not found"}
        write_json(results_dir() / "benchmark_maniskill_suite.json", summary)
        return summary

    task_summaries = [run_task(args, env_id, i) for i, env_id in enumerate(args.env_ids)]
    curve_rows = [row for task in task_summaries for row in task["curve_rows"]]
    exact_rows = [row for task in task_summaries for row in task["exact_rows"]]
    gap_rows = [row for task in task_summaries for row in task["gap_rows"]]
    closed_rows = [row for task in task_summaries for row in task["closed_rows"]]
    model_metrics = [row for task in task_summaries for row in task["model_metrics"]]

    curves = pd.DataFrame(curve_rows)
    exact = pd.DataFrame(exact_rows)
    gaps = pd.DataFrame(gap_rows)
    closed = pd.DataFrame(closed_rows)
    curves_path = results_dir() / "tables" / "benchmark_maniskill_curves.csv"
    exact_path = results_dir() / "tables" / "benchmark_maniskill_exact_law.csv"
    gaps_path = results_dir() / "tables" / "benchmark_maniskill_real_vs_predicted_gap.csv"
    metrics_path = results_dir() / "tables" / "benchmark_maniskill_model_metrics.csv"
    closed_path = results_dir() / "tables" / "benchmark_maniskill_closed_loop.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    pd.DataFrame(model_metrics).to_csv(metrics_path, index=False)
    if not closed.empty:
        closed.to_csv(closed_path, index=False)

    agg = curves.groupby(["benchmark", "scorer", "N"], dropna=False)[["success", "real_utility", "predicted_utility"]].mean().reset_index()
    agg_path = results_dir() / "tables" / "benchmark_maniskill_curves_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    seed_agg = curves.groupby(["seed", "scorer", "N"], dropna=False)["real_utility"].mean().reset_index()
    learned_n32 = seed_agg[(seed_agg["scorer"] == "learned_predicted_utility") & (seed_agg["N"] == 32)].set_index("seed")["real_utility"]
    random_n32 = seed_agg[(seed_agg["scorer"] == "random") & (seed_agg["N"] == 32)].set_index("seed")["real_utility"]
    oracle_n32 = seed_agg[(seed_agg["scorer"] == "oracle_real_utility") & (seed_agg["N"] == 32)].set_index("seed")["real_utility"]
    dense_n32 = seed_agg[(seed_agg["scorer"] == "dense_reward") & (seed_agg["N"] == 32)].set_index("seed")["real_utility"]
    common = sorted(set(learned_n32.index) & set(random_n32.index) & set(oracle_n32.index) & set(dense_n32.index))
    learned_minus_random = (learned_n32.loc[common] - random_n32.loc[common]).to_numpy()
    dense_minus_random = (dense_n32.loc[common] - random_n32.loc[common]).to_numpy()
    oracle_minus_random = (oracle_n32.loc[common] - random_n32.loc[common]).to_numpy()
    oracle_minus_learned = (oracle_n32.loc[common] - learned_n32.loc[common]).to_numpy()

    closed_delta = np.asarray([], dtype=float)
    if not closed.empty:
        closed_agg = closed.groupby(["seed", "scorer", "N"], dropna=False)["utility"].mean().reset_index()
        learned_closed = closed_agg[(closed_agg["scorer"] == "learned") & (closed_agg["N"] == 8)].set_index("seed")["utility"]
        random_closed = closed_agg[(closed_agg["scorer"] == "random") & (closed_agg["N"] == 8)].set_index("seed")["utility"]
        common_closed = sorted(set(learned_closed.index) & set(random_closed.index))
        closed_delta = (learned_closed.loc[common_closed] - random_closed.loc[common_closed]).to_numpy()

    plt.figure(figsize=(7.6, 4.8))
    for scorer, sub in agg.groupby("scorer"):
        if scorer in {"random", "dense_reward", "learned_predicted_utility", "oracle_real_utility"}:
            plt.plot(sub.groupby("N")["real_utility"].mean().index, sub.groupby("N")["real_utility"].mean().values, marker="o", label=scorer)
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("real utility")
    plt.title("ManiSkill state benchmark inference curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "benchmark_maniskill_curves.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    exact_mae = float(exact["utility_abs_error"].mean())
    gap_growth = float(gaps[gaps["N"] == 32]["predicted_minus_real_utility"].mean() - gaps[gaps["N"] == 1]["predicted_minus_real_utility"].mean())
    summary = {
        "experiment": "benchmark_maniskill_suite",
        "attempted": True,
        "available": True,
        "benchmark": "ManiSkill",
        "env_ids": args.env_ids,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "note": "CPU state-mode ManiSkill3 with joint-delta control. EE control was not used because Pinocchio is unavailable in this Windows environment.",
        "n_rollout_pools": int(sum(task["rollout_pool_count"] for task in task_summaries)),
        "n_rollouts": int(args.rollouts),
        "seeds": [int(s) for s in args.seeds],
        "N_values": N_VALUES,
        "model_metrics": model_metrics,
        "exact_law_utility_mae": exact_mae,
        "score_comparison": {
            "learned_minus_random_real_utility_N32": float(np.mean(learned_minus_random)),
            "dense_minus_random_real_utility_N32": float(np.mean(dense_minus_random)),
            "oracle_minus_random_real_utility_N32": float(np.mean(oracle_minus_random)),
            "oracle_minus_learned_real_utility_N32": float(np.mean(oracle_minus_learned)),
        },
        "real_vs_predicted_gap": {"gap_growth_N32_minus_N1": gap_growth},
        "closed_loop": {"learned_minus_random_utility_N8": float(np.mean(closed_delta)) if closed_delta.size else None},
        "confidence_intervals": {
            "learned_minus_random_real_utility_N32": ci95(learned_minus_random),
            "dense_minus_random_real_utility_N32": ci95(dense_minus_random),
            "oracle_minus_random_real_utility_N32": ci95(oracle_minus_random),
            "oracle_minus_learned_real_utility_N32": ci95(oracle_minus_learned),
            "closed_loop_learned_minus_random_utility_N8": ci95(closed_delta) if closed_delta.size else {"n": 0, "mean": None, "lo": None, "hi": None},
        },
        "artifacts": {
            "curves": str(curves_path),
            "curves_aggregate": str(agg_path),
            "exact_law": str(exact_path),
            "real_vs_predicted_gap": str(gaps_path),
            "model_metrics": str(metrics_path),
            "closed_loop": str(closed_path) if not closed.empty else None,
            "figure": str(fig_path),
        },
    }
    write_json(results_dir() / "benchmark_maniskill_suite.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="*", default=["PickCube-v1", "PushCube-v1", "PegInsertionSide-v1"])
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--train-states", type=int, default=4)
    parser.add_argument("--train-rollouts", type=int, default=32)
    parser.add_argument("--val-states", type=int, default=2)
    parser.add_argument("--val-rollouts", type=int, default=24)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--energy-penalty", type=float, default=0.002)
    parser.add_argument("--mc-trials", type=int, default=500)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--closed-loop-steps", type=int, default=5)
    parser.add_argument("--closed-loop-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[12001, 12002, 12003, 12004, 12005])
    args = parser.parse_args()
    summary = run(args)
    print(
        "benchmark ManiSkill complete: "
        f"available={summary.get('available')}, pools={summary.get('n_rollout_pools')}, "
        f"exact_mae={summary.get('exact_law_utility_mae')}"
    )


if __name__ == "__main__":
    main()
