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

from wam_inference_value.benchmarks.gym_manip_adapter import GymManipAdapter
from wam_inference_value.benchmarks.gym_manip_rollouts import benchmark_feature, run_closed_loop, sample_rollout_pool
from wam_inference_value.evaluation import N_VALUES, ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.models import HorizonWAM, WAMDataset
from wam_inference_value.theorem import binary_best_of_n_finite, simulate_best_of_n, utility_best_of_n_finite


BENCHMARK_N_VALUES = [1, 2, 4, 8, 16, 32]


def dataset_from_rollouts(adapter: GymManipAdapter, n_states: int, n_rollouts: int, horizon: int, seed: int, split: str) -> WAMDataset:
    xs = []
    ys = []
    for state_id in range(int(n_states)):
        state = adapter.reset(seed + 4099 * state_id)
        pool = sample_rollout_pool(adapter, state, n_rollouts, horizon, seed + 100_003 * (state_id + 1))
        actions = pool["actions"]
        records = pool["records"]
        x = benchmark_feature(adapter.feature_state(state), actions, horizon)
        final_states = np.asarray([r["final_state"] for r in records], dtype=float)
        utilities = np.asarray([r["utility"] for r in records], dtype=float)
        y = np.column_stack([final_states - state[None, :], utilities])
        xs.append(x)
        ys.append(y)
    return WAMDataset(np.vstack(xs), np.vstack(ys), {"benchmark": adapter.env_id, "split": split, "seed": int(seed)})


def eval_model(model: HorizonWAM, dataset: WAMDataset, split: str) -> dict:
    pred = model.predict(dataset.x)
    err = pred - dataset.y
    util = dataset.y[:, -1]
    pred_util = pred[:, -1]
    corr = float(np.corrcoef(util, pred_util)[0, 1]) if np.std(util) > 1e-12 and np.std(pred_util) > 1e-12 else 0.0
    return {
        "split": split,
        "n_samples": int(len(dataset.x)),
        "state_delta_mae": float(np.mean(np.abs(err[:, :-1]))),
        "utility_mae": float(np.mean(np.abs(err[:, -1]))),
        "utility_corr": corr,
    }


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    adapter = GymManipAdapter(env_id=args.env_id, horizon=args.horizon, success_threshold=args.success_threshold)
    try:
        train = dataset_from_rollouts(adapter, args.train_states, args.train_rollouts, args.horizon, args.seed, "train")
        val = dataset_from_rollouts(adapter, args.val_states, args.val_rollouts, args.horizon, args.seed + 10_000, "validation")
        model = HorizonWAM().fit(train)
        model_path = results_dir() / "models" / "benchmark_gym_manip_horizon_wam.npz"
        model.save(model_path, {"benchmark": adapter.env_id, "seed": args.seed})
        model_metrics = [eval_model(model, train, "train"), eval_model(model, val, "validation")]

        curve_rows = []
        exact_rows = []
        gap_rows = []
        rollout_pool_count = 0
        for seed in args.seeds:
            for state_id in range(args.states):
                state = adapter.reset(seed + 811 * state_id)
                pool = sample_rollout_pool(adapter, state, args.rollouts, args.horizon, seed + 65_537 * (state_id + 1))
                rollout_pool_count += 1
                actions = pool["actions"]
                records = pool["records"]
                real_utility = np.asarray([r["utility"] for r in records], dtype=float)
                success = np.asarray([float(r["success"]) for r in records], dtype=float)
                energy = np.asarray([r["energy"] for r in records], dtype=float)
                x = benchmark_feature(adapter.feature_state(state), actions, args.horizon)
                pred_utility = model.predict(x)[:, -1]
                rng = np.random.default_rng(seed + state_id)
                scorers = {
                    "random": rng.normal(size=len(real_utility)),
                    "low_energy": -energy,
                    "learned_predicted_utility": pred_utility,
                    "oracle_real_utility": real_utility,
                }
                for scorer, scores in scorers.items():
                    u_curve = utility_best_of_n_finite(scores, real_utility, BENCHMARK_N_VALUES)
                    s_curve = binary_best_of_n_finite(scores, success, BENCHMARK_N_VALUES)
                    pred_curve = utility_best_of_n_finite(scores, pred_utility, BENCHMARK_N_VALUES)
                    for n in BENCHMARK_N_VALUES:
                        curve_rows.append(
                            {
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
                                    "seed": int(seed),
                                    "state_id": int(state_id),
                                    "N": int(n),
                                    "predicted_minus_real_utility": pred_curve[n] - u_curve[n],
                                }
                            )
                learned_scores = scorers["learned_predicted_utility"]
                for n in BENCHMARK_N_VALUES:
                    exact_u = utility_best_of_n_finite(learned_scores, real_utility, [n])[n]
                    mc_u = simulate_best_of_n(learned_scores, real_utility, n, args.mc_trials, seed + 17 * n + state_id)
                    exact_rows.append({"seed": int(seed), "state_id": int(state_id), "N": int(n), "utility_abs_error": abs(exact_u - mc_u)})

        curves = pd.DataFrame(curve_rows)
        exact = pd.DataFrame(exact_rows)
        gaps = pd.DataFrame(gap_rows)
        curves_path = results_dir() / "tables" / "benchmark_gym_manip_curves.csv"
        exact_path = results_dir() / "tables" / "benchmark_gym_manip_exact_law.csv"
        gaps_path = results_dir() / "tables" / "benchmark_gym_manip_real_vs_predicted_gap.csv"
        curves.to_csv(curves_path, index=False)
        exact.to_csv(exact_path, index=False)
        gaps.to_csv(gaps_path, index=False)

        agg = curves.groupby(["scorer", "N"]).agg(success=("success", "mean"), real_utility=("real_utility", "mean")).reset_index()
        agg_path = results_dir() / "tables" / "benchmark_gym_manip_curves_aggregate.csv"
        agg.to_csv(agg_path, index=False)

        learned_n32 = curves[(curves["scorer"] == "learned_predicted_utility") & (curves["N"] == 32)].groupby("seed")["real_utility"].mean()
        random_n32 = curves[(curves["scorer"] == "random") & (curves["N"] == 32)].groupby("seed")["real_utility"].mean()
        oracle_n32 = curves[(curves["scorer"] == "oracle_real_utility") & (curves["N"] == 32)].groupby("seed")["real_utility"].mean()
        common = sorted(set(learned_n32.index) & set(random_n32.index) & set(oracle_n32.index))
        learned_minus_random = (learned_n32.loc[common] - random_n32.loc[common]).to_numpy()
        oracle_minus_learned = (oracle_n32.loc[common] - learned_n32.loc[common]).to_numpy()
        oracle_minus_random = (oracle_n32.loc[common] - random_n32.loc[common]).to_numpy()

        closed_rows = []
        for seed in args.seeds:
            for scorer in ["random", "learned", "oracle"]:
                for n in [1, 4, 16, 32]:
                    rec = run_closed_loop(
                        adapter,
                        model,
                        scorer,
                        n,
                        seed + 123_457,
                        steps=args.closed_loop_steps,
                        candidate_horizon=args.closed_loop_horizon,
                        feature_horizon=args.horizon,
                    )
                    rec.update({"seed": int(seed), "scorer": scorer, "N": int(n)})
                    closed_rows.append(rec)
        closed = pd.DataFrame(closed_rows)
        closed_path = results_dir() / "tables" / "benchmark_gym_manip_closed_loop.csv"
        closed.to_csv(closed_path, index=False)
        learned_closed = closed[(closed["scorer"] == "learned") & (closed["N"] == 32)].groupby("seed")["utility"].mean()
        random_closed = closed[(closed["scorer"] == "random") & (closed["N"] == 32)].groupby("seed")["utility"].mean()
        common_closed = sorted(set(learned_closed.index) & set(random_closed.index))
        closed_delta = (learned_closed.loc[common_closed] - random_closed.loc[common_closed]).to_numpy()

        plt.figure(figsize=(7, 4.5))
        for scorer, sub in agg.groupby("scorer"):
            plt.plot(sub["N"], sub["real_utility"], marker="o", label=scorer)
        plt.xscale("log", base=2)
        plt.xlabel("N")
        plt.ylabel("real utility")
        plt.title(f"Gymnasium/MuJoCo benchmark: {adapter.env_id}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fig_path = results_dir() / "figures" / "benchmark_gym_manip_curves.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()

        exact_mae = float(exact["utility_abs_error"].mean())
        gap_growth = float(
            gaps[gaps["N"] == 32]["predicted_minus_real_utility"].mean()
            - gaps[gaps["N"] == 1]["predicted_minus_real_utility"].mean()
        )
        summary = {
            "experiment": "benchmark_gym_manip_suite",
            "benchmark": adapter.env_id,
            "attempted": True,
            "available": True,
            "n_rollout_pools": int(rollout_pool_count),
            "n_rollouts": int(args.rollouts),
            "seeds": [int(s) for s in args.seeds],
            "N_values": BENCHMARK_N_VALUES,
            "model_path": str(model_path),
            "model_metrics": model_metrics,
            "exact_law_utility_mae": exact_mae,
            "score_comparison": {
                "learned_minus_random_real_utility_N32": float(np.mean(learned_minus_random)),
                "oracle_minus_learned_real_utility_N32": float(np.mean(oracle_minus_learned)),
                "oracle_minus_random_real_utility_N32": float(np.mean(oracle_minus_random)),
            },
            "real_vs_predicted_gap": {
                "gap_growth_N32_minus_N1": gap_growth,
            },
            "closed_loop": {
                "learned_minus_random_utility_N32": float(np.mean(closed_delta)),
            },
            "confidence_intervals": {
                "learned_minus_random_real_utility_N32": ci95(learned_minus_random),
                "oracle_minus_learned_real_utility_N32": ci95(oracle_minus_learned),
                "oracle_minus_random_real_utility_N32": ci95(oracle_minus_random),
                "closed_loop_learned_minus_random_utility_N32": ci95(closed_delta),
            },
            "artifacts": {
                "curves": str(curves_path),
                "curves_aggregate": str(agg_path),
                "exact_law": str(exact_path),
                "real_vs_predicted_gap": str(gaps_path),
                "closed_loop": str(closed_path),
                "figure": str(fig_path),
            },
        }
        write_json(results_dir() / "benchmark_gym_manip_suite.json", summary)
        write_json(results_dir() / "benchmark_rollout_pools.json", {"benchmark": adapter.env_id, "n_rollout_pools": int(rollout_pool_count), "n_rollouts": int(args.rollouts)})
        write_json(results_dir() / "benchmark_wam_training.json", {"benchmark": adapter.env_id, "model_path": str(model_path), "model_metrics": model_metrics})
        write_json(results_dir() / "benchmark_exact_law_validation.json", {"benchmark": adapter.env_id, "utility_mae": exact_mae, "artifact": str(exact_path)})
        write_json(results_dir() / "benchmark_score_comparison.json", summary["score_comparison"] | {"confidence_intervals": summary["confidence_intervals"]})
        write_json(results_dir() / "benchmark_real_vs_imagined_gap.json", summary["real_vs_predicted_gap"] | {"artifact": str(gaps_path)})
        write_json(results_dir() / "benchmark_closed_loop_eval.json", summary["closed_loop"] | {"confidence_intervals": summary["confidence_intervals"], "artifact": str(closed_path)})
        return summary
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Reacher-v5")
    parser.add_argument("--success-threshold", type=float, default=0.07)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--train-states", type=int, default=12)
    parser.add_argument("--train-rollouts", type=int, default=96)
    parser.add_argument("--val-states", type=int, default=4)
    parser.add_argument("--val-rollouts", type=int, default=64)
    parser.add_argument("--closed-loop-steps", type=int, default=20)
    parser.add_argument("--closed-loop-horizon", type=int, default=12)
    parser.add_argument("--mc-trials", type=int, default=800)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[9001, 9002, 9003, 9004, 9005])
    args = parser.parse_args()
    summary = run(args)
    print(
        "benchmark gym-manip complete: "
        f"benchmark={summary['benchmark']}, exact_mae={summary['exact_law_utility_mae']:.4f}, "
        f"learned-random N32={summary['score_comparison']['learned_minus_random_real_utility_N32']:.4f}"
    )


if __name__ == "__main__":
    main()
