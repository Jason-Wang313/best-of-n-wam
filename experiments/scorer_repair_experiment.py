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

from wam_inference_value.evaluation import N_VALUES, ci95, ensure_result_dirs, results_dir, seed_list_from_args, write_json
from wam_inference_value.rollouts import RolloutPool, generate_rollout_pools
from wam_inference_value.scorers import scores_for_pool
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import utility_best_of_n_finite


def feature_matrix(pool: RolloutPool) -> np.ndarray:
    records = pool.records
    predicted_utility = scores_for_pool(pool, "predicted_utility")
    predicted_distance = scores_for_pool(pool, "predicted_goal_distance")
    safety_penalized = scores_for_pool(pool, "safety_penalized")
    uncertainty_penalized = scores_for_pool(pool, "uncertainty_penalized")
    imagined_success = np.asarray([float(r.imagined.success) for r in records], dtype=float)
    imagined_distance = np.asarray([r.imagined.final_distance for r in records], dtype=float)
    imagined_energy = np.asarray([r.imagined.energy for r in records], dtype=float)
    mean_action_norm = np.asarray([r.mean_action_norm for r in records], dtype=float)
    max_action_norm = np.asarray([r.max_action_norm for r in records], dtype=float)
    safety = np.asarray([r.imagined.safety_violation for r in records], dtype=float)
    return np.column_stack(
        [
            predicted_utility,
            predicted_distance,
            safety_penalized,
            uncertainty_penalized,
            imagined_success,
            imagined_distance,
            imagined_energy,
            mean_action_norm,
            max_action_norm,
            safety,
        ]
    )


def fit_ridge_score(features: np.ndarray, targets: np.ndarray, pilot_idx: np.ndarray, eval_idx: np.ndarray, ridge: float) -> np.ndarray:
    x_train = features[pilot_idx]
    y_train = targets[pilot_idx]
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std <= 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_eval = (features[eval_idx] - mean) / std
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_eval = np.column_stack([np.ones(len(x_eval)), x_eval])
    penalty = float(ridge) * np.eye(x_train.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return np.asarray(x_eval @ weights, dtype=float)


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    seeds = seed_list_from_args(args)
    mismatches = args.mismatches or ["severe", "stuck_slip", "nonstationary"]
    rows = []
    seed_rows = []
    for seed in seeds:
        for mismatch in mismatches:
            pools = generate_rollout_pools(args.states, args.rollouts, mismatch, seed)
            for pool in pools:
                rng = np.random.default_rng(seed + 31337 * (pool.state_id + 1))
                indices = rng.permutation(len(pool.records))
                pilot_k = min(args.pilot_k, max(4, len(indices) // 3))
                pilot_idx = indices[:pilot_k]
                eval_idx = indices[pilot_k:]
                features = feature_matrix(pool)
                real = pool.real_utility
                real_eval = real[eval_idx]
                norm_real_eval = normalized_utility(real_eval)
                repaired = fit_ridge_score(features, real, pilot_idx, eval_idx, args.ridge)
                score_sets = {
                    "predicted_utility": scores_for_pool(pool, "predicted_utility")[eval_idx],
                    "safety_penalized": scores_for_pool(pool, "safety_penalized")[eval_idx],
                    "uncertainty_penalized": scores_for_pool(pool, "uncertainty_penalized")[eval_idx],
                    "pilot_calibrated_repair": repaired,
                    "random_score": scores_for_pool(pool, "random_score")[eval_idx],
                    "anti_real_utility": -real_eval,
                    "oracle_real_utility": real_eval,
                }
                for scorer, scores in score_sets.items():
                    curve = utility_best_of_n_finite(scores, norm_real_eval, N_VALUES)
                    raw_curve = utility_best_of_n_finite(scores, real_eval, N_VALUES)
                    for N in N_VALUES:
                        rows.append(
                            {
                                "seed": int(seed),
                                "state_id": int(pool.state_id),
                                "mismatch": mismatch,
                                "scorer": scorer,
                                "N": int(N),
                                "normalized_real_utility": float(curve[N]),
                                "real_utility": float(raw_curve[N]),
                                "pilot_k": int(pilot_k),
                                "heldout_n": int(len(eval_idx)),
                            }
                        )
    df = pd.DataFrame(rows)
    table_path = results_dir() / "tables" / "scorer_repair_experiment.csv"
    df.to_csv(table_path, index=False)
    agg = df.groupby(["scorer", "mismatch", "N"], dropna=False)[["normalized_real_utility", "real_utility"]].mean().reset_index()
    agg_path = results_dir() / "tables" / "scorer_repair_experiment_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    seed_agg = df.groupby(["seed", "scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
    for seed, sub in seed_agg.groupby("seed"):
        n64 = sub[sub["N"] == 64]
        lookup = {str(r["scorer"]): float(r["normalized_real_utility"]) for _, r in n64.iterrows()}
        seed_rows.append(
            {
                "seed": int(seed),
                "repair_minus_predicted_N64": lookup["pilot_calibrated_repair"] - lookup["predicted_utility"],
                "repair_minus_random_N64": lookup["pilot_calibrated_repair"] - lookup["random_score"],
                "oracle_minus_repair_N64": lookup["oracle_real_utility"] - lookup["pilot_calibrated_repair"],
                "repair_minus_anti_N64": lookup["pilot_calibrated_repair"] - lookup["anti_real_utility"],
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_path = results_dir() / "tables" / "scorer_repair_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)

    plt.figure(figsize=(7.4, 4.8))
    for scorer, sub in agg[agg["mismatch"] == "severe"].groupby("scorer"):
        if scorer in {"pilot_calibrated_repair", "predicted_utility", "safety_penalized", "random_score", "oracle_real_utility", "anti_real_utility"}:
            style = "--" if scorer in {"random_score", "anti_real_utility"} else "-"
            plt.plot(sub["N"], sub["normalized_real_utility"], marker="o", linestyle=style, label=scorer)
    plt.xscale("log", base=2)
    plt.xticks(N_VALUES, [str(n) for n in N_VALUES])
    plt.xlabel("N")
    plt.ylabel("heldout normalized real utility")
    plt.title("Pilot-calibrated scorer repair under mismatch")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "scorer_repair_experiment.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    summary = {
        "experiment": "scorer_repair_experiment",
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "pilot_k": args.pilot_k,
        "ridge": args.ridge,
        "mismatches": mismatches,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "confidence_intervals": {
            "repair_minus_predicted_N64": ci95(seed_df["repair_minus_predicted_N64"].to_numpy()),
            "repair_minus_random_N64": ci95(seed_df["repair_minus_random_N64"].to_numpy()),
            "oracle_minus_repair_N64": ci95(seed_df["oracle_minus_repair_N64"].to_numpy()),
            "repair_minus_anti_N64": ci95(seed_df["repair_minus_anti_N64"].to_numpy()),
        },
        "artifacts": {"table": str(table_path), "aggregate": str(agg_path), "seed_metrics": str(seed_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / "scorer_repair_experiment.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--rollouts", type=int, default=192)
    parser.add_argument("--pilot-k", type=int, default=48)
    parser.add_argument("--ridge", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--mismatches", nargs="*", default=["severe", "stuck_slip", "nonstationary"])
    args = parser.parse_args()
    summary = run(args)
    ci = summary["confidence_intervals"]["repair_minus_predicted_N64"]
    print(f"scorer repair complete: repair-predicted N64 CI mean={ci['mean']:.3f}, lo={ci['lo']:.3f}")


if __name__ == "__main__":
    main()
