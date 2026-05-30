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

from wam_inference_value.evaluation import (
    N_VALUES,
    add_backend_args,
    backend_suffix,
    ci95,
    ensure_result_dirs,
    load_model_for_backend,
    results_dir,
    seed_list_from_args,
    write_json,
)
from wam_inference_value.rollouts import generate_rollout_pools
from wam_inference_value.scorers import scores_for_pool
from wam_inference_value.theorem import (
    auc_kappa,
    binary_best_of_n_finite,
    simulate_best_of_n,
    tie_rate,
    utility_best_of_n_finite,
)


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    dynamics_backend = getattr(args, "dynamics_backend", "analytic")
    learned_model = load_model_for_backend(args)
    seeds = seed_list_from_args(args)
    rows = []
    for seed_id, seed in enumerate(seeds):
        pools = generate_rollout_pools(
            args.states,
            args.rollouts,
            args.mismatch,
            seed,
            dynamics_backend=dynamics_backend,
            learned_model=learned_model,
        )
        for pool in pools:
            scores = scores_for_pool(pool, args.scorer)
            success = pool.real_success
            utility = pool.real_utility
            exact_success = binary_best_of_n_finite(scores, success, N_VALUES)
            exact_utility = utility_best_of_n_finite(scores, utility, N_VALUES)
            for N in N_VALUES:
                mc_success = simulate_best_of_n(scores, success, N, args.mc_trials, seed + 31 * pool.state_id + N)
                mc_utility = simulate_best_of_n(scores, utility, N, args.mc_trials, seed + 131 * pool.state_id + N)
                rows.append(
                    {
                        "seed": int(seed),
                        "seed_id": int(seed_id),
                        "state_id": pool.state_id,
                        "N": N,
                        "p": float(np.mean(success)),
                        "kappa": auc_kappa(scores, success),
                        "tie_rate": tie_rate(scores),
                        "exact_success": exact_success[N],
                        "mc_success": mc_success,
                        "success_abs_error": abs(exact_success[N] - mc_success),
                        "exact_utility": exact_utility[N],
                        "mc_utility": mc_utility,
                        "utility_abs_error": abs(exact_utility[N] - mc_utility),
                    }
                )

    df = pd.DataFrame(rows)
    suffix = backend_suffix(args)
    table_path = results_dir() / "tables" / f"exp1_exact_rollout_law_validation{suffix}.csv"
    df.to_csv(table_path, index=False)

    by_n = df.groupby("N").agg(
        success_mae=("success_abs_error", "mean"),
        utility_mae=("utility_abs_error", "mean"),
        exact_success=("exact_success", "mean"),
        mc_success=("mc_success", "mean"),
        exact_utility=("exact_utility", "mean"),
        mc_utility=("mc_utility", "mean"),
    ).reset_index()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(by_n["N"], by_n["exact_success"], marker="o", label="finite law")
    plt.plot(by_n["N"], by_n["mc_success"], marker="s", linestyle="--", label="Monte Carlo")
    plt.xscale("log", base=2)
    plt.xticks(N_VALUES, [str(n) for n in N_VALUES])
    plt.xlabel("rollouts sampled at test time (N)")
    plt.ylabel("selected real success")
    plt.title("Exact finite law vs Monte Carlo sanity check")
    plt.legend()
    plt.tight_layout()
    fig_path = results_dir() / "figures" / f"exp1_exact_vs_mc_success{suffix}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    by_seed = df.groupby("seed").agg(
        success_mae=("success_abs_error", "mean"),
        utility_mae=("utility_abs_error", "mean"),
    ).reset_index()
    summary = {
        "experiment": "exact_rollout_law_validation",
        "dynamics_backend": dynamics_backend,
        "mismatch": args.mismatch,
        "scorer": args.scorer,
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "mc_trials": args.mc_trials,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "mean_success_mc_mae": float(df["success_abs_error"].mean()),
        "max_success_mc_error": float(df["success_abs_error"].max()),
        "mean_utility_mc_mae": float(df["utility_abs_error"].mean()),
        "max_utility_mc_error": float(df["utility_abs_error"].max()),
        "mean_tie_rate": float(df["tie_rate"].mean()),
        "confidence_intervals": {
            "mean_success_mc_mae": ci95(by_seed["success_mae"].to_numpy()),
            "mean_utility_mc_mae": ci95(by_seed["utility_mae"].to_numpy()),
        },
        "by_n": by_n.to_dict(orient="records"),
        "artifacts": {"table": str(table_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / f"exp1_exact_rollout_law_validation{suffix}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=72)
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--mc-trials", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mismatch", type=str, default="mild")
    parser.add_argument("--scorer", type=str, default="predicted_utility")
    add_backend_args(parser)
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp1 complete: "
        f"success MAE={summary['mean_success_mc_mae']:.5f}, "
        f"utility MAE={summary['mean_utility_mc_mae']:.5f}"
    )


if __name__ == "__main__":
    main()
