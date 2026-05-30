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
    add_backend_args,
    backend_suffix,
    ci95,
    marginal_greedy_allocate,
    results_dir,
    load_model_for_backend,
    seed_list_from_args,
    uniform_allocate,
    write_json,
    ensure_result_dirs,
)
from wam_inference_value.rollouts import generate_rollout_pools
from wam_inference_value.scorers import scores_for_pool
from wam_inference_value.theorem import auc_kappa, auc_only_constant_moment_curve, binary_best_of_n_finite


def p_coverage_curve(p: float, max_n: int) -> np.ndarray:
    ns = np.arange(1, max_n + 1, dtype=float)
    return 1.0 - np.power(1.0 - np.clip(p, 0.0, 1.0), ns)


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    dynamics_backend = getattr(args, "dynamics_backend", "analytic")
    learned_model = load_model_for_backend(args)
    seeds = seed_list_from_args(args)
    rows = []
    policies = {
        "uniform": None,
        "coverage_p_only": "coverage_curve",
        "auc_kappa": "auc_curve",
        "moment_law": "moment_curve",
        "uncertainty": "uncertainty_curve",
        "oracle": "oracle_curve",
    }
    for seed_id, seed in enumerate(seeds):
        pools = generate_rollout_pools(
            args.states,
            args.rollouts,
            args.mismatch,
            seed,
            dynamics_backend=dynamics_backend,
            learned_model=learned_model,
        )
        rng = np.random.default_rng(seed)
        eval_items = []
        for pool in pools:
            scores = scores_for_pool(pool, args.scorer)
            success = pool.real_success
            perm = rng.permutation(len(scores))
            pilot_idx = perm[: args.pilot_k]
            held_idx = perm[args.pilot_k :]
            max_n = min(args.max_n, len(held_idx))
            n_values = list(range(1, max_n + 1))
            pilot_scores = scores[pilot_idx]
            pilot_success = success[pilot_idx]
            held_scores = scores[held_idx]
            held_success = success[held_idx]
            pilot_curve_lookup = binary_best_of_n_finite(pilot_scores, pilot_success, [min(n, args.pilot_k) for n in n_values])
            moment_curve = np.asarray([pilot_curve_lookup[min(n, args.pilot_k)] for n in n_values], dtype=float)
            moment_curve = np.maximum.accumulate(moment_curve)
            held_lookup = binary_best_of_n_finite(held_scores, held_success, n_values)
            held_curve = np.asarray([held_lookup[n] for n in n_values], dtype=float)
            p_hat = float(np.mean(pilot_success))
            kappa = auc_kappa(pilot_scores, pilot_success)
            if not np.isfinite(kappa):
                auc_curve = p_coverage_curve(p_hat, max_n)
            else:
                auc_lookup = auc_only_constant_moment_curve(p_hat, kappa, n_values)
                auc_curve = np.maximum.accumulate(np.asarray([auc_lookup[n] for n in n_values], dtype=float))
            uncertainty_curve = np.maximum.accumulate(moment_curve + 0.08 * float(np.std(pilot_scores)))
            uncertainty_curve = np.clip(uncertainty_curve, 0.0, 1.0)
            eval_items.append(
                {
                    "held_curve": held_curve,
                    "moment_curve": moment_curve,
                    "coverage_curve": p_coverage_curve(p_hat, max_n),
                    "auc_curve": auc_curve,
                    "uncertainty_curve": uncertainty_curve,
                    "oracle_curve": held_curve,
                    "max_n": max_n,
                }
            )

        for mean_budget in args.mean_budgets:
            total_budget = int(mean_budget) * len(eval_items)
            for policy, key in policies.items():
                if policy == "uniform":
                    alloc = uniform_allocate(len(eval_items), total_budget, min(x["max_n"] for x in eval_items))
                else:
                    alloc = marginal_greedy_allocate([x[key] for x in eval_items], total_budget)
                actual = []
                for item, n in zip(eval_items, alloc):
                    n = max(1, min(int(n), len(item["held_curve"])))
                    actual.append(float(item["held_curve"][n - 1]))
                rows.append(
                    {
                        "seed": int(seed),
                        "seed_id": int(seed_id),
                        "policy": policy,
                        "mean_budget": int(mean_budget),
                        "success": float(np.mean(actual)),
                        "mean_allocated_samples": float(np.mean(alloc)),
                        "allocation_std": float(np.std(alloc)),
                        "pilot_k": int(args.pilot_k),
                        "num_states": len(eval_items),
                    }
                )

    df = pd.DataFrame(rows)
    suffix = backend_suffix(args)
    table_path = results_dir() / "tables" / f"exp6_adaptive_rollout_allocation{suffix}.csv"
    df.to_csv(table_path, index=False)
    plot_df = df.groupby(["policy", "mean_budget"], dropna=False).agg(success=("success", "mean")).reset_index()
    plt.figure(figsize=(7.4, 4.7))
    for policy, sub in plot_df.groupby("policy"):
        plt.plot(sub["mean_budget"], sub["success"], marker="o", label=policy)
    plt.xscale("log", base=2)
    plt.xlabel("mean rollout budget per state")
    plt.ylabel("heldout selected success")
    plt.title("Adaptive rollout allocation under a fixed inference budget")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / f"exp6_adaptive_allocation{suffix}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    ref_budget = max([b for b in args.mean_budgets if b <= 32], default=args.mean_budgets[-1])
    ref = df[df["mean_budget"] == ref_budget].groupby("policy", dropna=False).mean(numeric_only=True).reset_index()
    ref = ref.sort_values("success", ascending=False)
    uniform = float(ref[ref["policy"] == "uniform"]["success"].iloc[0])
    moment = float(ref[ref["policy"] == "moment_law"]["success"].iloc[0])
    oracle = float(ref[ref["policy"] == "oracle"]["success"].iloc[0])
    seed_metric_rows = []
    for seed, sub in df[df["mean_budget"] == ref_budget].groupby("seed"):
        uniform_seed = float(sub[sub["policy"] == "uniform"]["success"].iloc[0])
        moment_seed = float(sub[sub["policy"] == "moment_law"]["success"].iloc[0])
        oracle_seed = float(sub[sub["policy"] == "oracle"]["success"].iloc[0])
        seed_metric_rows.append(
            {
                "seed": int(seed),
                "moment_law_improvement_over_uniform": moment_seed - uniform_seed,
                "oracle_improvement_over_uniform": oracle_seed - uniform_seed,
            }
        )
    seed_metric_df = pd.DataFrame(seed_metric_rows)
    summary = {
        "experiment": "adaptive_rollout_allocation",
        "dynamics_backend": dynamics_backend,
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "pilot_k": args.pilot_k,
        "mismatch": args.mismatch,
        "scorer": args.scorer,
        "mean_budgets": args.mean_budgets,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "reference_budget": ref_budget,
        "ranking_at_reference_budget": ref.to_dict(orient="records"),
        "moment_law_improvement_over_uniform": moment - uniform,
        "oracle_improvement_over_uniform": oracle - uniform,
        "confidence_intervals": {
            "moment_law_improvement_over_uniform": ci95(seed_metric_df["moment_law_improvement_over_uniform"].to_numpy()),
            "oracle_improvement_over_uniform": ci95(seed_metric_df["oracle_improvement_over_uniform"].to_numpy()),
        },
        "artifacts": {"table": str(table_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / f"exp6_adaptive_rollout_allocation{suffix}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=120)
    parser.add_argument("--rollouts", type=int, default=384)
    parser.add_argument("--pilot-k", type=int, default=64)
    parser.add_argument("--max-n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--mismatch", type=str, default="mild")
    parser.add_argument("--scorer", type=str, default="predicted_utility")
    parser.add_argument("--mean-budgets", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    add_backend_args(parser)
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp6 complete: "
        f"moment-uniform delta={summary['moment_law_improvement_over_uniform']:.3f}, "
        f"oracle-uniform delta={summary['oracle_improvement_over_uniform']:.3f}"
    )


if __name__ == "__main__":
    main()
