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
    aggregate_curve_table,
    area_under_inference_curve,
    backend_suffix,
    ci95,
    curve_rows_for_pool,
    ensure_result_dirs,
    load_model_for_backend,
    results_dir,
    seed_list_from_args,
    write_json,
)
from wam_inference_value.rollouts import generate_rollout_pools
from wam_inference_value.scorers import SCORERS


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
            for scorer in SCORERS:
                for row in curve_rows_for_pool(pool, scorer, N_VALUES):
                    row.update({"seed": int(seed), "seed_id": int(seed_id)})
                    rows.append(row)
    df = pd.DataFrame(rows)
    suffix = backend_suffix(args)
    table_path = results_dir() / "tables" / f"exp4_score_function_comparison{suffix}.csv"
    df.to_csv(table_path, index=False)
    agg = aggregate_curve_table(rows, ["scorer", "N"])
    agg_path = results_dir() / "tables" / f"exp4_score_function_comparison_aggregate{suffix}.csv"
    agg.to_csv(agg_path, index=False)

    ranking_rows = []
    for scorer, sub in agg.groupby("scorer"):
        n64 = sub[sub["N"] == 64].iloc[0]
        n1 = sub[sub["N"] == 1].iloc[0]
        ranking_rows.append(
            {
                "scorer": scorer,
                "real_utility_N1": float(n1["real_utility"]),
                "real_utility_N64": float(n64["real_utility"]),
                "success_N64": float(n64["success"]),
                "mean_kappa": float(sub["kappa"].mean()),
                "area_real_utility": area_under_inference_curve(sub, "real_utility"),
                "gain_N64_minus_N1": float(n64["real_utility"] - n1["real_utility"]),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values("real_utility_N64", ascending=False)
    ranking_path = results_dir() / "tables" / f"exp4_scorer_ranking{suffix}.csv"
    ranking.to_csv(ranking_path, index=False)

    plt.figure(figsize=(7.6, 4.8))
    for scorer, sub in agg.groupby("scorer"):
        style = "--" if scorer == "random_score" else "-"
        plt.plot(sub["N"], sub["real_utility"], marker="o", linestyle=style, label=scorer)
    plt.xscale("log", base=2)
    plt.xticks(N_VALUES, [str(n) for n in N_VALUES])
    plt.xlabel("N")
    plt.ylabel("selected real utility")
    plt.title("Score functions control the value of test-time imagination")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / f"exp4_score_function_curves{suffix}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    oracle_n64 = float(ranking[ranking["scorer"] == "oracle_real_utility"]["real_utility_N64"].iloc[0])
    random_n64 = float(ranking[ranking["scorer"] == "random_score"]["real_utility_N64"].iloc[0])
    best_nonoracle = ranking[ranking["scorer"] != "oracle_real_utility"].iloc[0].to_dict()
    seed_metrics = []
    seed_agg = df.groupby(["seed", "scorer", "N"], dropna=False)[["real_utility", "normalized_real_utility", "success"]].mean().reset_index()
    for seed, sub in seed_agg.groupby("seed"):
        n64 = sub[sub["N"] == 64]
        seed_oracle = float(n64[n64["scorer"] == "oracle_real_utility"]["real_utility"].iloc[0])
        seed_random = float(n64[n64["scorer"] == "random_score"]["real_utility"].iloc[0])
        seed_best = n64[n64["scorer"] != "oracle_real_utility"].sort_values("real_utility", ascending=False).iloc[0]
        seed_metrics.append(
            {
                "seed": int(seed),
                "oracle_minus_random_N64": seed_oracle - seed_random,
                "best_nonoracle_minus_random_N64": float(seed_best["real_utility"] - seed_random),
                "best_nonoracle_normalized_real_utility_N64": float(seed_best["normalized_real_utility"]),
            }
        )
    seed_metric_df = pd.DataFrame(seed_metrics)
    summary = {
        "experiment": "score_function_comparison",
        "dynamics_backend": dynamics_backend,
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "mismatch": args.mismatch,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "scorers": SCORERS,
        "ranking": ranking.to_dict(orient="records"),
        "best_nonoracle_scorer": best_nonoracle,
        "oracle_minus_random_N64": oracle_n64 - random_n64,
        "best_nonoracle_minus_random_N64": float(best_nonoracle["real_utility_N64"] - random_n64),
        "confidence_intervals": {
            "oracle_minus_random_N64": ci95(seed_metric_df["oracle_minus_random_N64"].to_numpy()),
            "best_nonoracle_minus_random_N64": ci95(seed_metric_df["best_nonoracle_minus_random_N64"].to_numpy()),
            "best_nonoracle_normalized_real_utility_N64": ci95(
                seed_metric_df["best_nonoracle_normalized_real_utility_N64"].to_numpy()
            ),
        },
        "artifacts": {"table": str(table_path), "aggregate": str(agg_path), "ranking": str(ranking_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / f"exp4_score_function_comparison{suffix}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=96)
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--mismatch", type=str, default="mild")
    add_backend_args(parser)
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp4 complete: "
        f"best non-oracle={summary['best_nonoracle_scorer']['scorer']}, "
        f"best-random N64 delta={summary['best_nonoracle_minus_random_N64']:.3f}"
    )


if __name__ == "__main__":
    main()
