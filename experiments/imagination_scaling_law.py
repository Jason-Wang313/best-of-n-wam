from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wam_inference_value.audit import classify_profile, inference_value_profile, stop_rule
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, seed_list_from_args, write_json
from wam_inference_value.rollouts import generate_rollout_pools
from wam_inference_value.scorers import scores_for_pool


N_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
SCORERS = ["predicted_utility", "uncertainty_penalized", "random_score", "oracle_real_utility"]


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    seeds = seed_list_from_args(args)
    rows = []
    profile_rows = []
    for seed in seeds:
        for mismatch in args.mismatches:
            for horizon in args.horizons:
                for pool_size in args.pool_sizes:
                    started = time.perf_counter()
                    pools = generate_rollout_pools(
                        args.states,
                        pool_size,
                        mismatch,
                        seed,
                        horizon=horizon,
                    )
                    elapsed = time.perf_counter() - started
                    seconds_per_rollout = elapsed / max(1, args.states * pool_size)
                    for pool in pools:
                        real = pool.real_utility
                        for scorer in SCORERS:
                            scores = scores_for_pool(pool, scorer)
                            profile = inference_value_profile(scores, real, N_GRID, normalize=True)
                            cls = classify_profile(profile)
                            stop_n = stop_rule(profile, compute_cost_per_rollout=args.compute_cost)
                            profile_rows.append(
                                {
                                    "seed": int(seed),
                                    "state_id": int(pool.state_id),
                                    "mismatch": mismatch,
                                    "horizon": int(horizon),
                                    "pool_size": int(pool_size),
                                    "scorer": scorer,
                                    "profile_class": cls,
                                    "stop_n": int(stop_n),
                                    "gain_N128_minus_N1": float(profile["curve"][128] - profile["curve"][1]),
                                    "value_N1": float(profile["curve"][1]),
                                    "value_N128": float(profile["curve"][128]),
                                    "seconds_per_rollout_estimate": float(seconds_per_rollout),
                                }
                            )
                            for row in profile["rows"]:
                                N = int(row["N"])
                                rollout_steps = int(N * horizon)
                                rows.append(
                                    {
                                        "seed": int(seed),
                                        "state_id": int(pool.state_id),
                                        "mismatch": mismatch,
                                        "horizon": int(horizon),
                                        "pool_size": int(pool_size),
                                        "scorer": scorer,
                                        "N": N,
                                        "normalized_real_utility": float(row["value"]),
                                        "delta_value": float(row["delta_value"]),
                                        "delta_per_rollout": float(row["delta_per_rollout"]),
                                        "rollout_steps_proxy": rollout_steps,
                                        "wall_clock_seconds_proxy": float(seconds_per_rollout * N),
                                    }
                                )

    df = pd.DataFrame(rows)
    profiles = pd.DataFrame(profile_rows)
    table_path = results_dir() / "tables" / "imagination_scaling_law.csv"
    profile_path = results_dir() / "tables" / "imagination_scaling_profiles.csv"
    df.to_csv(table_path, index=False)
    profiles.to_csv(profile_path, index=False)
    agg = df.groupby(["mismatch", "horizon", "pool_size", "scorer", "N"], dropna=False)[
        ["normalized_real_utility", "delta_value", "delta_per_rollout", "rollout_steps_proxy", "wall_clock_seconds_proxy"]
    ].mean().reset_index()
    agg_path = results_dir() / "tables" / "imagination_scaling_law_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    seed_metrics = []
    for seed, sub in profiles.groupby("seed"):
        pred = sub[sub["scorer"] == "predicted_utility"]
        rand = sub[sub["scorer"] == "random_score"]
        oracle = sub[sub["scorer"] == "oracle_real_utility"]
        seed_metrics.append(
            {
                "seed": int(seed),
                "predicted_gain_N128_minus_N1": float(np.mean(pred["gain_N128_minus_N1"])),
                "random_gain_N128_minus_N1": float(np.mean(rand["gain_N128_minus_N1"])),
                "oracle_gain_N128_minus_N1": float(np.mean(oracle["gain_N128_minus_N1"])),
                "oracle_minus_predicted_gain": float(np.mean(oracle["gain_N128_minus_N1"]) - np.mean(pred["gain_N128_minus_N1"])),
                "median_stop_n_predicted": float(np.median(pred["stop_n"])),
                "helpful_profile_rate_predicted": float(np.mean(pred["profile_class"] == "helpful")),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_path = results_dir() / "tables" / "imagination_scaling_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False)

    plt.figure(figsize=(7.6, 4.8))
    plot_sub = agg[(agg["mismatch"] == "mild") & (agg["horizon"] == max(args.horizons)) & (agg["pool_size"] == max(args.pool_sizes))]
    for scorer, sub in plot_sub.groupby("scorer"):
        plt.plot(sub["rollout_steps_proxy"], sub["normalized_real_utility"], marker="o", label=scorer)
    plt.xscale("log", base=2)
    plt.xlabel("rollout-step compute proxy: N x horizon")
    plt.ylabel("normalized real utility")
    plt.title("Compute-quality frontier for robot imagination")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "imagination_scaling_frontier.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    summary = {
        "experiment": "imagination_scaling_law",
        "states": args.states,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "mismatches": args.mismatches,
        "horizons": args.horizons,
        "pool_sizes": args.pool_sizes,
        "N_values": N_GRID,
        "scorers": SCORERS,
        "confidence_intervals": {
            "predicted_gain_N128_minus_N1": ci95(seed_df["predicted_gain_N128_minus_N1"].to_numpy()),
            "random_gain_N128_minus_N1": ci95(seed_df["random_gain_N128_minus_N1"].to_numpy()),
            "oracle_gain_N128_minus_N1": ci95(seed_df["oracle_gain_N128_minus_N1"].to_numpy()),
            "oracle_minus_predicted_gain": ci95(seed_df["oracle_minus_predicted_gain"].to_numpy()),
            "median_stop_n_predicted": ci95(seed_df["median_stop_n_predicted"].to_numpy()),
            "helpful_profile_rate_predicted": ci95(seed_df["helpful_profile_rate_predicted"].to_numpy()),
        },
        "artifacts": {"table": str(table_path), "profiles": str(profile_path), "aggregate": str(agg_path), "seed_metrics": str(seed_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / "imagination_scaling_law.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--mismatches", nargs="*", default=["mild", "severe"])
    parser.add_argument("--horizons", nargs="*", type=int, default=[4, 8, 12])
    parser.add_argument("--pool-sizes", nargs="*", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--compute-cost", type=float, default=0.0015)
    args = parser.parse_args()
    summary = run(args)
    ci = summary["confidence_intervals"]["predicted_gain_N128_minus_N1"]
    print(f"imagination scaling complete: predicted gain mean={ci['mean']:.3f}, lo={ci['lo']:.3f}")


if __name__ == "__main__":
    main()
