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

from wam_inference_value.evaluation import N_VALUES, ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.rollouts import generate_rollout_pools
from wam_inference_value.scorers import scores_for_pool
from wam_inference_value.theorem import utility_best_of_n_finite


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    k_values = [k for k in [8, 16, 32, 64, 128] if k < args.rollouts]
    pools = generate_rollout_pools(args.states, args.rollouts, args.mismatch, args.seed)
    rows = []
    for seed_id in range(args.splits):
        for pool in pools:
            scores = scores_for_pool(pool, args.scorer)
            utility = pool.real_utility
            rng = np.random.default_rng(args.seed + seed_id * 1_000_003 + pool.state_id)
            perm = rng.permutation(len(scores))
            for K in k_values:
                pilot_idx = perm[:K]
                held_idx = perm[K:]
                pilot_curve = utility_best_of_n_finite(scores[pilot_idx], utility[pilot_idx], N_VALUES)
                held_ns = [N for N in N_VALUES if N <= len(held_idx)]
                held_curve = utility_best_of_n_finite(scores[held_idx], utility[held_idx], held_ns)
                for N in held_ns:
                    rows.append(
                        {
                            "split_seed": seed_id,
                            "state_id": pool.state_id,
                            "K": K,
                            "N": N,
                            "pilot_pred": pilot_curve[N],
                            "heldout_actual": held_curve[N],
                            "abs_error": abs(pilot_curve[N] - held_curve[N]),
                        }
                    )

    df = pd.DataFrame(rows)
    table_path = results_dir() / "tables" / "exp3_pilot_to_heldout_prediction.csv"
    df.to_csv(table_path, index=False)
    by_k = df.groupby(["K", "N"]).agg(heldout_mae=("abs_error", "mean")).reset_index()

    plt.figure(figsize=(7.2, 4.6))
    for N, sub in by_k.groupby("N"):
        if N in [2, 8, 32, 64]:
            plt.plot(sub["K"], sub["heldout_mae"], marker="o", label=f"N={N}")
    plt.xscale("log", base=2)
    plt.xlabel("pilot rollouts K")
    plt.ylabel("heldout MAE")
    plt.title("Pilot-to-heldout curve estimation improves with larger pilots")
    plt.legend()
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "exp3_pilot_to_heldout_mae.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    overall_by_k = df.groupby("K").agg(mae=("abs_error", "mean")).reset_index().sort_values("K")
    first_mae = float(overall_by_k.iloc[0]["mae"])
    last_mae = float(overall_by_k.iloc[-1]["mae"])
    ci_by_k = {}
    for k, sub in df.groupby("K"):
        per_split_state = sub.groupby(["split_seed", "state_id"])["abs_error"].mean().to_numpy()
        ci_by_k[str(int(k))] = ci95(per_split_state)
    first_k = int(overall_by_k.iloc[0]["K"])
    last_k = int(overall_by_k.iloc[-1]["K"])
    paired = (
        df[df["K"].isin([first_k, last_k])]
        .groupby(["K", "split_seed", "state_id"])["abs_error"]
        .mean()
        .unstack("K")
        .dropna()
    )
    improvement_ci = ci95((paired[first_k] - paired[last_k]).to_numpy()) if not paired.empty else ci95([])
    summary = {
        "experiment": "pilot_to_heldout_prediction",
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "splits": args.splits,
        "K_values": k_values,
        "overall_mae_by_K": overall_by_k.to_dict(orient="records"),
        "mae_reduction_first_to_last": first_mae - last_mae,
        "relative_mae_reduction": (first_mae - last_mae) / first_mae if first_mae > 0 else 0.0,
        "confidence_intervals": {
            "mae_by_K": ci_by_k,
            "mae_reduction_first_to_last": improvement_ci,
        },
        "artifacts": {"table": str(table_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / "exp3_pilot_to_heldout_prediction.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=96)
    parser.add_argument("--rollouts", type=int, default=384)
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mismatch", type=str, default="mild")
    parser.add_argument("--scorer", type=str, default="predicted_utility")
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp3 complete: "
        f"relative MAE reduction={summary['relative_mae_reduction']:.3f}"
    )


if __name__ == "__main__":
    main()
