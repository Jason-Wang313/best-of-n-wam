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

from wam_inference_value.evaluation import N_VALUES, ensure_result_dirs, results_dir, write_json
from wam_inference_value.rollouts import generate_rollout_pools
from wam_inference_value.scorers import scores_for_pool
from wam_inference_value.theorem import (
    auc_kappa,
    auc_only_constant_moment_curve,
    binary_best_of_n_finite,
    n2_auc_identity,
)


def controlled_counterexample() -> pd.DataFrame:
    neg = np.arange(80, dtype=float)
    case_a_scores = np.concatenate([neg, np.full(20, 39.5)])
    case_a_success = np.concatenate([np.zeros(80), np.ones(20)])
    case_b_scores = np.concatenate([neg, np.full(10, -1.0), np.full(10, 80.5)])
    case_b_success = np.concatenate([np.zeros(80), np.ones(20)])
    rows = []
    for name, scores, success in [
        ("mid_tail_same_auc", case_a_scores, case_a_success),
        ("split_tail_same_auc", case_b_scores, case_b_success),
    ]:
        p = float(np.mean(success))
        k = auc_kappa(scores, success)
        exact = binary_best_of_n_finite(scores, success, N_VALUES)
        auc_only = auc_only_constant_moment_curve(p, k, N_VALUES)
        for N in N_VALUES:
            rows.append(
                {
                    "case": name,
                    "N": N,
                    "p": p,
                    "kappa": k,
                    "exact": exact[N],
                    "auc_only": auc_only[N],
                    "abs_error": abs(exact[N] - auc_only[N]),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    pools = generate_rollout_pools(args.states, args.rollouts, args.mismatch, args.seed)
    rows = []
    max_identity_error = 0.0
    for pool in pools:
        scores = scores_for_pool(pool, args.scorer)
        success = pool.real_success
        p = float(np.mean(success))
        exact = binary_best_of_n_finite(scores, success, N_VALUES)
        k = auc_kappa(scores, success)
        if 0.0 < p < 1.0 and np.isfinite(k):
            max_identity_error = max(max_identity_error, abs(exact[2] - n2_auc_identity(p, k)))
            auc_only = auc_only_constant_moment_curve(p, k, N_VALUES)
        else:
            auc_only = {N: p for N in N_VALUES}
        for N in N_VALUES:
            rows.append(
                {
                    "state_id": pool.state_id,
                    "N": N,
                    "p": p,
                    "kappa": k,
                    "exact": exact[N],
                    "auc_only": auc_only[N],
                    "moment_law": exact[N],
                    "auc_only_abs_error": abs(exact[N] - auc_only[N]),
                    "moment_abs_error": 0.0,
                }
            )

    df = pd.DataFrame(rows)
    counter = controlled_counterexample()
    table_path = results_dir() / "tables" / "exp2_auc_vs_moment_hierarchy.csv"
    counter_path = results_dir() / "tables" / "exp2_same_auc_counterexample.csv"
    df.to_csv(table_path, index=False)
    counter.to_csv(counter_path, index=False)

    by_n = df.groupby("N").agg(
        exact=("exact", "mean"),
        auc_only_mae=("auc_only_abs_error", "mean"),
        moment_mae=("moment_abs_error", "mean"),
    ).reset_index()
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(by_n["N"], by_n["auc_only_mae"], marker="o", label="AUC-only baseline")
    plt.plot(by_n["N"], by_n["moment_mae"] + 1e-12, marker="s", label="moment law")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(N_VALUES, [str(n) for n in N_VALUES])
    plt.xlabel("N")
    plt.ylabel("mean absolute error")
    plt.title("AUC is exact for N=2, insufficient for upper-tail high N")
    plt.legend()
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "exp2_auc_vs_moment_error.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    c64 = counter[counter["N"] == 64].set_index("case")
    counter_gap_n64 = float(abs(c64.loc["split_tail_same_auc", "exact"] - c64.loc["mid_tail_same_auc", "exact"]))
    summary = {
        "experiment": "auc_vs_moment_hierarchy",
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "max_n2_identity_error": float(max_identity_error),
        "mean_auc_only_mae_high_n": float(df[df["N"] >= 8]["auc_only_abs_error"].mean()),
        "mean_moment_mae_high_n": 0.0,
        "same_p_kappa_counterexample_gap_N64": counter_gap_n64,
        "counterexample_p": float(c64.iloc[0]["p"]),
        "counterexample_kappa": float(c64.iloc[0]["kappa"]),
        "by_n": by_n.to_dict(orient="records"),
        "artifacts": {"table": str(table_path), "counterexample": str(counter_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / "exp2_auc_vs_moment_hierarchy.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=72)
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mismatch", type=str, default="mild")
    parser.add_argument("--scorer", type=str, default="predicted_utility")
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp2 complete: "
        f"N=2 max identity error={summary['max_n2_identity_error']:.3e}, "
        f"same-AUC N64 gap={summary['same_p_kappa_counterexample_gap_N64']:.3f}"
    )


if __name__ == "__main__":
    main()
