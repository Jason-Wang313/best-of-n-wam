from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from wam_inference_value.evaluation import (
    N_VALUES,
    aggregate_curve_table,
    ci95,
    curve_rows_for_pool,
    ensure_result_dirs,
    results_dir,
    seed_list_from_args,
    write_json,
)
from wam_inference_value.learned_wam import load_or_train_learned_wam_lite
from wam_inference_value.rollouts import generate_rollout_pools


BACKENDS = {
    "analytic_nominal_wam": "analytic",
    "learned_wam_lite": "learned",
    "oracle_true_dynamics": "oracle_true",
}


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    seeds = seed_list_from_args(args)
    if len(seeds) < 5:
        raise ValueError("learned_wam_vs_analytic_wam requires at least 5 seeds for confidence intervals")

    model_path = Path(args.model_path) if args.model_path else results_dir() / "models" / "learned_wam_lite.npz"
    learned_model = load_or_train_learned_wam_lite(
        model_path=model_path,
        train_if_missing=args.train_if_missing,
        id_mismatch=args.id_mismatch,
        seed=args.model_seed,
        train_states=args.train_states,
        train_rollouts=args.train_rollouts,
        val_states=args.val_states,
        val_rollouts=args.val_rollouts,
        max_horizon=args.max_horizon,
    )

    rows = []
    for seed_id, seed in enumerate(seeds):
        for backend_label, backend in BACKENDS.items():
            pools = generate_rollout_pools(
                args.states,
                args.rollouts,
                args.mismatch,
                seed,
                dynamics_backend=backend,
                learned_model=learned_model if backend == "learned" else None,
            )
            for pool in pools:
                for row in curve_rows_for_pool(pool, args.scorer, N_VALUES):
                    row.update({"seed": int(seed), "seed_id": int(seed_id), "backend": backend_label})
                    rows.append(row)

    df = pd.DataFrame(rows)
    table_path = results_dir() / "tables" / "learned_wam_vs_analytic_wam.csv"
    df.to_csv(table_path, index=False)

    agg = aggregate_curve_table(rows, ["backend", "N"])
    agg_path = results_dir() / "tables" / "learned_wam_vs_analytic_wam_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    plt.figure(figsize=(7.6, 4.8))
    for backend, sub in agg.groupby("backend"):
        sub = sub.sort_values("N")
        plt.plot(sub["N"], sub["real_utility"], marker="o", label=backend)
    plt.xscale("log", base=2)
    plt.xticks(N_VALUES, [str(n) for n in N_VALUES])
    plt.xlabel("N")
    plt.ylabel("selected real utility")
    plt.title("Learned WAM-lite vs analytic nominal WAM")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "learned_wam_vs_analytic_wam.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    seed_agg = (
        df.groupby(["seed", "backend", "N"], dropna=False)[["success", "real_utility", "normalized_real_utility"]]
        .mean()
        .reset_index()
    )
    backend_seed_rows = []
    delta_rows = []
    for seed, sub in seed_agg.groupby("seed"):
        n64_by_backend = {}
        for backend, backend_sub in sub.groupby("backend"):
            indexed = backend_sub.set_index("N")
            n1 = indexed.loc[1]
            n64 = indexed.loc[64]
            backend_seed_rows.append(
                {
                    "seed": int(seed),
                    "backend": backend,
                    "success_N64": float(n64["success"]),
                    "real_utility_N64": float(n64["real_utility"]),
                    "normalized_real_utility_N64": float(n64["normalized_real_utility"]),
                    "real_utility_gain_N64_minus_N1": float(n64["real_utility"] - n1["real_utility"]),
                }
            )
            n64_by_backend[backend] = float(n64["real_utility"])
        delta_rows.append(
            {
                "seed": int(seed),
                "learned_minus_analytic_real_utility_N64": n64_by_backend["learned_wam_lite"]
                - n64_by_backend["analytic_nominal_wam"],
                "oracle_minus_learned_real_utility_N64": n64_by_backend["oracle_true_dynamics"]
                - n64_by_backend["learned_wam_lite"],
                "oracle_minus_analytic_real_utility_N64": n64_by_backend["oracle_true_dynamics"]
                - n64_by_backend["analytic_nominal_wam"],
            }
        )

    backend_seed_df = pd.DataFrame(backend_seed_rows)
    delta_df = pd.DataFrame(delta_rows)
    backend_ci = {}
    for backend, sub in backend_seed_df.groupby("backend"):
        backend_ci[backend] = {
            "success_N64": ci95(sub["success_N64"].to_numpy()),
            "real_utility_N64": ci95(sub["real_utility_N64"].to_numpy()),
            "normalized_real_utility_N64": ci95(sub["normalized_real_utility_N64"].to_numpy()),
            "real_utility_gain_N64_minus_N1": ci95(sub["real_utility_gain_N64_minus_N1"].to_numpy()),
        }

    summary = {
        "experiment": "learned_wam_vs_analytic_wam",
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "mismatch": args.mismatch,
        "scorer": args.scorer,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "model_path": str(model_path),
        "backends": list(BACKENDS.keys()),
        "aggregate": agg.to_dict(orient="records"),
        "confidence_intervals": {
            "by_backend": backend_ci,
            "deltas": {
                "learned_minus_analytic_real_utility_N64": ci95(
                    delta_df["learned_minus_analytic_real_utility_N64"].to_numpy()
                ),
                "oracle_minus_learned_real_utility_N64": ci95(delta_df["oracle_minus_learned_real_utility_N64"].to_numpy()),
                "oracle_minus_analytic_real_utility_N64": ci95(
                    delta_df["oracle_minus_analytic_real_utility_N64"].to_numpy()
                ),
            },
        },
        "artifacts": {"table": str(table_path), "aggregate": str(agg_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / "learned_wam_vs_analytic_wam.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=72)
    parser.add_argument("--rollouts", type=int, default=192)
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--mismatch", type=str, default="mild")
    parser.add_argument("--scorer", type=str, default="predicted_utility")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--model-seed", type=int, default=101)
    parser.add_argument("--id-mismatch", type=str, default="mild")
    parser.add_argument("--train-states", type=int, default=64)
    parser.add_argument("--train-rollouts", type=int, default=96)
    parser.add_argument("--val-states", type=int, default=24)
    parser.add_argument("--val-rollouts", type=int, default=96)
    parser.add_argument("--max-horizon", type=int, default=12)
    args = parser.parse_args()
    summary = run(args)
    delta = summary["confidence_intervals"]["deltas"]["learned_minus_analytic_real_utility_N64"]
    print(
        "learned-vs-analytic complete: "
        f"learned-analytic N64 real utility delta={delta['mean']:.3f} +/- {delta['ci95']:.3f}"
    )


if __name__ == "__main__":
    main()
