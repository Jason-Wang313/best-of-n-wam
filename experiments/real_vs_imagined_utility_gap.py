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
    add_backend_args,
    aggregate_curve_table,
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


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    dynamics_backend = getattr(args, "dynamics_backend", "analytic")
    learned_model = load_model_for_backend(args)
    seeds = seed_list_from_args(args)
    mismatches = ["none", "mild", "severe", "stuck_slip"]
    rows = []
    for seed_id, seed in enumerate(seeds):
        for i, mismatch in enumerate(mismatches):
            pools = generate_rollout_pools(
                args.states,
                args.rollouts,
                mismatch,
                seed + 997 * i,
                dynamics_backend=dynamics_backend,
                learned_model=learned_model,
            )
            for pool in pools:
                for row in curve_rows_for_pool(pool, args.scorer, N_VALUES):
                    row.update({"seed": int(seed), "seed_id": int(seed_id)})
                    rows.append(row)
    df = pd.DataFrame(rows)
    suffix = backend_suffix(args)
    table_path = results_dir() / "tables" / f"exp5_real_vs_imagined_utility_gap{suffix}.csv"
    df.to_csv(table_path, index=False)
    agg = aggregate_curve_table(rows, ["mismatch", "N"])
    agg_path = results_dir() / "tables" / f"exp5_real_vs_imagined_utility_gap_aggregate{suffix}.csv"
    agg.to_csv(agg_path, index=False)

    plt.figure(figsize=(7.8, 4.9))
    for mismatch, sub in agg.groupby("mismatch"):
        sub = sub.sort_values("N")
        plt.plot(sub["N"], sub["imagined_utility"], marker="o", label=f"{mismatch} imagined")
        plt.plot(sub["N"], sub["real_utility"], marker="s", linestyle="--", label=f"{mismatch} real")
    plt.xscale("log", base=2)
    plt.xticks(N_VALUES, [str(n) for n in N_VALUES])
    plt.xlabel("N")
    plt.ylabel("selected utility")
    plt.title("WAM mismatch: imagined utility can improve while real utility saturates")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / f"exp5_imagined_vs_real_gap{suffix}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    def gap_summary_rows(table: pd.DataFrame, include_seed: bool) -> list[dict]:
        group_cols = ["seed", "mismatch"] if include_seed else ["mismatch"]
        out = []
        for keys, sub in table.groupby(group_cols):
            sub = sub.sort_values("N")
            n1 = sub[sub["N"] == 1].iloc[0]
            n64 = sub[sub["N"] == 64].iloc[0]
            if include_seed:
                seed, mismatch = keys
            else:
                seed = None
                mismatch = keys[0] if isinstance(keys, tuple) else keys
            row = {
                "mismatch": mismatch,
                "imagined_gain_N64_minus_N1": float(n64["imagined_utility"] - n1["imagined_utility"]),
                "real_gain_N64_minus_N1": float(n64["real_utility"] - n1["real_utility"]),
                "gap_N1": float(n1["gap_imagined_minus_real"]),
                "gap_N64": float(n64["gap_imagined_minus_real"]),
                "gap_growth": float(n64["gap_imagined_minus_real"] - n1["gap_imagined_minus_real"]),
                "normalized_gap_growth": float(
                    n64["normalized_gap_imagined_minus_real"] - n1["normalized_gap_imagined_minus_real"]
                ),
                "real_utility_N64": float(n64["real_utility"]),
                "normalized_real_utility_N64": float(n64["normalized_real_utility"]),
                "success_N64": float(n64["success"]),
            }
            if include_seed:
                row["seed"] = int(seed)
            out.append(row)
        return out

    gap_rows = gap_summary_rows(agg, include_seed=False)
    seed_agg = (
        df.groupby(["seed", "mismatch", "N"], dropna=False)[
            [
                "success",
                "real_utility",
                "imagined_utility",
                "gap_imagined_minus_real",
                "normalized_real_utility",
                "normalized_imagined_utility",
                "normalized_gap_imagined_minus_real",
            ]
        ]
        .mean()
        .reset_index()
    )
    seed_gap_rows = gap_summary_rows(seed_agg, include_seed=True)
    gap_df = pd.DataFrame(gap_rows)
    gap_path = results_dir() / "tables" / f"exp5_gap_summary{suffix}.csv"
    gap_df.to_csv(gap_path, index=False)
    seed_gap_df = pd.DataFrame(seed_gap_rows)

    none_gap = float(gap_df[gap_df["mismatch"] == "none"]["gap_growth"].iloc[0])
    severe_gap = float(gap_df[gap_df["mismatch"] == "severe"]["gap_growth"].iloc[0])
    stuck_gap = float(gap_df[gap_df["mismatch"] == "stuck_slip"]["gap_growth"].iloc[0])
    seed_ci_rows = []
    for seed, sub in seed_gap_df.groupby("seed"):
        none_seed = float(sub[sub["mismatch"] == "none"]["gap_growth"].iloc[0])
        severe_seed = float(sub[sub["mismatch"] == "severe"]["gap_growth"].iloc[0])
        stuck_seed = float(sub[sub["mismatch"] == "stuck_slip"]["gap_growth"].iloc[0])
        seed_ci_rows.append(
            {
                "seed": int(seed),
                "severe_gap_growth_minus_none": severe_seed - none_seed,
                "stuck_slip_gap_growth_minus_none": stuck_seed - none_seed,
            }
        )
    seed_ci_df = pd.DataFrame(seed_ci_rows)
    summary = {
        "experiment": "real_vs_imagined_utility_gap",
        "dynamics_backend": dynamics_backend,
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "scorer": args.scorer,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "gap_summary": gap_df.to_dict(orient="records"),
        "severe_gap_growth_minus_none": severe_gap - none_gap,
        "stuck_slip_gap_growth_minus_none": stuck_gap - none_gap,
        "confidence_intervals": {
            "severe_gap_growth_minus_none": ci95(seed_ci_df["severe_gap_growth_minus_none"].to_numpy()),
            "stuck_slip_gap_growth_minus_none": ci95(seed_ci_df["stuck_slip_gap_growth_minus_none"].to_numpy()),
        },
        "artifacts": {"table": str(table_path), "aggregate": str(agg_path), "gap_summary": str(gap_path), "figure": str(fig_path)},
    }
    write_json(results_dir() / f"exp5_real_vs_imagined_utility_gap{suffix}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=96)
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--scorer", type=str, default="predicted_utility")
    add_backend_args(parser)
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp5 complete: "
        f"severe-none gap growth delta={summary['severe_gap_growth_minus_none']:.3f}, "
        f"stuck-none gap growth delta={summary['stuck_slip_gap_growth_minus_none']:.3f}"
    )


if __name__ == "__main__":
    main()
