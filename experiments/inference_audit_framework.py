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

from wam_inference_value.audit import audit_score_distribution
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


AUDIT_SCORERS = [
    "predicted_utility",
    "uncertainty_penalized",
    "safety_penalized",
    "random_score",
    "oracle_real_utility",
    "anti_real_utility",
]


def _scores(pool, scorer: str) -> np.ndarray:
    if scorer == "anti_real_utility":
        return -pool.real_utility
    return scores_for_pool(pool, scorer)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return 0.0
    xx = x[mask] - float(np.mean(x[mask]))
    yy = y[mask] - float(np.mean(y[mask]))
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return 0.0 if denom <= 1e-12 else float(np.sum(xx * yy) / denom)


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    dynamics_backend = getattr(args, "dynamics_backend", "analytic")
    learned_model = load_model_for_backend(args)
    seeds = seed_list_from_args(args)
    mismatches = args.mismatches or ["mild", "severe", "stuck_slip", "nonstationary"]
    summary_rows = []
    curve_rows = []
    for seed in seeds:
        for mismatch in mismatches:
            pools = generate_rollout_pools(
                args.states,
                args.rollouts,
                mismatch,
                seed,
                dynamics_backend=dynamics_backend,
                learned_model=learned_model,
            )
            for pool in pools:
                real_utility = pool.real_utility
                imagined_utility = pool.imagined_utility
                for scorer in AUDIT_SCORERS:
                    scores = _scores(pool, scorer)
                    audit = audit_score_distribution(
                        scores,
                        real_utility,
                        N_VALUES,
                        imagined_utilities=imagined_utility,
                        top_fraction=args.top_fraction,
                        compute_cost_per_rollout=args.compute_cost,
                    )
                    profile = audit["profile"]
                    alignment = audit["alignment"]
                    decision = audit["decision"]
                    summary_rows.append(
                        {
                            "seed": int(seed),
                            "state_id": int(pool.state_id),
                            "mismatch": mismatch,
                            "dynamics_backend": dynamics_backend,
                            "scorer": scorer,
                            "profile_class": audit["profile_class"],
                            "decision_action": decision["action"],
                            "decision_reason": decision["reason"],
                            "recommended_n": int(decision["recommended_n"]),
                            "stop_n": int(audit["stop_n"]),
                            "value_N1": float(profile["curve"][1]),
                            "value_N64": float(profile["curve"][64]),
                            "gain_N64_minus_N1": float(profile["gain_last_minus_first"]),
                            "best_minus_N1": float(profile["best_minus_first"]),
                            "tail_real_uplift": float(alignment["tail_real_uplift"]),
                            "tail_imagined_uplift": float(alignment.get("tail_imagined_uplift", np.nan)),
                            "tail_hallucination_gap": float(alignment.get("tail_hallucination_gap", np.nan)),
                            "score_real_rank_corr": float(alignment["score_real_rank_corr"]),
                            "real_imagined_corr": float(alignment.get("real_imagined_corr", np.nan)),
                            "alignment_status": alignment["alignment_status"],
                        }
                    )
                    for row in profile["rows"]:
                        curve_rows.append(
                            {
                                "seed": int(seed),
                                "state_id": int(pool.state_id),
                                "mismatch": mismatch,
                                "dynamics_backend": dynamics_backend,
                                "scorer": scorer,
                                "N": int(row["N"]),
                                "normalized_real_utility": float(row["value"]),
                                "delta_value": float(row["delta_value"]),
                                "delta_per_rollout": float(row["delta_per_rollout"]),
                            }
                        )

    df = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_rows)
    suffix = backend_suffix(args)
    table_path = results_dir() / "tables" / f"inference_audit_framework{suffix}.csv"
    curve_path = results_dir() / "tables" / f"inference_audit_curves{suffix}.csv"
    df.to_csv(table_path, index=False)
    curves.to_csv(curve_path, index=False)

    seed_metrics = []
    for seed, sub in df.groupby("seed"):
        anti = sub[sub["scorer"] == "anti_real_utility"]
        aligned = sub[sub["alignment_status"] == "aligned"]
        harmful = sub[sub["profile_class"] == "harmful"]
        stop_early = sub[sub["decision_action"] == "stop_early"]
        seed_metrics.append(
            {
                "seed": int(seed),
                "tail_alignment_gain_corr": _corr(sub["tail_real_uplift"].to_numpy(), sub["gain_N64_minus_N1"].to_numpy()),
                "anti_block_high_n_rate": float(np.mean(anti["decision_action"] == "block_high_n")) if len(anti) else np.nan,
                "anti_harm_magnitude": float(np.mean(-(anti["gain_N64_minus_N1"].to_numpy()))) if len(anti) else np.nan,
                "harmful_profile_block_rate": float(np.mean(harmful["decision_action"] == "block_high_n")) if len(harmful) else np.nan,
                "aligned_profile_gain": float(np.mean(aligned["gain_N64_minus_N1"])) if len(aligned) else np.nan,
                "stop_rule_saved_rollout_fraction": float(np.mean((64 - stop_early["stop_n"]) / 63.0)) if len(stop_early) else 0.0,
                "oracle_gain": float(np.mean(sub[sub["scorer"] == "oracle_real_utility"]["gain_N64_minus_N1"])),
                "predicted_utility_gain": float(np.mean(sub[sub["scorer"] == "predicted_utility"]["gain_N64_minus_N1"])),
            }
        )
    seed_df = pd.DataFrame(seed_metrics)
    seed_metric_path = results_dir() / "tables" / f"inference_audit_seed_metrics{suffix}.csv"
    seed_df.to_csv(seed_metric_path, index=False)

    class_counts = df.groupby(["scorer", "profile_class"], dropna=False).size().reset_index(name="count")
    decision_counts = df.groupby(["scorer", "decision_action"], dropna=False).size().reset_index(name="count")
    class_counts_path = results_dir() / "tables" / f"inference_audit_profile_counts{suffix}.csv"
    decision_counts_path = results_dir() / "tables" / f"inference_audit_decision_counts{suffix}.csv"
    class_counts.to_csv(class_counts_path, index=False)
    decision_counts.to_csv(decision_counts_path, index=False)

    plt.figure(figsize=(7.5, 5.0))
    colors = {
        "helpful": "#287c5b",
        "saturating": "#666666",
        "harmful": "#b23a48",
        "unstable": "#c78100",
    }
    for cls, sub in df.groupby("profile_class"):
        plt.scatter(
            sub["tail_real_uplift"],
            sub["gain_N64_minus_N1"],
            s=18,
            alpha=0.65,
            label=cls,
            color=colors.get(cls, None),
        )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("high-score tail real-utility uplift")
    plt.ylabel("normalized gain: N64 - N1")
    plt.title("Inference audit: tail alignment predicts imagination value")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_path = results_dir() / "figures" / f"inference_audit_tail_alignment{suffix}.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    summary = {
        "experiment": "inference_audit_framework",
        "dynamics_backend": dynamics_backend,
        "states": args.states,
        "rollouts_per_state": args.rollouts,
        "mismatches": mismatches,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "scorers": AUDIT_SCORERS,
        "profile_counts": class_counts.to_dict(orient="records"),
        "decision_counts": decision_counts.to_dict(orient="records"),
        "tail_alignment_gain_corr": _corr(df["tail_real_uplift"].to_numpy(), df["gain_N64_minus_N1"].to_numpy()),
        "anti_block_high_n_rate": float(np.mean(df[df["scorer"] == "anti_real_utility"]["decision_action"] == "block_high_n")),
        "harmful_profile_block_rate": float(np.mean(df[df["profile_class"] == "harmful"]["decision_action"] == "block_high_n")),
        "median_recommended_n": float(np.median(df["recommended_n"])),
        "confidence_intervals": {
            "tail_alignment_gain_corr": ci95(seed_df["tail_alignment_gain_corr"].to_numpy()),
            "anti_block_high_n_rate": ci95(seed_df["anti_block_high_n_rate"].to_numpy()),
            "anti_harm_magnitude": ci95(seed_df["anti_harm_magnitude"].to_numpy()),
            "harmful_profile_block_rate": ci95(seed_df["harmful_profile_block_rate"].to_numpy()),
            "aligned_profile_gain": ci95(seed_df["aligned_profile_gain"].to_numpy()),
            "stop_rule_saved_rollout_fraction": ci95(seed_df["stop_rule_saved_rollout_fraction"].to_numpy()),
            "oracle_gain": ci95(seed_df["oracle_gain"].to_numpy()),
            "predicted_utility_gain": ci95(seed_df["predicted_utility_gain"].to_numpy()),
        },
        "artifacts": {
            "summary_table": str(table_path),
            "curve_table": str(curve_path),
            "seed_metrics": str(seed_metric_path),
            "profile_counts": str(class_counts_path),
            "decision_counts": str(decision_counts_path),
            "figure": str(fig_path),
        },
    }
    write_json(results_dir() / f"inference_audit_framework{suffix}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--rollouts", type=int, default=160)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--mismatches", nargs="*", default=["mild", "severe", "stuck_slip", "nonstationary"])
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--compute-cost", type=float, default=0.0015)
    add_backend_args(parser)
    parser.set_defaults(num_seeds=5)
    args = parser.parse_args()
    summary = run(args)
    print(
        "inference audit complete: "
        f"tail-gain corr={summary['tail_alignment_gain_corr']:.3f}, "
        f"anti block rate={summary['anti_block_high_n_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
