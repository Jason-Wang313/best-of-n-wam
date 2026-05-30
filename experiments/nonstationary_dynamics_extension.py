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

from wam_inference_value.envs import BlockPush2D
from wam_inference_value.evaluation import N_VALUES, ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.rollouts import make_rollout_pool
from wam_inference_value.scorers import scores_for_pool
from wam_inference_value.theorem import binary_best_of_n_finite, simulate_best_of_n


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    env = BlockPush2D()
    rows = []
    estimation_rows = []
    shift_rows = []
    n_select = 16
    pilot_k = max(4, min(int(getattr(args, "pilot_k", 48)), int(args.rollouts) // 2))
    for ep in range(args.episodes):
        state = env.sample_state(args.seed + 101 * ep, mismatch="nonstationary", state_id=ep)
        stale_pred = None
        for t in range(env.config.episode_horizon):
            pool = make_rollout_pool(
                env,
                state,
                state_id=t,
                mismatch="nonstationary",
                n_rollouts=args.rollouts,
                seed=args.seed + 10_003 * ep + 97 * t,
                horizon=env.config.horizon,
            )
            scores = scores_for_pool(pool, "predicted_utility")
            success = pool.real_success
            exact = binary_best_of_n_finite(scores, success, N_VALUES)
            mc = simulate_best_of_n(scores, success, n_select, args.mc_trials, args.seed + ep * 17 + t)
            phase = "pre_shift" if t < env.config.episode_horizon // 2 else "post_shift"
            rows.append(
                {
                    "episode": ep,
                    "t": t,
                    "phase": phase,
                    "p": float(np.mean(success)),
                    "exact_success_N16": exact[n_select],
                    "mc_success_N16": mc,
                    "abs_error_N16": abs(exact[n_select] - mc),
                }
            )
            rng = np.random.default_rng(args.seed + 99_991 * ep + 4_099 * t)
            perm = rng.permutation(len(scores))
            pilot_idx = perm[:pilot_k]
            heldout_idx = perm[pilot_k:]
            current_pred = binary_best_of_n_finite(scores[pilot_idx], success[pilot_idx], [n_select])[n_select]
            actual = binary_best_of_n_finite(scores[heldout_idx], success[heldout_idx], [n_select])[n_select]
            if stale_pred is None:
                stale_pred = current_pred
            stale_error = abs(stale_pred - actual)
            adaptive_error = abs(current_pred - actual)
            estimation_rows.append(
                {
                    "episode": ep,
                    "t": t,
                    "phase": phase,
                    "pilot_k": pilot_k,
                    "stale_pred_success_N16": stale_pred,
                    "adaptive_pred_success_N16": current_pred,
                    "heldout_actual_success_N16": actual,
                    "stale_abs_error_N16": stale_error,
                    "adaptive_abs_error_N16": adaptive_error,
                    "stale_minus_adaptive_abs_error_N16": stale_error - adaptive_error,
                }
            )
            best = int(np.argmax(scores))
            state = env.step(state, pool.records[best].actions[0], state.true_params, use_nonstationary_shift=True)
        shift_rows.append({"episode": ep, "final_distance": env.distance_to_target(state)})

    df = pd.DataFrame(rows)
    table_path = results_dir() / "tables" / "exp8_nonstationary_conditional_law.csv"
    df.to_csv(table_path, index=False)
    by_phase = df.groupby("phase").agg(
        p=("p", "mean"),
        exact_success_N16=("exact_success_N16", "mean"),
        mc_success_N16=("mc_success_N16", "mean"),
        abs_error_N16=("abs_error_N16", "mean"),
    ).reset_index()
    phase_path = results_dir() / "tables" / "exp8_nonstationary_phase_summary.csv"
    by_phase.to_csv(phase_path, index=False)
    est = pd.DataFrame(estimation_rows)
    est_path = results_dir() / "tables" / "exp8_nonstationary_stale_vs_adaptive.csv"
    est.to_csv(est_path, index=False)
    est_by_phase = est.groupby("phase").agg(
        stale_abs_error_N16=("stale_abs_error_N16", "mean"),
        adaptive_abs_error_N16=("adaptive_abs_error_N16", "mean"),
        stale_minus_adaptive_abs_error_N16=("stale_minus_adaptive_abs_error_N16", "mean"),
    ).reset_index()
    est_phase_path = results_dir() / "tables" / "exp8_nonstationary_stale_vs_adaptive_phase_summary.csv"
    est_by_phase.to_csv(est_phase_path, index=False)

    plt.figure(figsize=(6.8, 4.4))
    for col in ["p", "exact_success_N16"]:
        sub = df.groupby("t")[col].mean().reset_index()
        plt.plot(sub["t"], sub[col], marker="o", label=col)
    plt.axvline(env.config.episode_horizon // 2 - 0.5, color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("closed-loop time step")
    plt.ylabel("conditional rollout-pool statistic")
    plt.title("Nonstationary dynamics shift the conditional rollout distribution")
    plt.legend()
    plt.tight_layout()
    fig_path = results_dir() / "figures" / "exp8_nonstationary_shift.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    pre = by_phase[by_phase["phase"] == "pre_shift"]
    post = by_phase[by_phase["phase"] == "post_shift"]
    p_shift = float(post["p"].iloc[0] - pre["p"].iloc[0]) if len(pre) and len(post) else 0.0
    est_pre = est[est["phase"] == "pre_shift"].groupby("episode")["stale_abs_error_N16"].mean()
    est_post = est[est["phase"] == "post_shift"].groupby("episode")["stale_abs_error_N16"].mean()
    est_adaptive_post = est[est["phase"] == "post_shift"].groupby("episode")["adaptive_abs_error_N16"].mean()
    common_shift = sorted(set(est_pre.index) & set(est_post.index))
    common_adapt = sorted(set(est_post.index) & set(est_adaptive_post.index))
    stale_post_minus_pre = (
        (est_post.loc[common_shift] - est_pre.loc[common_shift]).to_numpy()
        if common_shift
        else np.asarray([], dtype=float)
    )
    stale_minus_adaptive_post = (
        (est_post.loc[common_adapt] - est_adaptive_post.loc[common_adapt]).to_numpy()
        if common_adapt
        else np.asarray([], dtype=float)
    )
    summary = {
        "experiment": "nonstationary_dynamics_extension",
        "episodes": args.episodes,
        "rollouts_per_state": args.rollouts,
        "pilot_k": pilot_k,
        "mean_abs_error_N16": float(df["abs_error_N16"].mean()),
        "pre_to_post_mean_p_shift": p_shift,
        "stale_post_minus_pre_abs_error_N16": float(np.mean(stale_post_minus_pre)) if stale_post_minus_pre.size else 0.0,
        "stale_minus_adaptive_post_abs_error_N16": float(np.mean(stale_minus_adaptive_post)) if stale_minus_adaptive_post.size else 0.0,
        "confidence_intervals": {
            "stale_post_minus_pre_abs_error_N16": ci95(stale_post_minus_pre),
            "stale_minus_adaptive_post_abs_error_N16": ci95(stale_minus_adaptive_post),
        },
        "phase_summary": by_phase.to_dict(orient="records"),
        "stale_vs_adaptive_phase_summary": est_by_phase.to_dict(orient="records"),
        "artifacts": {
            "table": str(table_path),
            "phase_summary": str(phase_path),
            "stale_vs_adaptive": str(est_path),
            "stale_vs_adaptive_phase_summary": str(est_phase_path),
            "figure": str(fig_path),
        },
    }
    write_json(results_dir() / "exp8_nonstationary_dynamics_extension.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--rollouts", type=int, default=160)
    parser.add_argument("--mc-trials", type=int, default=4000)
    parser.add_argument("--pilot-k", type=int, default=48)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    summary = run(args)
    print(
        "exp8 complete: "
        f"conditional law MAE={summary['mean_abs_error_N16']:.4f}, "
        f"p shift={summary['pre_to_post_mean_p_shift']:.3f}"
    )


if __name__ == "__main__":
    main()
