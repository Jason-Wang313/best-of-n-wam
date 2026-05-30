from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wam_inference_value.envs.block_push_2d import BlockPush2D
from wam_inference_value.envs.toy_envs import BaseToyEnv, make_toy_env, sample_toy_action_sequences
from wam_inference_value.models import EnsembleWAM, HorizonWAM, MLPDynamicsWAM, WAMDataset
from wam_inference_value.rollouts import sample_action_sequences
from wam_inference_value.stats import bootstrap_ci, claim_status_from_ci, normalized_utility, paired_bootstrap_ci
from wam_inference_value.theorem import binary_best_of_n_finite, simulate_best_of_n, utility_best_of_n_finite
from wam_inference_value.evaluation import N_VALUES, ensure_result_dirs, results_dir, write_json


ENV_NAMES = ["block_push", "drawer_pull", "slippery_grasp", "nonstationary_shift", "deformable_toy"]
BACKBONES = ["horizon_wam", "mlp_dynamics_wam", "ensemble_wam"]


def make_env(name: str):
    if name == "block_push":
        return BlockPush2D()
    return make_toy_env(name)


def state_vector(state: Any) -> np.ndarray:
    if hasattr(state, "obj_xy"):
        return np.concatenate([np.asarray(state.obj_xy, dtype=float), np.asarray(state.target_xy, dtype=float)])
    return np.asarray(state.vector, dtype=float)


def target_vector(state: Any) -> np.ndarray:
    if hasattr(state, "target_xy"):
        return np.asarray(state.target_xy, dtype=float)
    return np.asarray(state.target, dtype=float)


def sample_state(env: Any, env_name: str, seed: int, mismatch: str, state_id: int):
    if env_name == "block_push":
        return env.sample_state(seed, mismatch=mismatch, state_id=state_id)
    return env.sample_state(seed, mismatch=mismatch, state_id=state_id)


def sample_actions(env: Any, env_name: str, state: Any, n_rollouts: int, horizon: int, seed: int) -> np.ndarray:
    if env_name == "block_push":
        return sample_action_sequences(env, state, n_rollouts, horizon, seed)
    return sample_toy_action_sequences(env, state, n_rollouts, horizon, seed)


def true_params(env: Any, env_name: str, state: Any):
    return state.true_params if env_name == "block_push" else state.params


def nominal_params(env: Any, env_name: str):
    return env.nominal_params if env_name == "block_push" else env.imagined_params()


def rollout_metrics(env: Any, env_name: str, state: Any, actions: np.ndarray, params: Any, *, true: bool) -> list:
    return env.rollout_batch_metrics(state, actions, params, use_nonstationary_shift=true)


def feature_matrix(env_name: str, state: Any, actions: np.ndarray, max_horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=float)
    n, horizon, action_dim = actions.shape
    sv = state_vector(state)
    tv = target_vector(state)
    target_pad = np.zeros(2, dtype=float)
    target_pad[: min(2, len(tv))] = tv[: min(2, len(tv))]
    padded = np.zeros((n, max_horizon, action_dim), dtype=float)
    padded[:, :horizon, :] = actions
    flat = padded.reshape(n, max_horizon * action_dim)
    norms = np.linalg.norm(actions, axis=2)
    sums = actions.sum(axis=1)
    return np.column_stack(
        [
            np.repeat(sv[None, :], n, axis=0),
            np.repeat(target_pad[None, :], n, axis=0),
            flat,
            sums,
            norms.mean(axis=1),
            norms.max(axis=1),
            np.sum(actions * actions, axis=(1, 2)),
            np.full(n, horizon / max_horizon),
        ]
    )


def generate_dataset(env_name: str, mismatch: str, n_states: int, rollouts: int, seed: int, split: str) -> WAMDataset:
    env = make_env(env_name)
    max_horizon = int(getattr(env, "horizon", 10))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for state_id in range(int(n_states)):
        state = sample_state(env, env_name, seed + 7919 * state_id, mismatch, state_id)
        actions = sample_actions(env, env_name, state, rollouts, max_horizon, seed + 104729 * (state_id + 1))
        metrics = rollout_metrics(env, env_name, state, actions, true_params(env, env_name, state), true=True)
        final_vectors = []
        for seq in actions:
            final_state = env.rollout(state, seq, true_params(env, env_name, state), use_nonstationary_shift=True)[0]
            final_vectors.append(state_vector(final_state))
        final_vectors = np.asarray(final_vectors, dtype=float)
        x = feature_matrix(env_name, state, actions, max_horizon)
        y = np.column_stack([final_vectors - state_vector(state)[None, :], np.asarray([m.utility for m in metrics])])
        xs.append(x)
        ys.append(y)
    dataset = WAMDataset(
        np.vstack(xs),
        np.vstack(ys),
        {"env": env_name, "mismatch": mismatch, "split": split, "n_states": int(n_states), "rollouts": int(rollouts), "seed": int(seed)},
    )
    return dataset


def generate_all_datasets(states: int, rollouts: int, seed: int) -> dict:
    ensure_result_dirs()
    out = []
    for env_name in ENV_NAMES:
        for split, mismatch, offset in [("train", "mild", 0), ("validation", "mild", 1000), ("ood_severe", "severe", 2000), ("ood_shift", "nonstationary", 3000)]:
            dataset = generate_dataset(env_name, mismatch, states, rollouts, seed + offset, split)
            path = results_dir() / "datasets" / f"maxout_{env_name}_{split}.npz"
            dataset.save(path)
            out.append({"env": env_name, "split": split, "mismatch": mismatch, "samples": int(len(dataset.x)), "path": str(path)})
    summary = {"datasets": out}
    write_json(results_dir() / "maxout_dataset_summary.json", summary)
    pd.DataFrame(out).to_csv(results_dir() / "tables" / "maxout_dataset_summary.csv", index=False)
    return summary


def train_models_for_env(env_name: str, states: int, rollouts: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    train = generate_dataset(env_name, "mild", states, rollouts, seed, "train")
    val = generate_dataset(env_name, "mild", max(3, states // 2), rollouts, seed + 1000, "validation")
    ood = generate_dataset(env_name, "severe", max(3, states // 2), rollouts, seed + 2000, "ood_severe")
    models = {
        "horizon_wam": HorizonWAM().fit(train),
        "mlp_dynamics_wam": MLPDynamicsWAM(seed=seed, epochs=260).fit(train),
        "ensemble_wam": EnsembleWAM(n_models=5, seed=seed).fit(train),
    }
    metric_rows = []
    for name, model in models.items():
        for split, dataset in [("train", train), ("validation", val), ("ood_severe", ood)]:
            rec = model.evaluate(dataset)
            rec.update({"env": env_name, "split": split})
            metric_rows.append(rec)
        model.save(results_dir() / "models" / f"maxout_{env_name}_{name}.npz", {"env": env_name, "seed": seed})
    return models, {"model_metrics": metric_rows}


def train_all_backbones(states: int, rollouts: int, seed: int) -> dict:
    ensure_result_dirs()
    rows = []
    for i, env_name in enumerate(ENV_NAMES):
        _, summary = train_models_for_env(env_name, states, rollouts, seed + 503 * i)
        rows.extend(summary["model_metrics"])
    pd.DataFrame(rows).to_csv(results_dir() / "tables" / "maxout_model_metrics.csv", index=False)
    summary = {"model_metrics": rows, "backbones": BACKBONES, "envs": ENV_NAMES}
    write_json(results_dir() / "maxout_model_metrics.json", summary)
    return summary


def pool_for_env(env_name: str, seed: int, state_id: int, mismatch: str, n_rollouts: int):
    env = make_env(env_name)
    horizon = int(getattr(env, "horizon", 10))
    state = sample_state(env, env_name, seed + 7919 * state_id, mismatch, state_id)
    actions = sample_actions(env, env_name, state, n_rollouts, horizon, seed + 104729 * (state_id + 1))
    real = rollout_metrics(env, env_name, state, actions, true_params(env, env_name, state), true=True)
    imagined = rollout_metrics(env, env_name, state, actions, nominal_params(env, env_name), true=False)
    x = feature_matrix(env_name, state, actions, horizon)
    return env, state, actions, real, imagined, x


def scores_from_metrics(metrics: list, scorer: str, rng: np.random.Generator) -> np.ndarray:
    if scorer == "random":
        return rng.normal(size=len(metrics))
    if scorer == "predicted_utility":
        return np.asarray([m.utility for m in metrics], dtype=float)
    if scorer == "predicted_goal_distance":
        return np.asarray([-m.final_distance - 0.01 * m.energy for m in metrics], dtype=float)
    if scorer == "predicted_success":
        return np.asarray([2.0 * float(m.success) - m.final_distance - 0.01 * m.energy for m in metrics], dtype=float)
    if scorer == "safety_penalized":
        return np.asarray([m.utility - 0.8 * m.safety_violation for m in metrics], dtype=float)
    if scorer == "anti_real_utility":
        return np.asarray([-m.utility for m in metrics], dtype=float)
    raise ValueError(f"unknown scorer: {scorer}")


def curve_record(env_name: str, backend: str, scorer: str, seed: int, mismatch: str, N: int, scores: np.ndarray, real: list, imagined_utility: np.ndarray) -> dict:
    success = np.asarray([float(m.success) for m in real])
    real_utility = np.asarray([m.utility for m in real])
    s_curve = binary_best_of_n_finite(scores, success, [N])[N]
    u_curve = utility_best_of_n_finite(scores, real_utility, [N])[N]
    iu_curve = utility_best_of_n_finite(scores, imagined_utility, [N])[N]
    return {
        "env": env_name,
        "backend": backend,
        "scorer": scorer,
        "seed": int(seed),
        "mismatch": mismatch,
        "N": int(N),
        "success": s_curve,
        "real_utility": u_curve,
        "imagined_utility": iu_curve,
        "gap_imagined_minus_real": iu_curve - u_curve,
        "normalized_real_utility": utility_best_of_n_finite(scores, normalized_utility(real_utility), [N])[N],
    }


def run_multi_env_suite(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    all_rows = []
    model_rows = []
    falsification_rows = []
    for env_i, env_name in enumerate(ENV_NAMES):
        models, model_summary = train_models_for_env(env_name, args.train_states, args.train_rollouts, args.seed + 1000 * env_i)
        model_rows.extend(model_summary["model_metrics"])
        for seed in args.seeds:
            for mismatch in ["mild", "severe"]:
                env, state, actions, real, imagined, x = pool_for_env(env_name, seed + 17 * env_i, seed % 1000, mismatch, args.rollouts)
                real_utility = np.asarray([m.utility for m in real])
                imagined_utility_nominal = np.asarray([m.utility for m in imagined])
                backends = {"analytic_nominal": imagined_utility_nominal, "oracle_true": real_utility}
                for model_name, model in models.items():
                    pred = model.predict(x)
                    backends[model_name] = pred[:, -1]
                if "ensemble_wam" in models:
                    uncertainty = models["ensemble_wam"].uncertainty(x)[:, -1]
                else:
                    uncertainty = np.zeros(len(real_utility))
                for backend, imagined_utility in backends.items():
                    rng = np.random.default_rng(seed + len(backend))
                    scorer_values = {
                        "random": rng.normal(size=len(real)),
                        "predicted_utility": imagined_utility,
                        "predicted_goal_distance": scores_from_metrics(imagined, "predicted_goal_distance", rng),
                        "safety_penalized": imagined_utility - 0.8 * np.asarray([m.safety_violation for m in imagined]),
                        "uncertainty_penalized": imagined_utility - 0.25 * uncertainty,
                        "oracle_real_utility": real_utility,
                    }
                    for scorer, scores in scorer_values.items():
                        for N in N_VALUES:
                            all_rows.append(curve_record(env_name, backend, scorer, seed, mismatch, N, scores, real, imagined_utility))
                    bad_scores = -real_utility
                    for N in [1, 8, 64]:
                        falsification_rows.append(curve_record(env_name, backend, "anti_real_utility", seed, mismatch, N, bad_scores, real, imagined_utility))
                randomized_scores = np.random.default_rng(seed + 91_003 + env_i).permutation(imagined_utility_nominal)
                for N in [1, 8, 64]:
                    falsification_rows.append(
                        curve_record(
                            env_name,
                            "randomized_dynamics_wam",
                            "predicted_utility",
                            seed,
                            mismatch,
                            N,
                            randomized_scores,
                            real,
                            randomized_scores,
                        )
                    )

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir() / "tables" / "multi_env_curves.csv", index=False)
    pd.DataFrame(model_rows).to_csv(results_dir() / "tables" / "maxout_model_metrics.csv", index=False)
    falsification = pd.DataFrame(falsification_rows)
    falsification.to_csv(results_dir() / "tables" / "exp10_falsification_bad_scorer.csv", index=False)

    agg = df.groupby(["env", "backend", "scorer", "mismatch", "N"], dropna=False)[
        ["success", "real_utility", "imagined_utility", "gap_imagined_minus_real", "normalized_real_utility"]
    ].mean().reset_index()
    agg.to_csv(results_dir() / "tables" / "multi_env_curves_aggregate.csv", index=False)

    # Claim-oriented summaries.
    exp_rows = []
    for env_name in ENV_NAMES:
        sub = df[(df["env"] == env_name) & (df["mismatch"] == "mild")]
        for backend in ["analytic_nominal", "horizon_wam", "mlp_dynamics_wam", "ensemble_wam", "oracle_true"]:
            b = sub[sub["backend"] == backend]
            if b.empty:
                continue
            n1 = b[(b["scorer"] == "predicted_utility") & (b["N"] == 1)].groupby("seed")["real_utility"].mean()
            n64 = b[(b["scorer"] == "predicted_utility") & (b["N"] == 64)].groupby("seed")["real_utility"].mean()
            random64 = b[(b["scorer"] == "random") & (b["N"] == 64)].groupby("seed")["real_utility"].mean()
            common = sorted(set(n64.index) & set(n1.index) & set(random64.index))
            if common:
                exp_rows.append(
                    {
                        "env": env_name,
                        "backend": backend,
                        "n64_minus_n1": paired_bootstrap_ci(n64.loc[common], n1.loc[common], seed=11),
                        "n64_minus_random": paired_bootstrap_ci(n64.loc[common], random64.loc[common], seed=12),
                    }
                )
    severe_gap = []
    for env_name in ENV_NAMES:
        for backend in ["analytic_nominal", "horizon_wam", "mlp_dynamics_wam", "ensemble_wam"]:
            base = df[(df["env"] == env_name) & (df["backend"] == backend) & (df["scorer"] == "predicted_utility")]
            mild = base[(base["mismatch"] == "mild") & (base["N"] == 64)].groupby("seed")["gap_imagined_minus_real"].mean()
            severe = base[(base["mismatch"] == "severe") & (base["N"] == 64)].groupby("seed")["gap_imagined_minus_real"].mean()
            common = sorted(set(mild.index) & set(severe.index))
            if common:
                severe_gap.append({"env": env_name, "backend": backend, "severe_minus_mild_gap": paired_bootstrap_ci(severe.loc[common], mild.loc[common], seed=13)})

    plt.figure(figsize=(8, 5))
    plot_sub = agg[(agg["scorer"] == "predicted_utility") & (agg["mismatch"] == "mild")]
    for (env_name, backend), sub in plot_sub.groupby(["env", "backend"]):
        if backend in {"horizon_wam", "oracle_true", "analytic_nominal"}:
            plt.plot(sub["N"], sub["real_utility"], marker="o", label=f"{env_name}/{backend}")
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("real utility")
    plt.title("Multi-env inference curves")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(results_dir() / "figures" / "multi_env_inference_curves.png", dpi=160)
    plt.close()

    fals_agg = falsification.groupby(["env", "backend", "N"], dropna=False)["real_utility"].mean().reset_index()
    plt.figure(figsize=(7, 4.5))
    for env_name, sub in fals_agg[fals_agg["backend"] == "oracle_true"].groupby("env"):
        plt.plot(sub["N"], sub["real_utility"], marker="o", label=env_name)
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("real utility selected by anti-scorer")
    plt.title("Falsification: high-N amplifies a bad scorer")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(results_dir() / "figures" / "exp10_falsification_bad_scorer.png", dpi=160)
    plt.close()

    randomized64 = falsification[
        (falsification["backend"] == "randomized_dynamics_wam")
        & (falsification["mismatch"] == "mild")
        & (falsification["N"] == 64)
    ]["real_utility"].mean()
    oracle64 = df[
        (df["backend"] == "oracle_true")
        & (df["scorer"] == "predicted_utility")
        & (df["mismatch"] == "mild")
        & (df["N"] == 64)
    ]["real_utility"].mean()
    summary = {
        "experiment": "multi_env_suite",
        "envs": ENV_NAMES,
        "backbones": BACKBONES,
        "seeds": args.seeds,
        "n_states": args.states,
        "n_rollouts": args.rollouts,
        "N_values": N_VALUES,
        "model_metrics": model_rows,
        "inference_claims": exp_rows,
        "mismatch_gap_claims": severe_gap,
        "falsification": {
            "anti_scorer_mean_N64": float(falsification[falsification["N"] == 64]["real_utility"].mean()),
            "anti_scorer_mean_N1": float(falsification[falsification["N"] == 1]["real_utility"].mean()),
            "randomized_dynamics_mean_N64": float(randomized64),
            "oracle_true_mean_N64": float(oracle64),
            "randomized_dynamics_oracle_gap_N64": float(oracle64 - randomized64),
        },
        "artifacts": {
            "curves": str(results_dir() / "tables" / "multi_env_curves.csv"),
            "aggregate": str(results_dir() / "tables" / "multi_env_curves_aggregate.csv"),
            "falsification": str(results_dir() / "tables" / "exp10_falsification_bad_scorer.csv"),
            "figure": str(results_dir() / "figures" / "multi_env_inference_curves.png"),
        },
    }
    write_json(results_dir() / "multi_env_suite.json", summary)
    write_json(
        results_dir() / "exp10_falsification_bad_scorer.json",
        summary["falsification"]
        | {"artifacts": {"table": str(results_dir() / "tables" / "exp10_falsification_bad_scorer.csv"), "figure": str(results_dir() / "figures" / "exp10_falsification_bad_scorer.png")}},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=80)
    parser.add_argument("--train-states", type=int, default=8)
    parser.add_argument("--train-rollouts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[1001, 1002, 1003, 1004, 1005])
    args = parser.parse_args()
    summary = run_multi_env_suite(args)
    print(
        "multi-env complete: "
        f"envs={len(summary['envs'])}, backbones={len(summary['backbones'])}, seeds={len(summary['seeds'])}"
    )


if __name__ == "__main__":
    main()
