from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wam_inference_value.benchmarks.gym_manip_adapter import GymManipAdapter
from wam_inference_value.benchmarks.gym_manip_rollouts import sample_rollout_pool
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.models.simple_wam import RidgeRegressor
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32]


def fit_visual_model(x: np.ndarray, y: np.ndarray, *, ridge: float, seed: int):
    """Fit the strongest available CPU visual WAM-lite regressor."""

    try:
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=140,
            max_features=0.35,
            min_samples_leaf=2,
            random_state=int(seed),
            n_jobs=-1,
        )
        model.fit(x, y)
        return model, "extra_trees_visual_wam"
    except Exception:
        model = RidgeRegressor(ridge=ridge).fit(x, y)
        return model, "ridge_visual_wam"


def predict_visual_model(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=float)


def save_visual_model(model, model_type: str, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_type == "extra_trees_visual_wam":
        import joblib

        joblib.dump({"model": model, "metadata": metadata}, path)
    else:
        state = model.state_dict()
        np.savez(path, **state, metadata=np.asarray(metadata, dtype=object))


def image_features(frame: np.ndarray, grid: int = 16) -> np.ndarray:
    """Compact RGB features from a rendered benchmark frame."""

    img = np.asarray(frame, dtype=float)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("expected RGB frame")
    img = img[..., :3] / 255.0
    h, w, _ = img.shape
    crop = img[: h - (h % grid), : w - (w % grid), :]
    blocks = crop.reshape(grid, crop.shape[0] // grid, grid, crop.shape[1] // grid, 3).mean(axis=(1, 3))
    gray = np.mean(blocks, axis=2)
    channel_mean = img.mean(axis=(0, 1))
    channel_std = img.std(axis=(0, 1))
    center_crop = img[h // 8 : 7 * h // 8, w // 8 : 7 * w // 8]
    crop_h, crop_w, _ = center_crop.shape
    crop_grid = 12
    crop = center_crop[: crop_h - (crop_h % crop_grid), : crop_w - (crop_w % crop_grid), :]
    crop_blocks = crop.reshape(crop_grid, crop.shape[0] // crop_grid, crop_grid, crop.shape[1] // crop_grid, 3).mean(axis=(1, 3))

    # Color-position moments. Reacher has a red target and colored arm/end-effector
    # blobs, so these features act like a lightweight vision front-end rather than
    # privileged simulator state.
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx / max(1, w - 1)
    yy = yy / max(1, h - 1)
    moments = []
    for c in range(3):
        weights = img[..., c] + 1e-6
        total = float(np.sum(weights))
        moments.extend([float(np.sum(xx * weights) / total), float(np.sum(yy * weights) / total)])

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    masks = [
        (r > 0.35) & (r > g + 0.07) & (r > b + 0.07),
        (g > 0.25) & (g > r + 0.04),
        (b > 0.25) & (b > r + 0.04),
        (r > 0.18) & (b > 0.18) & (g < 0.16),
        np.std(img, axis=2) > 0.045,
    ]
    mask_features = []
    for mask in masks:
        area = float(np.mean(mask))
        if np.any(mask):
            mx = xx[mask]
            my = yy[mask]
            mask_features.extend(
                [
                    area,
                    float(np.mean(mx)),
                    float(np.mean(my)),
                    float(np.std(mx)),
                    float(np.std(my)),
                    float(np.min(mx)),
                    float(np.max(mx)),
                    float(np.min(my)),
                    float(np.max(my)),
                ]
            )
        else:
            mask_features.extend([0.0] * 9)
    return np.concatenate(
        [
            gray.reshape(-1),
            blocks.reshape(-1),
            crop_blocks.reshape(-1),
            channel_mean,
            channel_std,
            np.asarray(moments, dtype=float),
            np.asarray(mask_features, dtype=float),
        ]
    )


def action_features(actions: np.ndarray, max_horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=float)
    n, horizon, action_dim = actions.shape
    padded = np.zeros((n, int(max_horizon), action_dim), dtype=float)
    padded[:, :horizon, :] = actions
    norms = np.linalg.norm(actions, axis=2)
    return np.column_stack(
        [
            padded.reshape(n, int(max_horizon) * action_dim),
            actions.sum(axis=1),
            norms.mean(axis=1),
            norms.max(axis=1),
            np.sum(actions * actions, axis=(1, 2)),
            np.full(n, horizon / max(1, int(max_horizon))),
        ]
    )


def render_state(adapter: GymManipAdapter, state: np.ndarray) -> np.ndarray:
    adapter.set_state(state)
    frame = adapter.env.render()
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3 or float(np.std(frame)) <= 1e-6:
        raise RuntimeError(f"rendered frame is invalid: shape={frame.shape}, std={float(np.std(frame))}")
    return frame[..., :3]


def dataset_rows(adapter: GymManipAdapter, states: int, rollouts: int, horizon: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    x_rows = []
    y_rows = []
    meta_rows = []
    for state_id in range(int(states)):
        state = adapter.reset(seed + 4099 * state_id)
        frame = render_state(adapter, state)
        img_feat = image_features(frame)
        pool = sample_rollout_pool(adapter, state, rollouts, horizon, seed + 100_003 * (state_id + 1))
        actions = pool["actions"]
        records = pool["records"]
        af = action_features(actions, horizon)
        x = np.column_stack([np.repeat(img_feat[None, :], len(actions), axis=0), af])
        utilities = np.asarray([r["utility"] for r in records], dtype=float)
        success = np.asarray([float(r["success"]) for r in records], dtype=float)
        x_rows.append(x)
        y_rows.append(np.column_stack([utilities, success]))
        for i, rec in enumerate(records):
            meta_rows.append(
                {
                    "state_id": int(state_id),
                    "rollout_id": int(i),
                    "utility": float(rec["utility"]),
                    "success": float(rec["success"]),
                    "energy": float(rec["energy"]),
                }
            )
    return np.vstack(x_rows), np.vstack(y_rows), meta_rows


def run(args: argparse.Namespace) -> dict:
    ensure_result_dirs()
    adapter = GymManipAdapter(env_id=args.env_id, render_mode="rgb_array", horizon=args.horizon, success_threshold=args.success_threshold)
    try:
        train_x, train_y, _ = dataset_rows(adapter, args.train_states, args.train_rollouts, args.horizon, args.seed, )
        val_x, val_y, _ = dataset_rows(adapter, args.val_states, args.val_rollouts, args.horizon, args.seed + 10_000)
        model, model_type = fit_visual_model(train_x, train_y, ridge=args.ridge, seed=args.seed)
        val_pred = predict_visual_model(model, val_x)
        utility_mae = float(np.mean(np.abs(val_pred[:, 0] - val_y[:, 0])))
        utility_corr = float(np.corrcoef(val_pred[:, 0], val_y[:, 0])[0, 1]) if np.std(val_pred[:, 0]) > 1e-12 and np.std(val_y[:, 0]) > 1e-12 else 0.0
        success_mae = float(np.mean(np.abs(val_pred[:, 1] - val_y[:, 1])))

        model_suffix = "joblib" if model_type == "extra_trees_visual_wam" else "npz"
        model_path = results_dir() / "models" / f"benchmark_visual_reacher_wam_lite.{model_suffix}"
        save_visual_model(
            model,
            model_type,
            model_path,
            {"env_id": args.env_id, "mode": "rgb_frame_action_sequence", "model_type": model_type},
        )

        rows = []
        exact_rows = []
        seed_metrics = []
        example_frame = None
        for seed in args.seeds:
            for state_id in range(args.states):
                state = adapter.reset(seed + 811 * state_id)
                frame = render_state(adapter, state)
                if example_frame is None:
                    example_frame = frame.copy()
                img_feat = image_features(frame)
                pool = sample_rollout_pool(adapter, state, args.rollouts, args.horizon, seed + 65_537 * (state_id + 1))
                actions = pool["actions"]
                records = pool["records"]
                x = np.column_stack([np.repeat(img_feat[None, :], len(actions), axis=0), action_features(actions, args.horizon)])
                pred = predict_visual_model(model, x)
                real_utility = np.asarray([r["utility"] for r in records], dtype=float)
                energy = np.asarray([r["energy"] for r in records], dtype=float)
                random_scores = np.random.default_rng(seed + state_id).normal(size=len(real_utility))
                score_sets = {
                    "random": random_scores,
                    "visual_wam": pred[:, 0],
                    "visual_success_head": pred[:, 1],
                    "low_energy": -energy,
                    "oracle_real_utility": real_utility,
                }
                norm_utility = normalized_utility(real_utility)
                for scorer, scores in score_sets.items():
                    raw_curve = utility_best_of_n_finite(scores, real_utility, N_VALUES)
                    norm_curve = utility_best_of_n_finite(scores, norm_utility, N_VALUES)
                    for n in N_VALUES:
                        rows.append(
                            {
                                "seed": int(seed),
                                "state_id": int(state_id),
                                "scorer": scorer,
                                "N": int(n),
                                "real_utility": float(raw_curve[n]),
                                "normalized_real_utility": float(norm_curve[n]),
                            }
                        )
                for n in N_VALUES:
                    exact = utility_best_of_n_finite(pred[:, 0], real_utility, [n])[n]
                    mc = simulate_best_of_n(pred[:, 0], real_utility, n, args.mc_trials, seed + 17 * n + state_id)
                    exact_rows.append({"seed": int(seed), "state_id": int(state_id), "N": int(n), "utility_abs_error": abs(exact - mc)})

        df = pd.DataFrame(rows)
        exact_df = pd.DataFrame(exact_rows)
        table_path = results_dir() / "tables" / "benchmark_visual_wam_lite_curves.csv"
        exact_path = results_dir() / "tables" / "benchmark_visual_wam_lite_exact_law.csv"
        df.to_csv(table_path, index=False)
        exact_df.to_csv(exact_path, index=False)
        agg = df.groupby(["scorer", "N"], dropna=False)[["real_utility", "normalized_real_utility"]].mean().reset_index()
        agg_path = results_dir() / "tables" / "benchmark_visual_wam_lite_curves_aggregate.csv"
        agg.to_csv(agg_path, index=False)

        seed_agg = df.groupby(["seed", "scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
        for seed, sub in seed_agg.groupby("seed"):
            n32 = sub[sub["N"] == 32].set_index("scorer")["normalized_real_utility"]
            seed_metrics.append(
                {
                    "seed": int(seed),
                    "visual_minus_random_N32": float(n32["visual_wam"] - n32["random"]),
                    "oracle_minus_visual_N32": float(n32["oracle_real_utility"] - n32["visual_wam"]),
                    "visual_minus_low_energy_N32": float(n32["visual_wam"] - n32["low_energy"]),
                }
            )
        seed_df = pd.DataFrame(seed_metrics)
        seed_path = results_dir() / "tables" / "benchmark_visual_wam_lite_seed_metrics.csv"
        seed_df.to_csv(seed_path, index=False)

        plt.figure(figsize=(7.2, 4.6))
        for scorer, sub in agg.groupby("scorer"):
            if scorer in {"random", "visual_wam", "low_energy", "oracle_real_utility"}:
                plt.plot(sub["N"], sub["normalized_real_utility"], marker="o", label=scorer)
        plt.xscale("log", base=2)
        plt.xlabel("N")
        plt.ylabel("normalized real utility")
        plt.title("Benchmark RGB WAM-lite inference curves")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fig_path = results_dir() / "figures" / "benchmark_visual_wam_lite_curves.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()

        frame_path = results_dir() / "figures" / "benchmark_visual_wam_lite_frame.png"
        if example_frame is not None:
            iio.imwrite(frame_path, example_frame)

        summary = {
            "experiment": "benchmark_visual_wam_lite",
            "attempted": True,
            "verified": bool(utility_corr > args.min_corr and utility_mae < args.max_mae),
            "benchmark": args.env_id,
            "mode": "rgb_frame_action_sequence",
            "train_samples": int(len(train_x)),
            "validation_samples": int(len(val_x)),
            "model_path": str(model_path),
            "model_type": model_type,
            "validation": {
                "utility_mae": utility_mae,
                "utility_corr": utility_corr,
                "success_mae": success_mae,
            },
            "exact_law_utility_mae": float(exact_df["utility_abs_error"].mean()),
            "confidence_intervals": {
                "visual_minus_random_N32": ci95(seed_df["visual_minus_random_N32"].to_numpy()),
                "oracle_minus_visual_N32": ci95(seed_df["oracle_minus_visual_N32"].to_numpy()),
                "visual_minus_low_energy_N32": ci95(seed_df["visual_minus_low_energy_N32"].to_numpy()),
            },
            "artifacts": {
                "table": str(table_path),
                "aggregate": str(agg_path),
                "exact_law": str(exact_path),
                "seed_metrics": str(seed_path),
                "figure": str(fig_path),
                "frame": str(frame_path),
            },
        }
        write_json(results_dir() / "benchmark_visual_wam_lite.json", summary)
        return summary
    except Exception as exc:
        summary = {"experiment": "benchmark_visual_wam_lite", "attempted": True, "verified": False, "reason": f"{type(exc).__name__}: {exc}"}
        write_json(results_dir() / "benchmark_visual_wam_lite.json", summary)
        return summary
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Reacher-v5")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--success-threshold", type=float, default=0.07)
    parser.add_argument("--train-states", type=int, default=24)
    parser.add_argument("--train-rollouts", type=int, default=64)
    parser.add_argument("--val-states", type=int, default=8)
    parser.add_argument("--val-rollouts", type=int, default=64)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--mc-trials", type=int, default=800)
    parser.add_argument("--ridge", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=15001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[15001, 15002, 15003, 15004, 15005])
    parser.add_argument("--min-corr", type=float, default=0.20)
    parser.add_argument("--max-mae", type=float, default=0.80)
    args = parser.parse_args()
    summary = run(args)
    val = summary.get("validation") or {}
    print(
        "benchmark visual WAM-lite complete: "
        f"verified={summary.get('verified')}, "
        f"utility_corr={val.get('utility_corr')}, utility_mae={val.get('utility_mae')}"
    )


if __name__ == "__main__":
    main()
