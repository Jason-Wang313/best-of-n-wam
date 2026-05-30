from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wam_inference_value.envs import DrawerPull1D
from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json
from wam_inference_value.rendering import render_1d_state


def run() -> dict:
    ensure_result_dirs()
    env = DrawerPull1D()
    rng = np.random.default_rng(404)
    x_rows = []
    y_rows = []
    for i in range(80):
        state = env.sample_state(1000 + i, mismatch="mild", state_id=i)
        action = np.asarray([rng.uniform(0.15, 0.9), rng.uniform(-0.25, 0.35)])
        next_state = env.step(state, action, state.params)
        img = render_1d_state(float(state.vector[0]), float(state.target[0]), size=64)
        x_rows.append(np.concatenate([img.reshape(-1), action]))
        y_rows.append([next_state.vector[0] - state.vector[0], env.distance_to_target(next_state)])
    x = np.asarray(x_rows)
    y = np.asarray(y_rows)
    train = np.arange(0, 60)
    test = np.arange(60, 80)
    x_mean = x[train].mean(axis=0)
    x_std = np.where(x[train].std(axis=0) < 1e-8, 1.0, x[train].std(axis=0))
    xa = np.column_stack([np.ones(len(train)), (x[train] - x_mean) / x_std])
    coef = np.linalg.pinv(xa.T @ xa + 1e-3 * np.eye(xa.shape[1])) @ xa.T @ y[train]
    xt = np.column_stack([np.ones(len(test)), (x[test] - x_mean) / x_std])
    pred = xt @ coef
    mae = float(np.mean(np.abs(pred - y[test])))

    example = render_1d_state(0.25, 0.9, size=64)
    fig_path = results_dir() / "figures" / "visual_toy_render_example.png"
    plt.figure(figsize=(2.4, 2.4))
    plt.imshow(example, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    summary = {
        "experiment": "visual_optional",
        "attempted": True,
        "verified": mae < 0.2,
        "mode": "toy_low_res_state_render",
        "test_mae": mae,
        "artifact": str(fig_path),
    }
    write_json(results_dir() / "visual_optional.json", summary)
    return summary


def main() -> None:
    summary = run()
    print(f"visual optional attempted: verified={summary['verified']} test_mae={summary['test_mae']:.4f}")


if __name__ == "__main__":
    main()
