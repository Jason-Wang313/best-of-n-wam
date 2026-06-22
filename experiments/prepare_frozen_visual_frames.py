from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from wam_inference_value.benchmarks.gym_robotics_adapter import GymRoboticsAdapter, is_gym_robotics_available
from wam_inference_value.benchmarks.gym_robotics_rollouts import sample_rollout_pool
from wam_inference_value.evaluation import write_json


DEFAULT_ENV_IDS = ["FetchReach-v4", "FetchPush-v4", "FetchPickAndPlace-v4"]


def render_state(adapter: GymRoboticsAdapter, state: np.ndarray, image_size: int) -> np.ndarray:
    adapter.set_state(state)
    frame = np.asarray(adapter.env.render())
    if frame.ndim != 3 or frame.shape[2] < 3 or float(np.std(frame)) <= 1e-6:
        raise RuntimeError(f"invalid rendered frame shape={frame.shape} std={float(np.std(frame))}")
    image = Image.fromarray(frame[..., :3].astype("uint8"))
    image = image.resize((int(image_size), int(image_size)), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def collect_split(
    adapter: GymRoboticsAdapter,
    env_id: str,
    split: str,
    seeds: list[int],
    states: int,
    rollouts: int,
    horizon: int,
    image_size: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    for seed in seeds:
        for state_id in range(int(states)):
            state = adapter.reset(seed + 1231 * state_id)
            pool = sample_rollout_pool(adapter, state, int(rollouts), int(horizon), seed + 65_537 * (state_id + 1))
            for rollout_id, rec in enumerate(pool["records"]):
                frames.append(render_state(adapter, np.asarray(rec["final_state"], dtype=float), image_size))
                rows.append(
                    {
                        "split": split,
                        "benchmark": env_id,
                        "seed": int(seed),
                        "state_id": int(state_id),
                        "pool_id": f"{env_id}:{split}:{seed}:{state_id}",
                        "rollout_id": int(rollout_id),
                        "utility": float(rec["utility"]),
                        "success": float(rec["success"]),
                        "energy": float(rec["energy"]),
                    }
                )
    return rows, frames


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_frames: list[np.ndarray] = []
    unavailable = []
    for env_id in args.env_ids:
        ok, reason = is_gym_robotics_available(env_id)
        if not ok:
            unavailable.append({"env_id": env_id, "reason": reason})
            continue
        env_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(env_id)) % 10_000
        adapter = GymRoboticsAdapter(env_id=env_id, render_mode="rgb_array", horizon=args.horizon)
        try:
            for split, seeds, states, rollouts in [
                ("train", [args.seed + env_offset], args.train_states, args.train_rollouts),
                ("validation", [args.seed + 20_000 + env_offset], args.val_states, args.val_rollouts),
                ("eval", list(args.seeds), args.states, args.rollouts),
            ]:
                rows, frames = collect_split(adapter, env_id, split, seeds, states, rollouts, args.horizon, args.image_size)
                all_rows.extend(rows)
                all_frames.extend(frames)
        finally:
            adapter.close()

    metadata = pd.DataFrame(all_rows)
    frames_array = np.stack(all_frames, axis=0) if all_frames else np.zeros((0, args.image_size, args.image_size, 3), dtype=np.uint8)
    metadata_path = out_dir / "metadata.csv"
    frames_path = out_dir / ("frames.npz" if args.compress_frames else "frames.npy")
    metadata.to_csv(metadata_path, index=False)
    if args.compress_frames:
        np.savez_compressed(frames_path, frames=frames_array)
    else:
        np.save(frames_path, frames_array)
    summary = {
        "experiment": "prepare_frozen_visual_frames",
        "available": bool(all_rows),
        "env_ids": sorted(metadata["benchmark"].unique().tolist()) if not metadata.empty else [],
        "unavailable": unavailable,
        "image_size": int(args.image_size),
        "rows": int(len(metadata)),
        "frames_shape": list(frames_array.shape),
        "metadata": str(metadata_path),
        "frames": str(frames_path),
        "splits": metadata["split"].value_counts().to_dict() if not metadata.empty else {},
    }
    write_json(out_dir / "input_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare static final-frame input bundle for frozen visual GPU inference.")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "frozen_visual_inference_input"))
    parser.add_argument("--env-ids", nargs="*", default=DEFAULT_ENV_IDS)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--compress-frames", action="store_true")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--train-states", type=int, default=8)
    parser.add_argument("--train-rollouts", type=int, default=32)
    parser.add_argument("--val-states", type=int, default=3)
    parser.add_argument("--val-rollouts", type=int, default=32)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[64001, 64002, 64003, 64004, 64005])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "prepared frozen visual frames: "
        f"available={summary.get('available')} rows={summary.get('rows')} shape={summary.get('frames_shape')}"
    )


if __name__ == "__main__":
    main()
