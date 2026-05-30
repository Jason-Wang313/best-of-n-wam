from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v3 as iio
import numpy as np

from wam_inference_value.benchmarks.gym_manip_adapter import GymManipAdapter
from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json


def run() -> dict:
    ensure_result_dirs()
    try:
        adapter = GymManipAdapter(env_id="Reacher-v5", render_mode="rgb_array")
        try:
            adapter.reset(123)
            frame = adapter.env.render()
            frame = np.asarray(frame)
            verified = frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0 and float(np.std(frame)) > 1e-6
            out_path = results_dir() / "figures" / "benchmark_visual_reacher_frame.png"
            iio.imwrite(out_path, frame)
            summary = {
                "experiment": "benchmark_visual_optional",
                "attempted": True,
                "verified": bool(verified),
                "benchmark": "Reacher-v5",
                "frame_shape": list(frame.shape),
                "frame_std": float(np.std(frame)),
                "artifact": str(out_path),
            }
        finally:
            adapter.close()
    except Exception as exc:
        summary = {
            "experiment": "benchmark_visual_optional",
            "attempted": True,
            "verified": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    write_json(results_dir() / "benchmark_visual_optional.json", summary)
    return summary


def main() -> None:
    summary = run()
    print(f"benchmark visual optional attempted: verified={summary.get('verified')}")


if __name__ == "__main__":
    main()
