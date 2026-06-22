from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from argparse import Namespace

from experiments import frozen_visual_inference_probe as probe


def test_curve_seed_and_audit_rows_are_derived_from_candidate_scores() -> None:
    rows = []
    for seed in [1, 2]:
        for rollout_id, utility in enumerate([0.0, 0.25, 0.75, 1.0]):
            rows.append(
                {
                    "split": "eval",
                    "benchmark": "FetchReach-v4",
                    "seed": seed,
                    "state_id": 0,
                    "pool_id": f"pool-{seed}",
                    "rollout_id": rollout_id,
                    "utility": utility,
                    "success": float(utility > 0.5),
                    "energy": float(rollout_id),
                    "clip_task_text": utility,
                    "clip_mismatch_text": -utility,
                    "clip_ridge": utility,
                    "clip_ridge_shuffled": -utility,
                }
            )
    candidates = pd.DataFrame(rows)

    curves, exact = probe.curve_rows(candidates, mc_trials=50, seed=7)
    metrics = probe.seed_metric_rows(curves)
    audits = probe.audit_rows(curves)

    assert set(curves["scorer"]) >= {"clip_task_text", "clip_mismatch_text", "clip_ridge", "random", "oracle_real_utility"}
    assert not exact.empty
    assert f"clip_ridge_minus_random_N{max(probe.N_VALUES)}" in metrics.columns
    assert set(audits["scorer"]) >= {"clip_task_text", "clip_mismatch_text", "clip_ridge"}
    assert audits["audit_best_N"].between(min(probe.N_VALUES), max(probe.N_VALUES)).all()


def test_mock_backend_is_deterministic_and_normalized() -> None:
    backend = probe.EmbeddingBackend(model_name="mock", device="cpu", mock=True)
    frame = np.zeros((32, 32, 3), dtype="uint8")
    frame[..., 0] = 255

    emb1 = backend.image_embeddings([frame], batch_size=1)
    emb2 = backend.image_embeddings([frame], batch_size=1)
    text = backend.text_embeddings(["a robot gripper reaching the target"])

    assert emb1.shape == emb2.shape == (1, backend.embedding_dim)
    assert text.shape == (1, backend.embedding_dim)
    assert float(abs((emb1 * emb1).sum() - 1.0)) < 1e-5
    assert (emb1 == emb2).all()


def test_precomputed_input_path_writes_summary(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_dir.mkdir()
    rows = []
    frames = []
    for split, seed in [("train", 10), ("validation", 20), ("eval", 30)]:
        for rollout_id, utility in enumerate([0.0, 0.25, 0.75, 1.0]):
            rows.append(
                {
                    "split": split,
                    "benchmark": "FetchReach-v4",
                    "seed": seed,
                    "state_id": 0,
                    "pool_id": f"FetchReach-v4:{split}:{seed}:0",
                    "rollout_id": rollout_id,
                    "utility": utility,
                    "success": float(utility > 0.5),
                    "energy": float(rollout_id),
                }
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[..., 0] = rollout_id * 50
            frames.append(frame)
    pd.DataFrame(rows).to_csv(input_dir / "metadata.csv", index=False)
    np.save(input_dir / "frames.npy", np.stack(frames, axis=0))

    args = Namespace(
        precomputed_input_dir=str(input_dir),
        model_name="mock",
        device="cpu",
        mock_embeddings=True,
        batch_size=4,
        ridge=1.0,
        mc_trials=20,
        seed=99,
        max_exact_mae=1.0,
    )
    backend = probe.EmbeddingBackend(model_name="mock", device="cpu", mock=True)

    summary = probe.run_precomputed(args, backend, out_dir)

    assert summary["available"] is True
    assert summary["candidate_rows"] == 4
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "frozen_visual_inference_curves.csv").exists()
