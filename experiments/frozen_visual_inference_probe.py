from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wam_inference_value.benchmarks.gym_robotics_adapter import GymRoboticsAdapter, is_gym_robotics_available
from wam_inference_value.benchmarks.gym_robotics_rollouts import sample_rollout_pool
from wam_inference_value.evaluation import ci95, ensure_result_dirs, results_dir, write_json
from wam_inference_value.models.simple_wam import RidgeRegressor
from wam_inference_value.stats import normalized_utility
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32]
PROMPTS = {
    "FetchReach-v4": "a robot gripper reaching the target",
    "FetchPush-v4": "a robot gripper pushing the block to the target",
    "FetchPickAndPlace-v4": "a robot gripper placing the object at the target",
}


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def mock_image_embedding(frame: np.ndarray, dim: int = 256) -> np.ndarray:
    img = np.asarray(frame, dtype=np.float32)[..., :3] / 255.0
    h, w, _ = img.shape
    grid = 8
    crop = img[: h - (h % grid), : w - (w % grid), :]
    blocks = crop.reshape(grid, crop.shape[0] // grid, grid, crop.shape[1] // grid, 3).mean(axis=(1, 3)).reshape(-1)
    stats = np.concatenate([img.mean(axis=(0, 1)), img.std(axis=(0, 1)), blocks])
    if stats.size < dim:
        reps = int(math.ceil(dim / stats.size))
        stats = np.tile(stats, reps)
    return stats[:dim].astype(np.float32)


def mock_text_embedding(text: str, dim: int = 256) -> np.ndarray:
    rng = np.random.default_rng(stable_int(text))
    vec = rng.normal(size=dim).astype(np.float32)
    return vec


@dataclass
class EmbeddingBackend:
    model_name: str
    device: str
    mock: bool = False
    require_cuda: bool = False
    allow_cpu_fallback: bool = True
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model = None
        self.processor = None
        self.requested_device = self.device
        if self.mock:
            if self.require_cuda:
                raise RuntimeError("--require-cuda cannot be satisfied by mock embeddings")
            self.embedding_dim = 256
            self.runtime_metadata = {
                "requested_device": self.requested_device,
                "selected_device": "mock",
                "mock_embeddings": True,
                "require_cuda": bool(self.require_cuda),
                "cpu_fallback_allowed": bool(self.allow_cpu_fallback),
            }
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.runtime_metadata = torch_runtime_metadata(self.requested_device)
        self.runtime_metadata["require_cuda"] = bool(self.require_cuda)
        self.runtime_metadata["cpu_fallback_allowed"] = bool(self.allow_cpu_fallback)
        self.runtime_metadata["weights_format"] = "safetensors"
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.require_cuda and self.device != "cuda":
            raise RuntimeError(
                f"--require-cuda requested but selected device is {self.device}; "
                f"cuda_available={torch.cuda.is_available()}"
            )
        if self.device == "cuda":
            self.update_selected_device_metadata()
            self._cuda_execution_sanity_check()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name, use_safetensors=True)
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as exc:
            if not self._fallback_to_cpu(exc):
                raise
        self.model.eval()
        self.embedding_dim = int(self.model.config.projection_dim)
        self.update_selected_device_metadata()

    def _cuda_execution_sanity_check(self) -> None:
        import torch

        try:
            probe = torch.ones((1,), device="cuda")
            value = (probe + 1.0).detach().cpu().item()
            if float(value) != 2.0:
                raise RuntimeError(f"unexpected CUDA sanity value {value!r}")
            torch.cuda.synchronize()
        except Exception as exc:
            if self.requested_device == "auto" and self.allow_cpu_fallback and not self.require_cuda:
                print(f"warning: CUDA sanity check failed ({exc}); falling back to CPU", file=sys.stderr, flush=True)
                self.device = "cpu"
                self.runtime_metadata["fallback_to_cpu"] = True
                self.runtime_metadata["fallback_reason"] = f"cuda sanity check failed: {exc}"
                self.update_selected_device_metadata()
                return
            raise RuntimeError(f"CUDA sanity check failed before model load: {exc}") from exc

    def _fallback_to_cpu(self, exc: RuntimeError) -> bool:
        if self.requested_device == "auto" and self.device == "cuda" and self.allow_cpu_fallback and not self.require_cuda:
            from transformers import CLIPModel

            print(f"warning: CUDA backend failed ({exc}); falling back to CPU", file=sys.stderr, flush=True)
            self.device = "cpu"
            self.model = CLIPModel.from_pretrained(self.model_name, use_safetensors=True).to(self.device)
            self.model.eval()
            self.runtime_metadata["fallback_to_cpu"] = True
            self.runtime_metadata["fallback_reason"] = str(exc)
            self.update_selected_device_metadata()
            return True
        return False

    def update_selected_device_metadata(self) -> None:
        self.runtime_metadata["selected_device"] = self.device
        if self.device == "cuda":
            self.runtime_metadata.update(cuda_selected_metadata())

    def _image_feature_tensor(self, feats: Any) -> Any:
        import torch

        if isinstance(feats, torch.Tensor):
            return feats
        if getattr(feats, "image_embeds", None) is not None:
            return feats.image_embeds
        if getattr(feats, "pooler_output", None) is not None and self.model is not None:
            if int(feats.pooler_output.shape[-1]) == self.embedding_dim:
                return feats.pooler_output
            return self.model.visual_projection(feats.pooler_output)
        raise TypeError(f"unexpected image feature output type: {type(feats)!r}")

    def _text_feature_tensor(self, feats: Any) -> Any:
        import torch

        if isinstance(feats, torch.Tensor):
            return feats
        if getattr(feats, "text_embeds", None) is not None:
            return feats.text_embeds
        if getattr(feats, "pooler_output", None) is not None and self.model is not None:
            if int(feats.pooler_output.shape[-1]) == self.embedding_dim:
                return feats.pooler_output
            return self.model.text_projection(feats.pooler_output)
        raise TypeError(f"unexpected text feature output type: {type(feats)!r}")

    def image_embeddings(self, frames: list[np.ndarray], batch_size: int) -> np.ndarray:
        if self.mock:
            return l2_normalize(np.stack([mock_image_embedding(frame, self.embedding_dim) for frame in frames], axis=0))

        import torch

        assert self.model is not None and self.processor is not None
        self.update_selected_device_metadata()
        try:
            outs = []
            with torch.no_grad():
                for start in range(0, len(frames), int(batch_size)):
                    batch = frames[start : start + int(batch_size)]
                    inputs = self.processor(images=batch, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    feats = self._image_feature_tensor(self.model.get_image_features(**inputs))
                    feats = torch.nn.functional.normalize(feats, dim=1)
                    outs.append(feats.detach().cpu().numpy().astype(np.float32))
        except RuntimeError as exc:
            if self._fallback_to_cpu(exc):
                return self.image_embeddings(frames, batch_size)
            raise
        return np.vstack(outs)

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        if self.mock:
            return l2_normalize(np.stack([mock_text_embedding(text, self.embedding_dim) for text in texts], axis=0))

        import torch

        assert self.model is not None and self.processor is not None
        self.update_selected_device_metadata()
        try:
            with torch.no_grad():
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                feats = self._text_feature_tensor(self.model.get_text_features(**inputs))
                feats = torch.nn.functional.normalize(feats, dim=1)
                return feats.detach().cpu().numpy().astype(np.float32)
        except RuntimeError as exc:
            if self._fallback_to_cpu(exc):
                return self.text_embeddings(texts)
            raise


def torch_runtime_metadata(requested_device: str) -> dict[str, Any]:
    import torch

    metadata: dict[str, Any] = {
        "requested_device": requested_device,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        metadata.update(cuda_selected_metadata())
    return metadata


def cuda_selected_metadata() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {}
    idx = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda_selected_index": idx,
        "cuda_selected_name": torch.cuda.get_device_name(idx),
        "cuda_capability": list(torch.cuda.get_device_capability(idx)),
        "cuda_total_memory_gb": float(props.total_memory / (1024**3)),
    }


def render_state(adapter: GymRoboticsAdapter, state: np.ndarray) -> np.ndarray:
    adapter.set_state(state)
    frame = np.asarray(adapter.env.render())
    if frame.ndim != 3 or frame.shape[2] < 3 or float(np.std(frame)) <= 1e-6:
        raise RuntimeError(f"invalid rendered frame shape={frame.shape} std={float(np.std(frame))}")
    return frame[..., :3]


def mismatch_prompt(env_id: str, env_ids: list[str]) -> str:
    others = [candidate for candidate in env_ids if candidate != env_id]
    return PROMPTS[others[0] if others else env_id]


def collect_rows(
    adapter: GymRoboticsAdapter,
    env_id: str,
    split: str,
    seeds: list[int],
    states: int,
    rollouts: int,
    horizon: int,
    backend: EmbeddingBackend,
    batch_size: int,
    env_ids: list[str],
) -> pd.DataFrame:
    prompt_emb = backend.text_embeddings([PROMPTS[env_id], mismatch_prompt(env_id, env_ids)])
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for state_id in range(int(states)):
            state = adapter.reset(seed + 1231 * state_id)
            pool = sample_rollout_pool(adapter, state, int(rollouts), int(horizon), seed + 65_537 * (state_id + 1))
            frames = [render_state(adapter, np.asarray(rec["final_state"], dtype=float)) for rec in pool["records"]]
            emb = backend.image_embeddings(frames, batch_size=batch_size)
            scores = emb @ prompt_emb.T
            for rollout_id, rec in enumerate(pool["records"]):
                actions = np.asarray(rec["actions"], dtype=float)
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
                        "clip_task_text": float(scores[rollout_id, 0]),
                        "clip_mismatch_text": float(scores[rollout_id, 1]),
                        "embedding": emb[rollout_id].astype(np.float32),
                    }
                )
    return pd.DataFrame(rows)


def fit_ridge(rows: pd.DataFrame, ridge: float, *, shuffle: bool, seed: int) -> RidgeRegressor:
    x = np.stack(rows["embedding"].to_list(), axis=0)
    y = rows["utility"].to_numpy(dtype=float)
    if shuffle:
        rng = np.random.default_rng(seed)
        y = y.copy()
        rng.shuffle(y)
    return RidgeRegressor(ridge=ridge).fit(x, y)


def predict_ridge(model: RidgeRegressor, rows: pd.DataFrame) -> np.ndarray:
    x = np.stack(rows["embedding"].to_list(), axis=0)
    pred = np.asarray(model.predict(x), dtype=float)
    return pred.reshape(-1)


def rows_without_embeddings(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.drop(columns=["embedding"])


def curve_rows(eval_rows: pd.DataFrame, mc_trials: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    scorer_names = [
        "random",
        "clip_task_text",
        "clip_mismatch_text",
        "clip_ridge",
        "clip_ridge_shuffled",
        "low_energy",
        "oracle_real_utility",
    ]
    for pool_id, pool in eval_rows.groupby("pool_id", sort=True):
        utility = pool["utility"].to_numpy(dtype=float)
        norm_utility = normalized_utility(utility)
        success = pool["success"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed + stable_int(str(pool_id)) % 1_000_000)
        score_sets = {
            "random": rng.normal(size=len(pool)),
            "clip_task_text": pool["clip_task_text"].to_numpy(dtype=float),
            "clip_mismatch_text": pool["clip_mismatch_text"].to_numpy(dtype=float),
            "clip_ridge": pool["clip_ridge"].to_numpy(dtype=float),
            "clip_ridge_shuffled": pool["clip_ridge_shuffled"].to_numpy(dtype=float),
            "low_energy": -pool["energy"].to_numpy(dtype=float),
            "oracle_real_utility": utility,
        }
        meta = pool.iloc[0]
        for scorer in scorer_names:
            scores = np.asarray(score_sets[scorer], dtype=float)
            raw_curve = utility_best_of_n_finite(scores, utility, N_VALUES)
            norm_curve = utility_best_of_n_finite(scores, norm_utility, N_VALUES)
            succ_curve = utility_best_of_n_finite(scores, success, N_VALUES)
            for n in N_VALUES:
                rows.append(
                    {
                        "benchmark": meta["benchmark"],
                        "seed": int(meta["seed"]),
                        "state_id": int(meta["state_id"]),
                        "pool_id": pool_id,
                        "scorer": scorer,
                        "N": int(n),
                        "real_utility": float(raw_curve[n]),
                        "normalized_real_utility": float(norm_curve[n]),
                        "success": float(succ_curve[n]),
                    }
                )
            if scorer in {"clip_task_text", "clip_ridge"}:
                for n in N_VALUES:
                    exact = utility_best_of_n_finite(scores, utility, [n])[n]
                    mc = simulate_best_of_n(scores, utility, n, int(mc_trials), seed + 17 * n + stable_int(f"{pool_id}:{scorer}") % 1_000_000)
                    exact_rows.append(
                        {
                            "benchmark": meta["benchmark"],
                            "seed": int(meta["seed"]),
                            "state_id": int(meta["state_id"]),
                            "pool_id": pool_id,
                            "scorer": scorer,
                            "N": int(n),
                            "utility_abs_error": float(abs(exact - mc)),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(exact_rows)


def seed_metric_rows(curves: pd.DataFrame) -> pd.DataFrame:
    seed_agg = curves.groupby(["seed", "scorer", "N"], dropna=False)["normalized_real_utility"].mean().reset_index()
    metrics = []
    for seed, sub in seed_agg.groupby("seed"):
        nmax = sub[sub["N"] == max(N_VALUES)].set_index("scorer")["normalized_real_utility"]
        row = {"seed": int(seed)}
        for scorer in ["clip_task_text", "clip_mismatch_text", "clip_ridge", "clip_ridge_shuffled", "low_energy", "oracle_real_utility"]:
            row[f"{scorer}_minus_random_N{max(N_VALUES)}"] = float(nmax[scorer] - nmax["random"])
        metrics.append(row)
    return pd.DataFrame(metrics)


def audit_rows(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool_id, scorer), sub in curves.groupby(["pool_id", "scorer"], dropna=False):
        if scorer == "oracle_real_utility":
            continue
        ordered = sub.sort_values("N")
        nmax_row = ordered[ordered["N"] == max(N_VALUES)].iloc[0]
        best_row = ordered.loc[ordered["normalized_real_utility"].idxmax()]
        rows.append(
            {
                "pool_id": pool_id,
                "benchmark": nmax_row["benchmark"],
                "seed": int(nmax_row["seed"]),
                "state_id": int(nmax_row["state_id"]),
                "scorer": scorer,
                "blind_N": int(max(N_VALUES)),
                "blind_normalized_real_utility": float(nmax_row["normalized_real_utility"]),
                "audit_best_N": int(best_row["N"]),
                "audit_best_normalized_real_utility": float(best_row["normalized_real_utility"]),
                "audit_best_minus_blind": float(best_row["normalized_real_utility"] - nmax_row["normalized_real_utility"]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_outputs(
    eval_df: pd.DataFrame,
    env_summaries: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    args: argparse.Namespace,
    backend: EmbeddingBackend,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "frozen_visual_inference_candidates.csv"
    rows_without_embeddings(eval_df).to_csv(candidates_path, index=False)
    curves, exact = curve_rows(eval_df, args.mc_trials, args.seed)
    curves_path = out_dir / "frozen_visual_inference_curves.csv"
    exact_path = out_dir / "frozen_visual_inference_exact_law.csv"
    curves.to_csv(curves_path, index=False)
    exact.to_csv(exact_path, index=False)
    aggregate = curves.groupby(["scorer", "N"], dropna=False)[["real_utility", "normalized_real_utility", "success"]].mean().reset_index()
    aggregate_path = out_dir / "frozen_visual_inference_curves_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)
    seed_metrics = seed_metric_rows(curves)
    seed_metrics_path = out_dir / "frozen_visual_inference_seed_metrics.csv"
    seed_metrics.to_csv(seed_metrics_path, index=False)
    audits = audit_rows(curves)
    audits_path = out_dir / "frozen_visual_inference_audit_best_n.csv"
    audits.to_csv(audits_path, index=False)

    plt.figure(figsize=(7.5, 4.8))
    for scorer, sub in aggregate.groupby("scorer"):
        if scorer in {"random", "clip_task_text", "clip_mismatch_text", "clip_ridge", "clip_ridge_shuffled", "oracle_real_utility"}:
            plt.plot(sub["N"], sub["normalized_real_utility"], marker="o", label=scorer)
    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("normalized real utility")
    plt.title("Frozen visual scorer score-tail audit")
    plt.legend(fontsize=7)
    plt.tight_layout()
    figure_path = out_dir / "frozen_visual_inference_curves.png"
    plt.savefig(figure_path, dpi=170)
    plt.close()

    exact_mae = float(exact["utility_abs_error"].mean()) if not exact.empty else None
    cis = {
        column: ci95(seed_metrics[column].to_numpy(dtype=float))
        for column in seed_metrics.columns
        if column != "seed"
    }
    audit_cis = {
        scorer: ci95(sub["audit_best_minus_blind"].to_numpy(dtype=float))
        for scorer, sub in audits.groupby("scorer")
    }
    summary = {
        "experiment": "frozen_visual_inference_probe",
        "attempted": True,
        "available": True,
        "verified": bool(
            exact_mae is not None
            and exact_mae < args.max_exact_mae
            and (
                cis.get(f"clip_task_text_minus_random_N{max(N_VALUES)}", {}).get("lo", -1.0) > 0.0
                or cis.get(f"clip_ridge_minus_random_N{max(N_VALUES)}", {}).get("lo", -1.0) > 0.0
            )
        ),
        "model_name": args.model_name,
        "device": backend.device,
        "requested_device": backend.requested_device,
        "gpu_verified": bool(backend.device == "cuda" and not args.mock_embeddings),
        "require_cuda": bool(getattr(args, "require_cuda", False)),
        "cpu_fallback_allowed": bool(not getattr(args, "no_cpu_fallback", False)),
        "runtime": backend.runtime_metadata,
        "mock_embeddings": bool(args.mock_embeddings),
        "env_ids": [row["benchmark"] for row in env_summaries],
        "unavailable": unavailable,
        "n_values": N_VALUES,
        "candidate_rows": int(len(eval_df)),
        "rollout_pools": int(eval_df["pool_id"].nunique()),
        "exact_law_utility_mae": exact_mae,
        "env_summaries": env_summaries,
        "confidence_intervals": cis,
        "audit_best_n_diagnostic_cis": audit_cis,
        "claim_boundaries": {
            "real_robot": False,
            "modern_vla_scale_sota": False,
            "policy_training": False,
            "evidence_type": "frozen public visual-language inference over simulator rollout-pool final frames",
        },
        "artifacts": {
            "protocol": str(ROOT / "docs" / "frozen_visual_gpu_protocol.json"),
            "candidates": str(candidates_path),
            "curves": str(curves_path),
            "aggregate": str(aggregate_path),
            "exact_law": str(exact_path),
            "seed_metrics": str(seed_metrics_path),
            "audit_best_n": str(audits_path),
            "figure": str(figure_path),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def run_precomputed(args: argparse.Namespace, backend: EmbeddingBackend, out_dir: Path) -> dict[str, Any]:
    input_dir = Path(args.precomputed_input_dir)
    metadata_path = input_dir / "metadata.csv"
    frames_path = input_dir / "frames.npy"
    compressed_frames_path = input_dir / "frames.npz"
    metadata = pd.read_csv(metadata_path)
    if frames_path.exists():
        frames = np.load(frames_path)
    elif compressed_frames_path.exists():
        frames = np.load(compressed_frames_path)["frames"]
    else:
        raise FileNotFoundError(f"expected {frames_path} or {compressed_frames_path}")
    if len(metadata) != len(frames):
        raise ValueError(f"metadata rows {len(metadata)} != frames {len(frames)}")
    embeddings = backend.image_embeddings([frame for frame in frames], batch_size=args.batch_size)
    metadata = metadata.copy()
    metadata["embedding"] = list(embeddings)
    unavailable: list[dict[str, Any]] = []
    env_summaries = []
    all_eval = []
    for env_id, env_rows in metadata.groupby("benchmark", sort=True):
        train = env_rows[env_rows["split"] == "train"].copy()
        val = env_rows[env_rows["split"] == "validation"].copy()
        eval_rows = env_rows[env_rows["split"] == "eval"].copy()
        prompt_emb = backend.text_embeddings([PROMPTS[str(env_id)], mismatch_prompt(str(env_id), sorted(metadata["benchmark"].unique()))])
        env_emb = np.stack(env_rows["embedding"].to_list(), axis=0)
        env_scores = env_emb @ prompt_emb.T
        metadata.loc[env_rows.index, "clip_task_text"] = env_scores[:, 0]
        metadata.loc[env_rows.index, "clip_mismatch_text"] = env_scores[:, 1]
        train = metadata.loc[train.index].copy()
        val = metadata.loc[val.index].copy()
        eval_rows = metadata.loc[eval_rows.index].copy()
        ridge = fit_ridge(train, args.ridge, shuffle=False, seed=args.seed + stable_int(str(env_id)) % 10_000)
        shuffled = fit_ridge(train, args.ridge, shuffle=True, seed=args.seed + 50_000 + stable_int(str(env_id)) % 10_000)
        val_pred = predict_ridge(ridge, val)
        val_y = val["utility"].to_numpy(dtype=float)
        val_mae = float(np.mean(np.abs(val_pred - val_y)))
        val_corr = float(np.corrcoef(val_pred, val_y)[0, 1]) if np.std(val_pred) > 1e-12 and np.std(val_y) > 1e-12 else 0.0
        eval_rows["clip_ridge"] = predict_ridge(ridge, eval_rows)
        eval_rows["clip_ridge_shuffled"] = predict_ridge(shuffled, eval_rows)
        all_eval.append(eval_rows)
        env_summaries.append(
            {
                "benchmark": str(env_id),
                "train_samples": int(len(train)),
                "validation_samples": int(len(val)),
                "eval_samples": int(len(eval_rows)),
                "ridge_validation_utility_mae": val_mae,
                "ridge_validation_utility_corr": val_corr,
            }
        )
    eval_df = pd.concat(all_eval, ignore_index=True)
    return write_summary_outputs(eval_df, env_summaries, unavailable, args, backend, out_dir)


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    out_dir = results_dir() / "frozen_visual_inference_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = EmbeddingBackend(
        model_name=args.model_name,
        device=args.device,
        mock=args.mock_embeddings,
        require_cuda=bool(getattr(args, "require_cuda", False)),
        allow_cpu_fallback=not bool(getattr(args, "no_cpu_fallback", False)),
    )
    if args.precomputed_input_dir:
        return run_precomputed(args, backend, out_dir)
    unavailable = []
    env_summaries = []
    all_eval = []
    for env_id in args.env_ids:
        ok, reason = is_gym_robotics_available(env_id)
        if not ok:
            unavailable.append({"env_id": env_id, "reason": reason})
            continue
        adapter = GymRoboticsAdapter(env_id=env_id, render_mode="rgb_array", horizon=args.horizon)
        try:
            env_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(env_id)) % 10_000
            train = collect_rows(
                adapter,
                env_id,
                "train",
                [args.seed + env_offset],
                args.train_states,
                args.train_rollouts,
                args.horizon,
                backend,
                args.batch_size,
                list(args.env_ids),
            )
            val = collect_rows(
                adapter,
                env_id,
                "validation",
                [args.seed + 20_000 + env_offset],
                args.val_states,
                args.val_rollouts,
                args.horizon,
                backend,
                args.batch_size,
                list(args.env_ids),
            )
            ridge = fit_ridge(train, args.ridge, shuffle=False, seed=args.seed + env_offset)
            shuffled = fit_ridge(train, args.ridge, shuffle=True, seed=args.seed + 50_000 + env_offset)
            val_pred = predict_ridge(ridge, val)
            val_y = val["utility"].to_numpy(dtype=float)
            val_mae = float(np.mean(np.abs(val_pred - val_y)))
            val_corr = float(np.corrcoef(val_pred, val_y)[0, 1]) if np.std(val_pred) > 1e-12 and np.std(val_y) > 1e-12 else 0.0
            eval_rows = collect_rows(
                adapter,
                env_id,
                "eval",
                list(args.seeds),
                args.states,
                args.rollouts,
                args.horizon,
                backend,
                args.batch_size,
                list(args.env_ids),
            )
            eval_rows["clip_ridge"] = predict_ridge(ridge, eval_rows)
            eval_rows["clip_ridge_shuffled"] = predict_ridge(shuffled, eval_rows)
            all_eval.append(eval_rows)
            env_summaries.append(
                {
                    "benchmark": env_id,
                    "train_samples": int(len(train)),
                    "validation_samples": int(len(val)),
                    "eval_samples": int(len(eval_rows)),
                    "ridge_validation_utility_mae": val_mae,
                    "ridge_validation_utility_corr": val_corr,
                }
            )
        finally:
            adapter.close()

    if not all_eval:
        summary = {
            "experiment": "frozen_visual_inference_probe",
            "attempted": True,
            "available": False,
            "unavailable": unavailable,
        }
        write_json(out_dir / "summary.json", summary)
        return summary

    eval_df = pd.concat(all_eval, ignore_index=True)
    return write_summary_outputs(eval_df, env_summaries, unavailable, args, backend, out_dir)


def run_gpu_preflight(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    out_dir = results_dir() / "frozen_visual_inference_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = EmbeddingBackend(
        model_name=args.model_name,
        device=args.device,
        mock=args.mock_embeddings,
        require_cuda=True,
        allow_cpu_fallback=False,
    )
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[..., 0] = 64
    frame[..., 1] = 128
    frame[..., 2] = 192
    image_emb = backend.image_embeddings([frame], batch_size=1)
    text_emb = backend.text_embeddings(["a robot gripper reaching the target"])
    score = float((image_emb @ text_emb.T)[0, 0])
    verified = bool(backend.device == "cuda" and not args.mock_embeddings)
    if not verified:
        raise RuntimeError(f"GPU preflight did not finish on CUDA; selected device={backend.device}")
    summary = {
        "experiment": "frozen_visual_gpu_preflight",
        "attempted": True,
        "available": True,
        "verified": verified,
        "model_name": args.model_name,
        "device": backend.device,
        "requested_device": backend.requested_device,
        "gpu_verified": verified,
        "require_cuda": True,
        "cpu_fallback_allowed": False,
        "runtime": backend.runtime_metadata,
        "image_embedding_shape": list(image_emb.shape),
        "text_embedding_shape": list(text_emb.shape),
        "image_text_score": score,
        "claim_boundaries": {
            "real_robot": False,
            "modern_vla_scale_sota": False,
            "policy_training": False,
            "evidence_type": "CUDA smoke test for frozen visual-language inference only",
        },
    }
    write_json(out_dir / "gpu_preflight.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen visual-language inference score-tail probe.")
    parser.add_argument("--env-ids", nargs="*", default=["FetchReach-v4", "FetchPush-v4", "FetchPickAndPlace-v4"])
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mock-embeddings", action="store_true")
    parser.add_argument("--require-cuda", action="store_true", help="Fail unless the selected device is CUDA.")
    parser.add_argument("--no-cpu-fallback", action="store_true", help="Disable automatic CUDA-to-CPU fallback after runtime CUDA errors.")
    parser.add_argument("--gpu-preflight", action="store_true", help="Run one strict CUDA CLIP smoke batch and write gpu_preflight.json.")
    parser.add_argument("--precomputed-input-dir", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--train-states", type=int, default=8)
    parser.add_argument("--train-rollouts", type=int, default=32)
    parser.add_argument("--val-states", type=int, default=3)
    parser.add_argument("--val-rollouts", type=int, default=32)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--mc-trials", type=int, default=500)
    parser.add_argument("--ridge", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[64001, 64002, 64003, 64004, 64005])
    parser.add_argument("--max-exact-mae", type=float, default=0.08)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gpu_preflight:
        args.require_cuda = True
        args.no_cpu_fallback = True
        summary = run_gpu_preflight(args)
    else:
        summary = run(args)
    print(
        "frozen visual inference probe: "
        f"available={summary.get('available')} verified={summary.get('verified')} "
        f"device={summary.get('device')} gpu_verified={summary.get('gpu_verified')} "
        f"pools={summary.get('rollout_pools')} "
        f"exact_mae={summary.get('exact_law_utility_mae')}"
    )


if __name__ == "__main__":
    main()
