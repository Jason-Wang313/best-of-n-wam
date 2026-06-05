from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from benchmark_libero_learned_action_head import (
    parse_task_ids,
    phase_targets,
    sanitize,
    scripted_action,
    task_index,
)
from libero_object_grasp_tuning import ALL_LIBERO_OBJECT_TASKS
from wam_inference_value.benchmarks.libero_adapter import LIBEROAdapter, LIBEROUnavailableError, is_libero_available
from wam_inference_value.stats import bootstrap_ci


RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for path in [RESULTS, RESULTS / "tables", RESULTS / "models", REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def artifact_layout(output_tag: str | None) -> dict[str, Path | str]:
    tag = str(output_tag or "").strip()
    if not tag or tag == "canonical":
        prefix = "benchmark_libero_visual_language_bc_policy"
        report_name = "libero_visual_language_bc_policy_report.md"
        tag = ""
    else:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_").lower()
        if not safe:
            safe = "tagged"
        prefix = f"benchmark_libero_{safe}"
        report_name = f"libero_{safe}_report.md"
        tag = safe
    return {
        "tag": tag,
        "prefix": prefix,
        "json": RESULTS / f"{prefix}.json",
        "episodes_csv": RESULTS / "tables" / f"{prefix}_episodes.csv",
        "model": RESULTS / "models" / f"{prefix}.npz",
        "report": REPORTS / report_name,
        "json_rel": f"results/{prefix}.json",
        "episodes_csv_rel": f"results/tables/{prefix}_episodes.csv",
        "model_rel": f"results/models/{prefix}.npz",
        "report_rel": f"reports/{report_name}",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "task_id",
        "task_name",
        "seed",
        "success",
        "total_reward",
        "initial_distance",
        "final_distance",
        "progress",
        "energy",
        "steps",
        "failure_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _obs_vector(raw: dict[str, Any], key: str, size: int) -> np.ndarray:
    if key not in raw:
        return np.zeros(size, dtype=float)
    try:
        arr = np.asarray(raw[key], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(size, dtype=float)
    if arr.size < size:
        arr = np.pad(arr, (0, size - arr.size))
    return arr[:size]


def _image_features(raw: dict[str, Any], key: str, grid: int) -> np.ndarray:
    if key not in raw:
        pooled = np.zeros((grid, grid), dtype=float)
        pooled_rgb = np.zeros((grid, grid, 3), dtype=float)
        color = np.zeros((0, 3), dtype=float)
    else:
        img = np.asarray(raw[key])
        if img.ndim != 3 or img.shape[-1] < 3:
            pooled = np.zeros((grid, grid), dtype=float)
            pooled_rgb = np.zeros((grid, grid, 3), dtype=float)
            color = np.zeros((0, 3), dtype=float)
        else:
            img = img[..., :3].astype(float) / 255.0
            h, w, _ = img.shape
            h2 = max(grid, (h // grid) * grid)
            w2 = max(grid, (w // grid) * grid)
            img = img[:h2, :w2]
            if img.shape[0] != h2 or img.shape[1] != w2:
                pad_h = max(0, h2 - img.shape[0])
                pad_w = max(0, w2 - img.shape[1])
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            gray = img.mean(axis=2)
            pooled = gray.reshape(grid, h2 // grid, grid, w2 // grid).mean(axis=(1, 3))
            pooled_rgb = img.reshape(grid, h2 // grid, grid, w2 // grid, 3).mean(axis=(1, 3))
            color = img.reshape(-1, 3)
    stats = []
    if color.size:
        stats.extend([color.mean(axis=0), color.std(axis=0), color.min(axis=0), color.max(axis=0)])
    else:
        stats.extend([np.zeros(3, dtype=float) for _ in range(4)])
    return np.concatenate([pooled.reshape(-1), pooled_rgb.reshape(-1), *stats]).astype(float)


def _language_features(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(int(dim), dtype=float)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % int(dim)
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def visual_language_feature(
    adapter: LIBEROAdapter,
    args: argparse.Namespace,
    prev_action: np.ndarray | None,
    step: int,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = adapter._raw_obs()
    language = str(getattr(adapter.task, "language", "") or getattr(adapter.task, "name", "") or "")
    prev = np.zeros(adapter.action_dim, dtype=float) if prev_action is None else np.asarray(prev_action, dtype=float).reshape(-1)
    if prev.size < adapter.action_dim:
        prev = np.pad(prev, (0, adapter.action_dim - prev.size))
    prev = prev[: adapter.action_dim]
    horizon = float(max(max_steps, 1))
    clock = np.asarray(
        [
            float(step) / horizon,
            np.sin(2.0 * np.pi * float(step) / horizon),
            np.cos(2.0 * np.pi * float(step) / horizon),
            np.sin(4.0 * np.pi * float(step) / horizon),
            np.cos(4.0 * np.pi * float(step) / horizon),
        ],
        dtype=float,
    )
    blocks = [
        ("rgb_agentview", _image_features(raw, "agentview_image", args.image_grid), args.image_weight),
        ("rgb_eye_in_hand", _image_features(raw, "robot0_eye_in_hand_image", args.image_grid), args.image_weight),
        ("robot_proprio", _obs_vector(raw, "robot0_proprio-state", 39), args.proprio_weight),
        ("eef_pos", _obs_vector(raw, "robot0_eef_pos", 3), args.proprio_weight),
        ("eef_quat", _obs_vector(raw, "robot0_eef_quat", 4), args.proprio_weight),
        ("gripper_qpos", _obs_vector(raw, "robot0_gripper_qpos", 2), args.proprio_weight),
        ("gripper_qvel", _obs_vector(raw, "robot0_gripper_qvel", 2), args.proprio_weight),
        ("prev_action", prev, args.prev_action_weight),
        ("step_clock", clock, args.clock_weight),
        ("language_hash", _language_features(language, args.language_hash_dim), args.language_weight),
    ]
    feature = np.concatenate([arr for _, arr, _ in blocks]).astype(float)
    weights = np.concatenate([np.full(arr.size, float(weight), dtype=float) for _, arr, weight in blocks])
    return feature, weights


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def knn_predict(
    x_train_z: np.ndarray,
    y_train: np.ndarray,
    x_z: np.ndarray,
    *,
    k: int,
    temperature: float,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    if candidate_indices is not None and candidate_indices.size:
        x_candidates = x_train_z[candidate_indices]
        y_candidates = y_train[candidate_indices]
    else:
        x_candidates = x_train_z
        y_candidates = y_train
    diff = x_candidates - x_z.reshape(1, -1)
    dist2 = np.sum(diff * diff, axis=1)
    k_eff = min(int(k), len(dist2))
    idx = np.argpartition(dist2, k_eff - 1)[:k_eff]
    local = dist2[idx]
    scale = max(float(temperature), 1e-8)
    weights = np.exp(-(local - float(np.min(local))) / scale)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 1e-12:
        weights = np.ones_like(weights)
    return np.average(y_candidates[idx], axis=0, weights=weights)


def _import_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional benchmark env.
        raise RuntimeError(f"torch unavailable for tiny neural VLA policy: {type(exc).__name__}: {exc}") from exc
    return torch


def train_tiny_neural_vla(
    x_train_z: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, np.ndarray], list[float]]:
    torch = _import_torch()
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(max(1, int(getattr(args, "torch_threads", 1))))
    device = torch.device("cpu")

    y_mean = np.mean(y_train, axis=0)
    y_scale = np.std(y_train, axis=0)
    y_scale[y_scale < 1e-8] = 1.0
    y_z = (y_train - y_mean.reshape(1, -1)) / y_scale.reshape(1, -1)

    x_tensor = torch.as_tensor(x_train_z.astype(np.float32), device=device)
    y_tensor = torch.as_tensor(y_z.astype(np.float32), device=device)
    n_features = int(x_tensor.shape[1])
    n_actions = int(y_tensor.shape[1])
    hidden = int(args.neural_hidden_dim)
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden),
        torch.nn.LayerNorm(hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.LayerNorm(hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, n_actions),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.neural_lr),
        weight_decay=float(args.neural_weight_decay),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    batch_size = min(int(args.neural_batch_size), int(x_tensor.shape[0]))
    losses: list[float] = []
    model.train()
    for _ in range(int(args.neural_epochs)):
        perm = torch.randperm(int(x_tensor.shape[0]), generator=generator, device=device)
        epoch_loss = 0.0
        n_seen = 0
        for start in range(0, int(x_tensor.shape[0]), batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_tensor[idx])
            loss = torch.nn.functional.smooth_l1_loss(pred, y_tensor[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            n_batch = int(idx.numel())
            epoch_loss += float(loss.detach().cpu()) * n_batch
            n_seen += n_batch
        losses.append(epoch_loss / max(n_seen, 1))

    arrays: dict[str, np.ndarray] = {
        "y_mean": y_mean.astype(np.float32),
        "y_scale": y_scale.astype(np.float32),
        "neural_train_loss": np.asarray(losses, dtype=np.float32),
    }
    for name, tensor in model.state_dict().items():
        arrays[f"torch_{name.replace('.', '_')}"] = tensor.detach().cpu().numpy().astype(np.float32)
    model.eval()
    return model, arrays, losses


def neural_predict(model: Any, x_z: np.ndarray, y_mean: np.ndarray, y_scale: np.ndarray) -> np.ndarray:
    torch = _import_torch()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_z.reshape(1, -1).astype(np.float32))
        pred_z = model(x_tensor).detach().cpu().numpy()[0]
    return pred_z * y_scale + y_mean


def neural_parameter_count(arrays: dict[str, np.ndarray]) -> int:
    return int(sum(int(arr.size) for key, arr in arrays.items() if key.startswith("torch_")))


def current_language(adapter: LIBEROAdapter) -> str:
    return str(getattr(adapter.task, "language", "") or getattr(adapter.task, "name", "") or "")


def collect_scripted_episode(adapter: LIBEROAdapter, args: argparse.Namespace) -> dict[str, Any]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    feature_weights: np.ndarray | None = None
    language = current_language(adapter)
    prev_action = np.zeros(adapter.action_dim, dtype=float)
    initial_distance = float(adapter.task_distance())
    total_reward = 0.0
    energy = 0.0
    steps = 0
    phases = phase_targets(adapter, args)
    if not phases:
        return {"success": False, "features": features, "actions": actions, "feature_weights": feature_weights, "failure_reason": "phase targets unavailable"}
    for _, target, gripper, n_steps in phases:
        for _ in range(int(n_steps)):
            if getattr(adapter, "last_done", False) or adapter.evaluate_success():
                break
            feature, weights = visual_language_feature(adapter, args, prev_action, steps, args.eval_steps)
            feature_weights = weights
            action = scripted_action(adapter, target, gripper, args.servo_gain)
            features.append(feature)
            actions.append(action)
            try:
                _, reward, done, truncated, _ = adapter.step(action)
            except ValueError as exc:
                return {
                    "success": bool(adapter.evaluate_success()),
                    "features": features,
                    "actions": actions,
                    "feature_weights": feature_weights,
                    "failure_reason": str(exc),
                }
            total_reward += float(reward)
            energy += float(np.sum(action * action))
            steps += 1
            prev_action = action
            if done or truncated:
                break
        if getattr(adapter, "last_done", False) or adapter.evaluate_success():
            break
    final_distance = float(adapter.task_distance())
    return {
        "success": bool(adapter.evaluate_success()),
        "features": features,
        "actions": actions,
        "feature_weights": feature_weights,
        "failure_reason": None,
        "total_reward": float(total_reward),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "progress": float(initial_distance - final_distance),
        "energy": float(energy),
        "steps": int(steps),
        "language": language,
    }


def run_bc_episode(
    adapter: LIBEROAdapter,
    args: argparse.Namespace,
    mean: np.ndarray,
    scale: np.ndarray,
    feature_weights: np.ndarray,
    predict_action: Callable[[np.ndarray, str], np.ndarray],
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    prev_action = np.zeros(adapter.action_dim, dtype=float)
    initial_distance = float(adapter.task_distance())
    total_reward = 0.0
    energy = 0.0
    steps = 0
    failure_reason = None
    language = current_language(adapter)
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    feature_weight_trace: np.ndarray | None = None
    for _ in range(int(args.eval_steps)):
        if getattr(adapter, "last_done", False) or adapter.evaluate_success():
            break
        feature, weights = visual_language_feature(adapter, args, prev_action, steps, args.eval_steps)
        if weights.shape != feature_weights.shape:
            failure_reason = "feature weight shape changed"
            break
        z = ((feature - mean) / scale) * feature_weights
        action = predict_action(z, language)
        action = np.clip(action, adapter.action_low, adapter.action_high)
        if record_trace:
            features.append(feature)
            actions.append(action)
            feature_weight_trace = weights
        try:
            _, reward, done, truncated, _ = adapter.step(action)
        except ValueError as exc:
            failure_reason = str(exc)
            break
        total_reward += float(reward)
        energy += float(np.sum(action * action))
        steps += 1
        prev_action = action
        if done or truncated:
            break
    final_distance = float(adapter.task_distance())
    return {
        "success": bool(adapter.evaluate_success()),
        "total_reward": float(total_reward),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "progress": float(initial_distance - final_distance),
        "energy": float(energy),
        "steps": int(steps),
        "failure_reason": failure_reason,
        "language": language,
        "features": features,
        "actions": actions,
        "feature_weights": feature_weight_trace,
    }


def write_report(summary: dict[str, Any], path: Path | None = None) -> None:
    ci = (summary.get("confidence_intervals") or {}).get("eval_success_rate") or {}
    policy = summary.get("policy") if isinstance(summary.get("policy"), dict) else {}
    policy_type = str(policy.get("type") or "unavailable")
    is_neural = bool(policy.get("is_neural"))
    distilled = bool(policy.get("distilled_from_teacher"))
    model_class_text = (
        "a distilled tiny neural RGB/proprio/language behavior-cloned policy"
        if distilled
        else
        "a tiny neural RGB/proprio/language behavior-cloned policy"
        if is_neural
        else "a low-dimensional RGB/proprio/language behavior-cloned kNN policy"
    )
    boundary = (
        "- This is a distilled tiny neural visual-language-action style smoke; the teacher is used only for training labels. It is not VLA-scale pretraining or broad LIBERO policy evidence."
        if distilled
        else
        "- This is a tiny neural visual-language-action style smoke, not VLA-scale pretraining or broad LIBERO policy evidence."
        if is_neural
        else "- This uses RGB observations and task language, but it is still a lightweight feature-kNN behavior clone, not a modern vision-language policy."
    )
    language_filter = (
        "- It encodes task language as hashed text features and does not restrict evaluation by task ID or simulator object state."
        if is_neural
        else "- It uses task language to restrict nearest-neighbor candidates to demonstrations with the same instruction."
    )
    task_count = len(summary.get("tasks") or [])
    output_tag = str(summary.get("output_tag") or "")
    scope_line = (
        f"- This tagged smoke artifact evaluates `{task_count}` task(s); it is auxiliary model-class evidence and does not replace the canonical all-task LIBERO artifact."
        if output_tag
        else "- The default artifact evaluates all ten LIBERO Object tasks, not all LIBERO suites."
    )
    lines = [
        "# LIBERO Visual-Language BC Policy Report",
        "",
        f"This optional artifact evaluates {model_class_text} on LIBERO Object tasks. The policy receives rendered `agentview` and wrist RGB features, robot proprioception, task language, previous action memory, and a finite-horizon step clock.",
        "",
        "## Summary",
        "",
        f"- Available: `{summary.get('available')}`.",
        f"- Verified: `{summary.get('verified')}`.",
        f"- Policy type: `{policy_type}`.",
        f"- Neural/action-head parameters: `{policy.get('vla_scale_parameters')}`.",
        f"- Train action examples: `{summary.get('train_examples')}`.",
        f"- Eval episodes: `{summary.get('eval_episodes')}`.",
        f"- Eval successes: `{summary.get('eval_successes')}`.",
        f"- Eval success rate: `{ci.get('mean')}` with bootstrap CI [`{ci.get('lo')}`, `{ci.get('hi')}`].",
        "",
        "## Claim Boundary",
        "",
        boundary,
        "- It does not use simulator object state, scripted phase labels, task IDs, or commanded target points at evaluation time.",
        language_filter,
        scope_line,
        "- Training labels include closed-loop retrieval-teacher rollouts, but evaluation uses only the saved neural action head."
        if distilled
        else "- Demonstrations come from the hand-coded object-tuned scripted controller.",
    ]
    (path or (REPORTS / "libero_visual_language_bc_policy_report.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def unavailable_summary(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "benchmark_libero_visual_language_bc_policy",
        "available": False,
        "attempted": True,
        "verified": False,
        "reason": reason,
        "tasks": parse_task_ids(args.tasks, args.suite),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", nargs="+", default=ALL_LIBERO_OBJECT_TASKS)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[200, 201, 202, 203, 204])
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--camera-width", type=int, default=64)
    parser.add_argument("--camera-height", type=int, default=64)
    parser.add_argument("--offscreen-renderer", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=280)
    parser.add_argument("--policy-backend", choices=["knn", "tiny_neural_vla", "distilled_neural_vla"], default="knn")
    parser.add_argument(
        "--distill-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds for closed-loop teacher rollouts used only to train distilled_neural_vla.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional artifact tag. Empty preserves the canonical benchmark_libero_visual_language_bc_policy outputs.",
    )
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-temperature", type=float, default=0.05)
    parser.add_argument("--neural-hidden-dim", type=int, default=128)
    parser.add_argument("--neural-epochs", type=int, default=60)
    parser.add_argument("--neural-batch-size", type=int, default=512)
    parser.add_argument("--neural-lr", type=float, default=3e-4)
    parser.add_argument("--neural-weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--image-grid", type=int, default=8)
    parser.add_argument("--language-hash-dim", type=int, default=64)
    parser.add_argument("--image-weight", type=float, default=0.45)
    parser.add_argument("--language-weight", type=float, default=1.25)
    parser.add_argument("--proprio-weight", type=float, default=1.0)
    parser.add_argument("--prev-action-weight", type=float, default=0.75)
    parser.add_argument("--clock-weight", type=float, default=2.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    parser.add_argument("--min-success-ci-lo", type=float, default=0.6)
    parser.add_argument("--fail-on-low-success", action="store_true")
    parser.add_argument("--safe-lift", type=float, default=0.25)
    parser.add_argument("--approach-z-offset", type=float, default=0.055)
    parser.add_argument("--grasp-z-offset", type=float, default=0.035)
    parser.add_argument("--place-z-offset", type=float, default=0.11)
    parser.add_argument("--grasp-offset-x", type=float, default=0.0)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--servo-gain", type=float, default=8.0)
    parser.add_argument("--above-steps", type=int, default=35)
    parser.add_argument("--descend-steps", type=int, default=25)
    parser.add_argument("--close-steps", type=int, default=35)
    parser.add_argument("--lift-steps", type=int, default=45)
    parser.add_argument("--move-steps", type=int, default=60)
    parser.add_argument("--place-steps", type=int, default=25)
    parser.add_argument("--open-steps", type=int, default=35)
    parser.add_argument("--retreat-steps", type=int, default=20)
    parser.add_argument("--object-grasp-tuning", dest="object_grasp_tuning", action="store_true", default=True)
    parser.add_argument("--disable-object-grasp-tuning", dest="object_grasp_tuning", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dirs()
    paths = artifact_layout(args.output_tag)
    ok, reason = is_libero_available()
    if not ok:
        summary = unavailable_summary(reason, args)
        summary["output_tag"] = paths["tag"]
        summary["artifact_paths"] = {
            "json": paths["json_rel"],
            "episodes_csv": paths["episodes_csv_rel"],
            "report": paths["report_rel"],
        }
        write_json(paths["json"], summary)  # type: ignore[arg-type]
        write_report(summary, paths["report"])  # type: ignore[arg-type]
        print(reason)
        return

    task_ids = parse_task_ids(args.tasks, args.suite)
    rows: list[dict[str, Any]] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    language_parts: list[np.ndarray] = []
    feature_weights: np.ndarray | None = None
    adapter: LIBEROAdapter | None = None
    try:
        adapter = LIBEROAdapter(
            suite=args.suite,
            task_index=task_index(task_ids[0]),
            horizon=args.horizon,
            controller=args.controller,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            use_camera_obs=True,
            has_offscreen_renderer=bool(args.offscreen_renderer),
        )
        for tid in task_ids:
            for seed in args.train_seeds:
                adapter.reset(int(seed), task_id=tid)
                out = collect_scripted_episode(adapter, args)
                if out["features"] and out["actions"]:
                    x_parts.append(np.vstack(out["features"]))
                    y_parts.append(np.vstack(out["actions"]))
                    language_parts.append(np.full(len(out["features"]), str(out.get("language", "")), dtype=object))
                    if feature_weights is None:
                        feature_weights = np.asarray(out["feature_weights"], dtype=float)
                rows.append(
                    {
                        "split": "train_scripted_visual_language",
                        "task_id": tid,
                        "task_name": str(getattr(adapter.task, "name", tid)),
                        "seed": int(seed),
                        **{k: out.get(k) for k in ["success", "total_reward", "initial_distance", "final_distance", "progress", "energy", "steps", "failure_reason"]},
                    }
                )
                print(f"train {tid} seed={seed} success={out.get('success')} examples={len(out['features'])}", flush=True)
        if not x_parts or feature_weights is None:
            raise RuntimeError("no train examples collected")
        x = np.vstack(x_parts)
        y = np.vstack(y_parts)
        train_languages = np.concatenate(language_parts)
        mean, scale = standardize_fit(x)
        x_z = ((x - mean) / scale) * feature_weights.reshape(1, -1)
        language_to_indices = {
            str(language): np.flatnonzero(train_languages == language)
            for language in sorted({str(v) for v in train_languages.tolist()})
        }
        model_path = paths["model"]  # type: ignore[assignment]
        distill_seeds = [int(s) for s in (args.distill_seeds if args.distill_seeds is not None else args.train_seeds)]
        distill_examples = 0
        distill_successes: list[float] = []
        if args.policy_backend == "distilled_neural_vla":

            def teacher_predict(z: np.ndarray, language: str) -> np.ndarray:
                return knn_predict(
                    x_z,
                    y,
                    z,
                    k=args.knn_k,
                    temperature=args.knn_temperature,
                    candidate_indices=language_to_indices.get(language),
                )

            teacher_x_parts: list[np.ndarray] = []
            teacher_y_parts: list[np.ndarray] = []
            teacher_language_parts: list[np.ndarray] = []
            for tid in task_ids:
                for seed in distill_seeds:
                    adapter.reset(int(seed), task_id=tid)
                    out = run_bc_episode(adapter, args, mean, scale, feature_weights, teacher_predict, record_trace=True)
                    if out["features"] and out["actions"]:
                        teacher_x = np.vstack(out["features"])
                        teacher_y = np.vstack(out["actions"])
                        teacher_x_parts.append(teacher_x)
                        teacher_y_parts.append(teacher_y)
                        teacher_language_parts.append(np.full(len(teacher_x), str(out.get("language", "")), dtype=object))
                        distill_examples += int(len(teacher_x))
                    distill_successes.append(float(out.get("success", False)))
                    rows.append(
                        {
                            "split": "train_distilled_teacher_visual_language",
                            "task_id": tid,
                            "task_name": str(getattr(adapter.task, "name", tid)),
                            "seed": int(seed),
                            **{k: out.get(k) for k in ["success", "total_reward", "initial_distance", "final_distance", "progress", "energy", "steps", "failure_reason"]},
                        }
                    )
                    print(f"distill {tid} seed={seed} success={out.get('success')} examples={len(out['features'])}", flush=True)
            if not teacher_x_parts:
                raise RuntimeError("no distillation examples collected")
            x = np.vstack([x, *teacher_x_parts])
            y = np.vstack([y, *teacher_y_parts])
            train_languages = np.concatenate([train_languages, *teacher_language_parts])
            mean, scale = standardize_fit(x)
            x_z = ((x - mean) / scale) * feature_weights.reshape(1, -1)
            language_to_indices = {
                str(language): np.flatnonzero(train_languages == language)
                for language in sorted({str(v) for v in train_languages.tolist()})
            }

        neural_arrays: dict[str, np.ndarray] = {}
        neural_losses: list[float] = []
        neural_model: Any | None = None
        if args.policy_backend in {"tiny_neural_vla", "distilled_neural_vla"}:
            neural_model, neural_arrays, neural_losses = train_tiny_neural_vla(x_z, y, args)
            y_mean = neural_arrays["y_mean"].astype(float)
            y_scale = neural_arrays["y_scale"].astype(float)

            def predict_action(z: np.ndarray, _: str) -> np.ndarray:
                return neural_predict(neural_model, z, y_mean, y_scale)

        else:

            def predict_action(z: np.ndarray, language: str) -> np.ndarray:
                return knn_predict(
                    x_z,
                    y,
                    z,
                    k=args.knn_k,
                    temperature=args.knn_temperature,
                    candidate_indices=language_to_indices.get(language),
                )

        np.savez(
            model_path,
            x_train=x.astype(np.float32),
            y_train=y.astype(np.float32),
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            feature_weights=feature_weights.astype(np.float32),
            train_languages=train_languages,
            task_ids=np.asarray(task_ids, dtype=object),
            train_seeds=np.asarray(args.train_seeds, dtype=int),
            distill_seeds=np.asarray(distill_seeds if args.policy_backend == "distilled_neural_vla" else [], dtype=int),
            eval_seeds=np.asarray(args.eval_seeds, dtype=int),
            policy_backend=np.asarray(args.policy_backend),
            **neural_arrays,
        )
        for tid in task_ids:
            for seed in args.eval_seeds:
                adapter.reset(int(seed), task_id=tid)
                out = run_bc_episode(adapter, args, mean, scale, feature_weights, predict_action)
                rows.append(
                    {
                        "split": "eval_visual_language_bc",
                        "task_id": tid,
                        "task_name": str(getattr(adapter.task, "name", tid)),
                        "seed": int(seed),
                        **{k: out.get(k) for k in ["success", "total_reward", "initial_distance", "final_distance", "progress", "energy", "steps", "failure_reason"]},
                    }
                )
                print(f"eval {tid} seed={seed} success={out.get('success')} progress={out.get('progress')}", flush=True)
    except (LIBEROUnavailableError, RuntimeError, ValueError) as exc:
        summary = unavailable_summary(f"{type(exc).__name__}: {exc}", args)
        summary["rows"] = rows
        summary["output_tag"] = paths["tag"]
        summary["artifact_paths"] = {
            "json": paths["json_rel"],
            "episodes_csv": paths["episodes_csv_rel"],
            "report": paths["report_rel"],
        }
        write_json(paths["json"], summary)  # type: ignore[arg-type]
        write_csv(paths["episodes_csv"], rows)  # type: ignore[arg-type]
        write_report(summary, paths["report"])  # type: ignore[arg-type]
        if args.fail_on_low_success:
            raise
        return
    finally:
        if adapter is not None:
            adapter.close()

    eval_rows = [r for r in rows if r.get("split") == "eval_visual_language_bc"]
    successes = [float(r.get("success", False)) for r in eval_rows]
    ci = bootstrap_ci(successes, seed=args.seed, n_boot=args.bootstrap_samples)
    train_successes = [float(r.get("success", False)) for r in rows if r.get("split") == "train_scripted_visual_language"]
    verified = (
        len(eval_rows) >= len(task_ids) * len(args.eval_seeds)
        and (ci.get("mean") or 0.0) >= float(args.min_success_rate)
        and (ci.get("lo") or 0.0) >= float(args.min_success_ci_lo)
        and (
            args.policy_backend not in {"tiny_neural_vla", "distilled_neural_vla"}
            or int(sum(successes)) > 0
        )
    )
    is_neural_backend = args.policy_backend in {"tiny_neural_vla", "distilled_neural_vla"}
    policy_type = (
        "distilled_tiny_neural_vla_behavior_cloning"
        if args.policy_backend == "distilled_neural_vla"
        else "tiny_neural_vla_behavior_cloning"
        if args.policy_backend == "tiny_neural_vla"
        else "rgb_proprio_language_knn_behavior_cloning"
    )
    summary = {
        "experiment": "benchmark_libero_visual_language_bc_policy",
        "available": True,
        "attempted": True,
        "verified": bool(verified),
        "output_tag": paths["tag"],
        "tasks": task_ids,
        "train_seeds": [int(s) for s in args.train_seeds],
        "eval_seeds": [int(s) for s in args.eval_seeds],
        "train_episodes": int(len(train_successes)),
        "train_successes": int(sum(train_successes)),
        "train_examples": int(len(x)),
        "distill_episodes": int(len(distill_successes)),
        "distill_successes": int(sum(distill_successes)),
        "distill_examples": int(distill_examples),
        "eval_episodes": int(len(eval_rows)),
        "eval_successes": int(sum(successes)),
        "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
        "confidence_intervals": {"eval_success_rate": ci},
        "policy": {
            "type": policy_type,
            "is_neural": is_neural_backend,
            "is_short_neural_smoke": bool(paths["tag"]) and is_neural_backend,
            "pretrained_vla": False,
            "vla_scale_parameters": neural_parameter_count(neural_arrays)
            if is_neural_backend
            else None,
            "uses_rgb": True,
            "uses_language": True,
            "uses_robot_proprio": True,
            "uses_simulator_object_state": False,
            "uses_task_id": False,
            "uses_phase_index": False,
            "uses_target_point_command": False,
            "uses_previous_action": True,
            "uses_step_clock": True,
            "uses_language_candidate_filter": args.policy_backend == "knn",
            "distilled_from_teacher": args.policy_backend == "distilled_neural_vla",
            "teacher_used_only_for_training": args.policy_backend == "distilled_neural_vla",
            "image_grid": int(args.image_grid),
            "language_hash_dim": int(args.language_hash_dim),
            "neural_hidden_dim": int(args.neural_hidden_dim) if is_neural_backend else None,
            "neural_epochs": int(args.neural_epochs) if is_neural_backend else None,
            "neural_final_train_loss": float(neural_losses[-1]) if neural_losses else None,
            "knn_k": int(args.knn_k) if args.policy_backend == "knn" else None,
            "knn_temperature": float(args.knn_temperature) if args.policy_backend == "knn" else None,
        },
        "object_grasp_tuning": bool(getattr(args, "object_grasp_tuning", True)),
        "model_path": str(model_path.relative_to(ROOT)),
        "artifact_paths": {
            "json": paths["json_rel"],
            "episodes_csv": paths["episodes_csv_rel"],
            "report": paths["report_rel"],
        },
        "note": (
            "Distilled tiny neural RGB/proprio/language time-conditioned BC policy; the retrieval teacher is used only to generate training labels, not at evaluation time. This is neural model-class smoke evidence, not VLA-scale or pretrained evidence."
            if args.policy_backend == "distilled_neural_vla"
            else
            "Tiny neural RGB/proprio/language time-conditioned BC policy without simulator object state, task IDs, phase labels, target-point commands, or language-candidate retrieval; a VLA-style smoke, not VLA-scale evidence."
            if args.policy_backend == "tiny_neural_vla"
            else "RGB/proprio/language time-conditioned BC policy without simulator object state, task IDs, phase labels, or target-point commands; not full LIBERO or modern VLA evidence."
        ),
    }
    write_json(paths["json"], summary)  # type: ignore[arg-type]
    write_csv(paths["episodes_csv"], rows)  # type: ignore[arg-type]
    write_report(summary, paths["report"])  # type: ignore[arg-type]
    print(json.dumps(sanitize(summary), indent=2))
    if args.fail_on_low_success and not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
