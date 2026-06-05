from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from .evaluation import results_dir, write_json
from .modern_vla_execution_probe import (
    MODEL_ID,
    default_libero_config,
    default_libero_python,
    default_libero_source,
)


CHILD_CODE = r"""
import json
import os
import tempfile
import time
import traceback

started = time.time()

def emit(payload):
    payload["elapsed_seconds"] = float(time.time() - started)
    print(json.dumps(payload))

try:
    import numpy as np
    import torch
    import lerobot.configs.policies as policies
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    original_load_state_dict = torch.nn.Module.load_state_dict

    def assign_load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return original_load_state_dict(self, state_dict, strict=strict, assign=True)
        except TypeError:
            return original_load_state_dict(self, state_dict, strict=strict)

    torch.nn.Module.load_state_dict = assign_load_state_dict

    class ReopenableNamedTemporaryFile:
        def __init__(self, mode="w+", *args, **kwargs):
            fd, self.name = tempfile.mkstemp()
            os.close(fd)
            self._file = open(self.name, mode, encoding=kwargs.get("encoding") or None)

        def __enter__(self):
            return self._file

        def __exit__(self, exc_type, exc, tb):
            try:
                self._file.close()
            finally:
                try:
                    os.unlink(self.name)
                except OSError:
                    pass
            return False

    policies.tempfile.NamedTemporaryFile = ReopenableNamedTemporaryFile

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from wam_inference_value.benchmarks.libero_adapter import LIBEROAdapter

    suite = os.environ.get("WAM_LIBERO_SUITE", "libero_object")
    task_index = int(os.environ.get("WAM_LIBERO_TASK_INDEX", "0"))
    seeds = [int(x) for x in os.environ.get("WAM_LIBERO_EVAL_SEEDS", "300,301,302,303,304").split(",") if x.strip()]
    horizon = int(os.environ.get("WAM_LIBERO_HORIZON", "10"))
    max_steps = int(os.environ.get("WAM_LIBERO_MAX_STEPS", str(horizon)))
    model_id = os.environ.get("WAM_VLA_MODEL_ID", "HuggingFaceVLA/smolvla_libero")

    adapter = LIBEROAdapter(
        suite=suite,
        task_index=task_index,
        horizon=horizon,
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    adapter.reset(seeds[0] if seeds else 0)
    first_raw = adapter._raw_obs()
    task = str(getattr(adapter.task, "language", "") or getattr(adapter.task, "name", "") or adapter.task_name)

    def load_processor_stats(policy, repo_id):
        loaded = {"preprocessor": False, "postprocessor": False}
        pre_path = hf_hub_download(
            repo_id=repo_id,
            filename="policy_preprocessor_step_5_normalizer_processor.safetensors",
        )
        pre = load_file(pre_path, device="cpu")
        if "observation.state.mean" in pre and "observation.state.std" in pre:
            policy.normalize_inputs.load_state_dict(
                {
                    "buffer_observation_state.mean": pre["observation.state.mean"],
                    "buffer_observation_state.std": pre["observation.state.std"],
                },
                strict=False,
            )
        if "action.mean" in pre and "action.std" in pre:
            policy.normalize_targets.load_state_dict(
                {"buffer_action.mean": pre["action.mean"], "buffer_action.std": pre["action.std"]},
                strict=False,
            )
        loaded["preprocessor"] = True
        for filename in (
            "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ):
            try:
                post_path = hf_hub_download(repo_id=repo_id, filename=filename)
                post = load_file(post_path, device="cpu")
                if "action.mean" in post and "action.std" in post:
                    policy.unnormalize_outputs.load_state_dict(
                        {"buffer_action.mean": post["action.mean"], "buffer_action.std": post["action.std"]},
                        strict=False,
                    )
                    loaded["postprocessor"] = True
                    loaded["postprocessor_file"] = filename
                    break
            except Exception:
                continue
        return loaded

    try:
        policy = SmolVLAPolicy.from_pretrained(model_id)
        processor_stats = load_processor_stats(policy, model_id)
        parameter_count = int(sum(p.numel() for p in policy.parameters()))
    except Exception as exc:
        adapter.close()
        emit(
            {
                "ok": False,
                "verified": False,
                "failure_stage": "policy_load",
                "policy_loaded": False,
                "processor_stats_loaded": False,
                "action_selected": False,
                "libero_steps_succeeded": False,
                "heldout_libero_policy_eval": False,
                "model_id": model_id,
                "task": task,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "trace_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        raise SystemExit(0)

    input_features = getattr(policy.config, "input_features", {}) or {}
    image_sources = ["agentview_image", "robot0_eye_in_hand_image", "agentview_image"]

    def image_tensor(raw, key, shape):
        arr = np.asarray(raw.get(key), dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            arr = np.zeros((adapter.camera_height, adapter.camera_width, 3), dtype=np.float32)
        arr = arr[..., :3] / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if len(shape) == 3 and tuple(tensor.shape[1:]) != tuple(shape):
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=tuple(int(v) for v in shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        return tensor

    def feature_shape(feature, fallback):
        return tuple(int(v) for v in getattr(feature, "shape", fallback))

    def build_batch(raw):
        proprio = np.asarray(raw.get("robot0_proprio-state", np.zeros(0)), dtype=np.float32).reshape(-1)
        image_index = 0
        batch = {"task": task}
        for key, feature in input_features.items():
            shape = feature_shape(feature, (3, adapter.camera_height, adapter.camera_width))
            if key.startswith("observation.images"):
                source_key = image_sources[min(image_index, len(image_sources) - 1)]
                batch[key] = image_tensor(raw, source_key, shape)
                image_index += 1
            elif key == "observation.state":
                state_dim = int(shape[0]) if shape else 0
                state = proprio.copy()
                if state.size < state_dim:
                    state = np.pad(state, (0, state_dim - state.size))
                batch[key] = torch.from_numpy(state[:state_dim]).reshape(1, state_dim)
        if "observation.state" not in batch:
            state = proprio if proprio.size else np.zeros(8, dtype=np.float32)
            state_dim = min(8, state.size)
            batch["observation.state"] = torch.from_numpy(state[:state_dim]).reshape(1, state_dim)
        return batch

    def map_action(raw_action):
        action = np.zeros(adapter.action_dim, dtype=float)
        n = min(adapter.action_dim, raw_action.size)
        action[:n] = raw_action[:n]
        if adapter.action_dim > raw_action.size and raw_action.size:
            action[-1] = raw_action[-1]
        return np.clip(action, adapter.action_low, adapter.action_high)

    episodes = []
    raw_action_head = None
    mapped_action_head = None
    action_selected_any = False
    try:
        for episode_id, seed in enumerate(seeds):
            adapter.reset(seed)
            if hasattr(policy, "reset"):
                policy.reset()
            total_reward = 0.0
            energy = 0.0
            steps = 0
            done = False
            truncated = False
            step_error = None
            initial_distance = float(adapter.task_distance())
            for _ in range(max_steps):
                raw = first_raw if episode_id == 0 and steps == 0 else adapter._raw_obs()
                batch = build_batch(raw)
                with torch.no_grad():
                    raw_action = policy.select_action(batch).detach().cpu().numpy().reshape(-1)
                action_selected_any = True
                action = map_action(raw_action)
                if raw_action_head is None:
                    raw_action_head = raw_action[:8].astype(float).tolist()
                    mapped_action_head = action.astype(float).tolist()
                _, reward, done, truncated, _ = adapter.step(action)
                total_reward += float(reward)
                energy += float(np.sum(action * action))
                steps += 1
                if bool(done) or bool(truncated) or bool(adapter.evaluate_success()):
                    break
            final_distance = float(adapter.task_distance())
            success = bool(adapter.evaluate_success())
            episodes.append(
                {
                    "episode_id": int(episode_id),
                    "seed": int(seed),
                    "steps": int(steps),
                    "success": success,
                    "success_int": int(success),
                    "initial_distance": initial_distance,
                    "final_distance": final_distance,
                    "progress": float(initial_distance - final_distance),
                    "total_reward": float(total_reward),
                    "energy": float(energy),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "step_error": step_error,
                }
            )
    except Exception as exc:
        adapter.close()
        emit(
            {
                "ok": False,
                "verified": False,
                "failure_stage": "episode_execution",
                "policy_loaded": True,
                "parameter_count": parameter_count,
                "processor_stats_loaded": processor_stats,
                "action_selected": action_selected_any,
                "libero_steps_succeeded": bool(episodes),
                "heldout_libero_policy_eval": False,
                "model_id": model_id,
                "task": task,
                "suite": suite,
                "task_index": task_index,
                "seeds": seeds,
                "episodes": episodes,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "trace_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        raise SystemExit(0)

    adapter.close()
    successes = int(sum(int(row["success"]) for row in episodes))
    emit(
        {
            "ok": True,
            "verified": bool(episodes) and action_selected_any,
            "failure_stage": None,
            "policy_loaded": True,
            "parameter_count": parameter_count,
            "processor_stats_loaded": processor_stats,
            "action_selected": action_selected_any,
            "libero_steps_succeeded": bool(episodes),
            "heldout_libero_policy_eval": bool(episodes) and action_selected_any,
            "model_id": model_id,
            "task": task,
            "suite": suite,
            "task_index": task_index,
            "horizon": horizon,
            "max_steps": max_steps,
            "eval_seeds": seeds,
            "eval_episodes": len(episodes),
            "eval_successes": successes,
            "eval_success_rate": float(successes / max(len(episodes), 1)),
            "libero_action_dim": int(adapter.action_dim),
            "input_feature_keys": list(input_features.keys()),
            "raw_action_head": raw_action_head,
            "mapped_action": mapped_action_head,
            "episodes": episodes,
        }
    )
except Exception as exc:
    emit(
        {
            "ok": False,
            "verified": False,
            "failure_stage": "setup",
            "policy_loaded": False,
            "action_selected": False,
            "libero_steps_succeeded": False,
            "heldout_libero_policy_eval": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "trace_tail": traceback.format_exc().splitlines()[-16:],
        }
    )
"""


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> dict[str, Any]:
    if n <= 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    phat = float(successes) / float(n)
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return {
        "n": int(n),
        "mean": phat,
        "lo": max(0.0, float((centre - margin) / denom)),
        "hi": min(1.0, float((centre + margin) / denom)),
        "method": "wilson",
    }


def _run_child(
    root: Path,
    *,
    python_path: Path,
    libero_source: Path,
    libero_config: Path,
    model_id: str,
    suite: str,
    task_index: int,
    seeds: list[int],
    horizon: int,
    max_steps: int,
    timeout_s: int,
) -> dict[str, Any]:
    if not python_path.exists():
        return {
            "returncode": None,
            "stdout_tail": [],
            "stderr_tail": [],
            "child": {
                "ok": False,
                "verified": False,
                "failure_stage": "python_missing",
                "error_type": "PythonMissing",
                "error": str(python_path),
                "heldout_libero_policy_eval": False,
            },
        }
    env = os.environ.copy()
    pieces = [str(libero_source), str(root / "src"), str(root / "experiments")]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([*pieces, existing] if existing else pieces)
    env["LIBERO_CONFIG_PATH"] = str(libero_config)
    env["WAM_VLA_MODEL_ID"] = model_id
    env["WAM_LIBERO_SUITE"] = suite
    env["WAM_LIBERO_TASK_INDEX"] = str(int(task_index))
    env["WAM_LIBERO_EVAL_SEEDS"] = ",".join(str(int(seed)) for seed in seeds)
    env["WAM_LIBERO_HORIZON"] = str(int(horizon))
    env["WAM_LIBERO_MAX_STEPS"] = str(int(max_steps))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        proc = subprocess.run(
            [str(python_path), "-c", CHILD_CODE],
            cwd=str(root),
            env=env,
            text=True,
            capture_output=True,
            timeout=int(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "returncode": None,
            "stdout_tail": stdout.splitlines()[-20:],
            "stderr_tail": stderr.splitlines()[-30:],
            "child": {
                "ok": False,
                "verified": False,
                "failure_stage": "timeout",
                "error_type": "TimeoutExpired",
                "error": f"timed out after {timeout_s}s",
                "heldout_libero_policy_eval": False,
            },
        }
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        child_payload = json.loads(stdout_lines[-1]) if stdout_lines else {}
    except json.JSONDecodeError:
        crash_label = "WindowsAccessViolation" if proc.returncode == 3221225477 else f"ProcessReturnCode{proc.returncode}"
        child_payload = {
            "ok": False,
            "verified": False,
            "failure_stage": "process_crash" if proc.returncode else "non_json_output",
            "error_type": crash_label if proc.returncode else "NonJsonOutput",
            "error": proc.stdout[-2000:],
            "heldout_libero_policy_eval": False,
            "returncode": proc.returncode,
        }
    return {
        "returncode": proc.returncode,
        "stdout_tail": stdout_lines[-20:],
        "stderr_tail": proc.stderr.splitlines()[-30:],
        "child": child_payload if isinstance(child_payload, dict) else {},
    }


def _write_episode_table(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "episode_id",
        "seed",
        "steps",
        "success",
        "initial_distance",
        "final_distance",
        "progress",
        "total_reward",
        "energy",
        "done",
        "truncated",
        "step_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in episodes:
            writer.writerow({key: row.get(key) for key in columns})


def _existing_compatible_episodes(
    path: Path,
    *,
    model_id: str,
    suite: str,
    task_index: int,
    horizon: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    def as_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    compatible = (
        payload.get("model_id") == model_id
        and payload.get("suite") == suite
        and as_int(payload.get("task_index")) == int(task_index)
        and as_int(payload.get("horizon")) == int(horizon)
        and as_int(payload.get("max_steps")) == int(max_steps)
    )
    episodes = payload.get("episodes")
    if not compatible or not isinstance(episodes, list):
        return []
    return [dict(row) for row in episodes if isinstance(row, dict)]


def _merge_episodes(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for row in [*existing, *new]:
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            continue
        by_seed[seed] = dict(row)
    merged: list[dict[str, Any]] = []
    for episode_id, seed in enumerate(sorted(by_seed)):
        row = dict(by_seed[seed])
        row["episode_id"] = int(episode_id)
        merged.append(row)
    return merged


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _compact_attempt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "verified": bool(payload.get("verified")),
        "heldout_libero_policy_eval": bool(payload.get("heldout_libero_policy_eval")),
        "suite": payload.get("suite"),
        "task_index": payload.get("task_index"),
        "horizon": payload.get("horizon"),
        "max_steps": payload.get("max_steps"),
        "requested_eval_seeds": payload.get("requested_eval_seeds"),
        "eval_episodes": payload.get("eval_episodes"),
        "eval_successes": payload.get("eval_successes"),
        "failure_stage": payload.get("failure_stage"),
        "error_type": payload.get("error_type"),
        "child_returncode": payload.get("child_returncode"),
    }


def _merge_attempt_history(*items: Any) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, list):
            candidates = item
        else:
            candidates = [item]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            summary = _compact_attempt_summary(candidate)
            if not summary:
                continue
            key = json.dumps(summary, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            history.append(summary)
    return history


def modern_vla_libero_policy_eval_markdown(payload: dict[str, Any]) -> str:
    ci = payload.get("success_ci") if isinstance(payload.get("success_ci"), dict) else {}
    lines = [
        "# Modern VLA LIBERO Heldout Policy Eval",
        "",
        f"- attempted: `{payload.get('attempted')}`",
        f"- verified evaluation ran: `{payload.get('verified')}`",
        f"- heldout sparse-success eval: `{payload.get('heldout_libero_policy_eval')}`",
        f"- model id: `{payload.get('model_id')}`",
        f"- suite/task: `{payload.get('suite')}` / `{payload.get('task_index')}`",
        f"- eval episodes: `{payload.get('eval_episodes')}`",
        f"- eval seeds: `{payload.get('eval_seeds')}`",
        f"- successes: `{payload.get('eval_successes')}`",
        f"- success rate: `{payload.get('eval_success_rate')}`",
        f"- success CI: `n={ci.get('n')}, lo={ci.get('lo')}, hi={ci.get('hi')}, method={ci.get('method')}`",
        f"- runtime Python: `{payload.get('runtime_python')}`",
        f"- parameter count: `{payload.get('parameter_count')}`",
        f"- processor stats loaded: `{payload.get('processor_stats_loaded')}`",
        f"- action selected: `{payload.get('action_selected')}`",
        f"- LIBERO steps succeeded: `{payload.get('libero_steps_succeeded')}`",
        f"- failure stage: `{payload.get('failure_stage')}`",
        f"- error type: `{payload.get('error_type')}`",
        f"- child return code: `{payload.get('child_returncode')}`",
        f"- error: `{payload.get('error')}`",
        "",
        "This artifact reports sparse-success policy execution only. It does not promote a positive modern VLA performance claim unless the claim ledger/readiness gates also find heldout nonzero success with confidence intervals.",
    ]
    return "\n".join(lines) + "\n"


def run_modern_vla_libero_policy_eval(
    root: Path,
    *,
    python_path: Path | None = None,
    libero_source: Path | None = None,
    libero_config: Path | None = None,
    output_results_dir: Path | None = None,
    model_id: str = MODEL_ID,
    suite: str = "libero_object",
    task_index: int = 0,
    seeds: list[int] | None = None,
    horizon: int = 10,
    max_steps: int | None = None,
    timeout_s: int = 1800,
    append_existing: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output_results_dir = (output_results_dir or results_dir()).resolve()
    python_path = (python_path or default_libero_python(root)).resolve()
    libero_source = (libero_source or default_libero_source(root)).resolve()
    libero_config = (libero_config or default_libero_config(root)).resolve()
    seeds = [int(seed) for seed in (seeds or [300, 301, 302, 303, 304])]
    max_steps = int(horizon if max_steps is None else max_steps)
    child = _run_child(
        root,
        python_path=python_path,
        libero_source=libero_source,
        libero_config=libero_config,
        model_id=model_id,
        suite=suite,
        task_index=task_index,
        seeds=seeds,
        horizon=horizon,
        max_steps=max_steps,
        timeout_s=timeout_s,
    )
    child_payload = child.get("child") if isinstance(child.get("child"), dict) else {}
    episodes = child_payload.get("episodes") if isinstance(child_payload.get("episodes"), list) else []
    result_path = output_results_dir / "modern_vla_libero_policy_eval.json"
    last_attempt_path = output_results_dir / "modern_vla_libero_policy_eval_last_attempt.json"
    previous_payload = _load_payload(result_path)
    previous_last_attempt = _load_payload(last_attempt_path)
    existing_episodes = (
        _existing_compatible_episodes(
            result_path,
            model_id=model_id,
            suite=suite,
            task_index=task_index,
            horizon=horizon,
            max_steps=max_steps,
        )
        if append_existing
        else []
    )
    new_child_episodes = child_payload.get("episodes") if isinstance(child_payload.get("episodes"), list) else []
    episodes = _merge_episodes(existing_episodes, new_child_episodes)
    eval_episodes = int(len(episodes))
    eval_successes = int(sum(int(bool(row.get("success"))) for row in episodes))
    success_ci = wilson_ci(eval_successes, eval_episodes)
    verified = (
        child.get("returncode") == 0
        and child_payload.get("verified") is True
        and child_payload.get("heldout_libero_policy_eval") is True
        and child_payload.get("action_selected") is True
        and eval_episodes > 0
    )
    table_path = output_results_dir / "tables" / "modern_vla_libero_policy_eval_episodes.csv"
    if episodes:
        _write_episode_table(table_path, episodes)
    payload = {
        "experiment": "modern_vla_libero_policy_eval",
        "attempted": True,
        "verified": bool(verified),
        "heldout_libero_policy_eval": bool(verified),
        "scope": "heldout sparse-success pretrained SmolVLA policy execution on LIBERO",
        "model_id": model_id,
        "runtime_python": str(python_path),
        "libero_source": str(libero_source),
        "libero_config": str(libero_config),
        "suite": suite,
        "task_index": int(task_index),
        "horizon": int(horizon),
        "max_steps": int(max_steps),
        "eval_seeds": sorted({int(row.get("seed")) for row in episodes if row.get("seed") is not None}),
        "requested_eval_seeds": seeds,
        "n_eval_seeds": len({int(row.get("seed")) for row in episodes if row.get("seed") is not None}),
        "eval_episodes": eval_episodes,
        "eval_successes": eval_successes,
        "eval_success_rate": None if eval_episodes <= 0 else float(eval_successes / eval_episodes),
        "success_ci": success_ci,
        "confidence_intervals": {"eval_success_rate": success_ci},
        "policy_loaded": child_payload.get("policy_loaded") is True,
        "parameter_count": child_payload.get("parameter_count"),
        "processor_stats_loaded": child_payload.get("processor_stats_loaded"),
        "action_selected": child_payload.get("action_selected") is True,
        "libero_steps_succeeded": child_payload.get("libero_steps_succeeded") is True,
        "input_feature_keys": child_payload.get("input_feature_keys"),
        "raw_action_head": child_payload.get("raw_action_head"),
        "mapped_action": child_payload.get("mapped_action"),
        "episodes": episodes,
        "failure_stage": child_payload.get("failure_stage"),
        "error_type": child_payload.get("error_type"),
        "error": child_payload.get("error"),
        "trace_tail": child_payload.get("trace_tail"),
        "child_returncode": child.get("returncode"),
        "stdout_tail": child.get("stdout_tail"),
        "stderr_tail": child.get("stderr_tail"),
        "child_payload": child_payload,
        "append_existing": bool(append_existing),
        "n_existing_compatible_episodes": len(existing_episodes),
        "n_new_child_episodes": len(new_child_episodes),
        "artifacts": {"episode_table": str(table_path) if episodes else None},
        "note": "A zero-success verified eval is still a real evaluation artifact, but it is not evidence of positive modern VLA policy performance.",
    }
    if append_existing and not verified and not new_child_episodes and int(previous_payload.get("eval_episodes") or 0) > 0:
        latest_attempt_summary = _compact_attempt_summary(payload)
        attempt_history = _merge_attempt_history(
            previous_payload.get("attempt_history"),
            previous_last_attempt,
            payload,
        )
        payload["previous_last_attempt_summary"] = _compact_attempt_summary(previous_last_attempt)
        payload["attempt_history"] = attempt_history
        write_json(last_attempt_path, payload)
        preserved = dict(previous_payload)
        artifacts = dict(preserved.get("artifacts") or {})
        artifacts["last_attempt"] = str(last_attempt_path)
        preserved["artifacts"] = artifacts
        preserved["latest_attempt_preserved_previous"] = True
        preserved["latest_attempt_summary"] = latest_attempt_summary
        preserved["attempt_history"] = attempt_history
        write_json(result_path, preserved)
        report_path = root / "reports" / "modern_vla_libero_policy_eval_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(modern_vla_libero_policy_eval_markdown(preserved), encoding="utf-8")
        preserved["artifacts"]["report"] = str(report_path)
        write_json(result_path, preserved)
        return preserved
    write_json(result_path, payload)
    report_path = root / "reports" / "modern_vla_libero_policy_eval_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(modern_vla_libero_policy_eval_markdown(payload), encoding="utf-8")
    payload["artifacts"]["report"] = str(report_path)
    write_json(result_path, payload)
    return payload
