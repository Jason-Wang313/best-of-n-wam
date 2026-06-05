from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .evaluation import results_dir, write_json


MODEL_ID = "HuggingFaceVLA/smolvla_libero"


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
    seed = int(os.environ.get("WAM_LIBERO_SEED", "200"))
    horizon = int(os.environ.get("WAM_LIBERO_HORIZON", "5"))
    model_id = os.environ.get("WAM_VLA_MODEL_ID", "lerobot/smolvla_base")

    adapter = LIBEROAdapter(
        suite=suite,
        task_index=task_index,
        horizon=horizon,
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    adapter.reset(seed)
    raw = adapter._raw_obs()
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
        policy_loaded = True
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
                "libero_step_succeeded": False,
                "heldout_libero_policy_eval": False,
                "model_id": model_id,
                "task": task,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "trace_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        raise SystemExit(0)

    def image_tensor(key, shape):
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

    proprio = np.asarray(raw.get("robot0_proprio-state", np.zeros(0)), dtype=np.float32).reshape(-1)
    image_sources = ["agentview_image", "robot0_eye_in_hand_image", "agentview_image"]
    image_index = 0
    batch = {"task": task}
    input_features = getattr(policy.config, "input_features", {}) or {}
    for key, feature in input_features.items():
        shape = feature_shape(feature, (3, adapter.camera_height, adapter.camera_width))
        if key.startswith("observation.images"):
            source_key = image_sources[min(image_index, len(image_sources) - 1)]
            batch[key] = image_tensor(source_key, shape)
            image_index += 1
        elif key == "observation.state":
            state_dim = int(shape[0]) if shape else 0
            state = proprio.copy()
            if state.size < state_dim:
                state = np.pad(state, (0, state_dim - state.size))
            batch[key] = torch.from_numpy(state[:state_dim]).reshape(1, state_dim)
    if "observation.state" not in batch:
        state = proprio if proprio.size else np.zeros(8, dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state[:8]).reshape(1, min(8, state.size))

    try:
        with torch.no_grad():
            raw_action = policy.select_action(batch).detach().cpu().numpy().reshape(-1)
        action_selected = True
    except Exception as exc:
        adapter.close()
        emit(
            {
                "ok": False,
                "verified": False,
                "failure_stage": "action_selection",
                "policy_loaded": policy_loaded,
                "parameter_count": parameter_count,
                "processor_stats_loaded": processor_stats,
                "action_selected": False,
                "libero_step_succeeded": False,
                "heldout_libero_policy_eval": False,
                "model_id": model_id,
                "task": task,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "trace_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        raise SystemExit(0)

    action = np.zeros(adapter.action_dim, dtype=float)
    n = min(adapter.action_dim, raw_action.size)
    action[:n] = raw_action[:n]
    if adapter.action_dim > raw_action.size and raw_action.size:
        action[-1] = raw_action[-1]
    action = np.clip(action, adapter.action_low, adapter.action_high)

    try:
        _, reward, done, truncated, _ = adapter.step(action)
        success = bool(adapter.evaluate_success())
        distance = float(adapter.task_distance())
        adapter.close()
        emit(
            {
                "ok": True,
                "verified": True,
                "failure_stage": None,
                "policy_loaded": policy_loaded,
                "parameter_count": parameter_count,
                "processor_stats_loaded": processor_stats,
                "action_selected": action_selected,
                "libero_step_succeeded": True,
                "heldout_libero_policy_eval": False,
                "model_id": model_id,
                "task": task,
                "suite": suite,
                "task_index": task_index,
                "seed": seed,
                "horizon": horizon,
                "libero_action_dim": int(adapter.action_dim),
                "smolvla_action_dim": int(raw_action.size),
                "input_feature_keys": list(input_features.keys()),
                "raw_action_head": raw_action[:8].astype(float).tolist(),
                "mapped_action": action.astype(float).tolist(),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "success_after_one_step": success,
                "distance_after_one_step": distance,
            }
        )
    except Exception as exc:
        adapter.close()
        emit(
            {
                "ok": False,
                "verified": False,
                "failure_stage": "libero_step",
                "policy_loaded": policy_loaded,
                "parameter_count": parameter_count,
                "processor_stats_loaded": processor_stats,
                "action_selected": action_selected,
                "libero_step_succeeded": False,
                "heldout_libero_policy_eval": False,
                "model_id": model_id,
                "task": task,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "trace_tail": traceback.format_exc().splitlines()[-16:],
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
            "libero_step_succeeded": False,
            "heldout_libero_policy_eval": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "trace_tail": traceback.format_exc().splitlines()[-16:],
        }
    )
"""


def default_libero_python(root: Path) -> Path:
    return root.parent / "external_benchmarks" / ".venvs" / "libero310" / "Scripts" / "python.exe"


def default_libero_source(root: Path) -> Path:
    return root.parent / "external_benchmarks" / "LIBERO"


def default_libero_config(root: Path) -> Path:
    return root.parent / "external_benchmarks" / ".libero"


def _run_child(
    root: Path,
    *,
    python_path: Path,
    libero_source: Path,
    libero_config: Path,
    model_id: str,
    suite: str,
    task_index: int,
    seed: int,
    horizon: int,
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
    env["WAM_LIBERO_SEED"] = str(int(seed))
    env["WAM_LIBERO_HORIZON"] = str(int(horizon))
    proc = subprocess.run(
        [str(python_path), "-c", CHILD_CODE],
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        timeout=int(timeout_s),
        check=False,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        child_payload = json.loads(stdout_lines[-1]) if stdout_lines else {}
    except json.JSONDecodeError:
        child_payload = {
            "ok": False,
            "verified": False,
            "failure_stage": "non_json_output",
            "error_type": "NonJsonOutput",
            "error": proc.stdout[-2000:],
            "heldout_libero_policy_eval": False,
        }
    return {
        "returncode": proc.returncode,
        "stdout_tail": stdout_lines[-20:],
        "stderr_tail": proc.stderr.splitlines()[-30:],
        "child": child_payload if isinstance(child_payload, dict) else {},
    }


def modern_vla_libero_execution_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Modern VLA LIBERO Execution Probe",
        "",
        f"- attempted: `{payload.get('attempted')}`",
        f"- verified one-step execution: `{payload.get('verified')}`",
        f"- heldout sparse-success eval: `{payload.get('heldout_libero_policy_eval')}`",
        f"- model id: `{payload.get('model_id')}`",
        f"- runtime Python: `{payload.get('runtime_python')}`",
        f"- LIBERO source: `{payload.get('libero_source')}`",
        f"- failure stage: `{payload.get('failure_stage')}`",
        f"- policy loaded: `{payload.get('policy_loaded')}`",
        f"- parameter count: `{payload.get('parameter_count')}`",
        f"- processor stats loaded: `{payload.get('processor_stats_loaded')}`",
        f"- action selected: `{payload.get('action_selected')}`",
        f"- LIBERO step succeeded: `{payload.get('libero_step_succeeded')}`",
        f"- input feature keys: `{payload.get('input_feature_keys')}`",
        f"- raw action head: `{payload.get('raw_action_head')}`",
        f"- success after one step: `{payload.get('success_after_one_step')}`",
        f"- distance after one step: `{payload.get('distance_after_one_step')}`",
        f"- error type: `{payload.get('error_type')}`",
        f"- error: `{payload.get('error')}`",
        "",
        "This is a compatibility/execution attempt only. It is not a modern VLA LIBERO performance result.",
    ]
    return "\n".join(lines) + "\n"


def run_modern_vla_libero_execution_probe(
    root: Path,
    *,
    python_path: Path | None = None,
    libero_source: Path | None = None,
    libero_config: Path | None = None,
    output_results_dir: Path | None = None,
    model_id: str = MODEL_ID,
    suite: str = "libero_object",
    task_index: int = 0,
    seed: int = 200,
    horizon: int = 5,
    timeout_s: int = 900,
) -> dict[str, Any]:
    root = root.resolve()
    output_results_dir = (output_results_dir or results_dir()).resolve()
    python_path = (python_path or default_libero_python(root)).resolve()
    libero_source = (libero_source or default_libero_source(root)).resolve()
    libero_config = (libero_config or default_libero_config(root)).resolve()
    child = _run_child(
        root,
        python_path=python_path,
        libero_source=libero_source,
        libero_config=libero_config,
        model_id=model_id,
        suite=suite,
        task_index=task_index,
        seed=seed,
        horizon=horizon,
        timeout_s=timeout_s,
    )
    child_payload = child.get("child") if isinstance(child.get("child"), dict) else {}
    verified = (
        child.get("returncode") == 0
        and child_payload.get("verified") is True
        and child_payload.get("action_selected") is True
        and child_payload.get("libero_step_succeeded") is True
    )
    payload = {
        "experiment": "modern_vla_libero_execution_probe",
        "attempted": True,
        "verified": bool(verified),
        "scope": "one-step pretrained SmolVLA compatibility probe on LIBERO; not heldout sparse-success evaluation",
        "model_id": model_id,
        "runtime_python": str(python_path),
        "libero_source": str(libero_source),
        "libero_config": str(libero_config),
        "suite": suite,
        "task_index": int(task_index),
        "seed": int(seed),
        "horizon": int(horizon),
        "policy_loaded": child_payload.get("policy_loaded") is True,
        "parameter_count": child_payload.get("parameter_count"),
        "processor_stats_loaded": child_payload.get("processor_stats_loaded"),
        "action_selected": child_payload.get("action_selected") is True,
        "libero_step_succeeded": child_payload.get("libero_step_succeeded") is True,
        "input_feature_keys": child_payload.get("input_feature_keys"),
        "raw_action_head": child_payload.get("raw_action_head"),
        "mapped_action": child_payload.get("mapped_action"),
        "success_after_one_step": child_payload.get("success_after_one_step"),
        "distance_after_one_step": child_payload.get("distance_after_one_step"),
        "heldout_libero_policy_eval": False,
        "failure_stage": child_payload.get("failure_stage"),
        "error_type": child_payload.get("error_type"),
        "error": child_payload.get("error"),
        "trace_tail": child_payload.get("trace_tail"),
        "child_returncode": child.get("returncode"),
        "stdout_tail": child.get("stdout_tail"),
        "stderr_tail": child.get("stderr_tail"),
        "child_payload": child_payload,
        "note": "No modern VLA LIBERO result is promoted unless heldout sparse-success metrics exist for this policy class.",
    }
    write_json(output_results_dir / "modern_vla_libero_execution_probe.json", payload)
    report_path = root / "reports" / "modern_vla_libero_execution_probe_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(modern_vla_libero_execution_markdown(payload), encoding="utf-8")
    payload["artifacts"] = {"report": str(report_path)}
    write_json(output_results_dir / "modern_vla_libero_execution_probe.json", payload)
    return payload
