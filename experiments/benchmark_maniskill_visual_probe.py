from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wam_inference_value.benchmarks.maniskill_adapter import is_maniskill_available
from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json


def _child_code(env_id: str, kwargs: dict[str, Any], *, do_step: bool, do_render: bool) -> str:
    return f"""
import json
import traceback
import numpy as np
import gymnasium as gym
import mani_skill
import mani_skill.envs  # noqa: F401

try:
    from mani_skill.render import shaders
    shaders.set_shader_pack(shaders.PREBUILT_SHADER_CONFIGS.get("minimal"))
except Exception:
    pass

env = None
try:
    kwargs = {repr(kwargs)}
    env = gym.make({env_id!r}, **kwargs)
    obs, info = env.reset(seed=123)
    shapes = {{}}

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                walk((prefix + "/" if prefix else "") + str(key), child)
        elif hasattr(value, "shape"):
            shapes[prefix or "obs"] = tuple(value.shape)
        else:
            arr = np.asarray(value)
            shapes[prefix or "obs"] = tuple(arr.shape)

    walk("", obs)
    if {do_step!r}:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        shapes["step_reward_shape"] = tuple(np.asarray(reward).shape)
    if {do_render!r}:
        frame = env.render()
        arr = np.asarray(frame)
        shapes["render"] = tuple(arr.shape)
        shapes["render_std"] = float(np.std(arr))
    print(json.dumps({{"ok": True, "shapes": shapes, "mani_skill_version": getattr(mani_skill, "__version__", "unknown")}}, default=str))
except Exception as exc:
    print(json.dumps({{
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "trace_tail": traceback.format_exc().splitlines()[-10:],
        "mani_skill_version": getattr(mani_skill, "__version__", "unknown"),
    }}, default=str))
finally:
    try:
        if env is not None:
            env.close()
    except Exception:
        pass
"""


def _run_attempt(attempt: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    code = _child_code(
        str(attempt["env_id"]),
        dict(attempt["kwargs"]),
        do_step=bool(attempt.get("do_step", False)),
        do_render=bool(attempt.get("do_render", False)),
    )
    try:
        child_env = os.environ.copy()
        env_overrides = {str(k): str(v) for k, v in dict(attempt.get("env") or {}).items()}
        child_env.update(env_overrides)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=child_env,
            timeout=int(timeout_s),
        )
        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        payload: dict[str, Any]
        try:
            payload = json.loads(stdout_lines[-1]) if stdout_lines else {}
        except json.JSONDecodeError:
            payload = {"ok": False, "error_type": "NonJsonOutput", "error": proc.stdout[-1200:]}
        payload.update(
            {
                "name": attempt["name"],
                "category": attempt["category"],
                "env_id": attempt["env_id"],
                "kwargs": attempt["kwargs"],
                "env": env_overrides,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr.splitlines()[-12:],
            }
        )
        if proc.returncode != 0 and payload.get("ok", False):
            payload["ok"] = False
            payload["error_type"] = "NonZeroReturn"
            payload["error"] = f"child process returned {proc.returncode}"
        return payload
    except subprocess.TimeoutExpired as exc:
        return {
            "name": attempt["name"],
            "category": attempt["category"],
            "env_id": attempt["env_id"],
            "kwargs": attempt["kwargs"],
            "env": dict(attempt.get("env") or {}),
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": f"probe exceeded {timeout_s}s",
            "returncode": -1,
            "stdout_tail": (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else "",
        }


def _attempts() -> list[dict[str, Any]]:
    sensor_32 = {"width": 32, "height": 32, "shader_pack": "minimal"}
    sensor_64 = {"width": 64, "height": 64, "shader_pack": "minimal"}
    human_32 = {"width": 32, "height": 32, "shader_pack": "minimal"}
    cpu_render_env = {
        "SAPIEN_RENDER_DEVICE": "cpu",
        "SAPIEN_VULKAN_DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
    }
    swiftshader_env = {
        "SAPIEN_RENDER_DEVICE": "cpu",
        "SAPIEN_VULKAN_DEVICE": "swiftshader",
        "VK_ICD_FILENAMES": "",
        "CUDA_VISIBLE_DEVICES": "",
    }
    disable_vk_layer_env = {
        "SAPIEN_RENDER_DEVICE": "cpu",
        "VK_LOADER_LAYERS_DISABLE": "*",
        "CUDA_VISIBLE_DEVICES": "",
    }
    return [
        {
            "name": "state_baseline_joint_delta",
            "category": "state_baseline",
            "env_id": "PickCube-v1",
            "kwargs": {"obs_mode": "state", "control_mode": "pd_joint_delta_pos", "render_mode": None},
            "do_step": True,
        },
        {
            "name": "rgb_minimal_64",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_64,
            },
        },
        {
            "name": "rgb_minimal_32",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "rgb_minimal_32_env_cpu_render",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "env": cpu_render_env,
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "rgb_minimal_32_env_swiftshader",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "env": swiftshader_env,
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "rgb_minimal_32_env_disable_vk_layers",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "env": disable_vk_layer_env,
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "rgb_render_backend_cpu_32",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
                "render_backend": "cpu",
            },
        },
        {
            "name": "rgb_render_backend_sapien_cpu_32",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
                "render_backend": "sapien_cpu",
            },
        },
        {
            "name": "rgb_physx_cpu_render_cpu_32",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
                "sim_backend": "physx_cpu",
                "render_backend": "cpu",
            },
        },
        {
            "name": "rgbd_minimal_32",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "rgbd",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "state_rgb_minimal_32",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "state+rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "sensor_data_minimal_32",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "sensor_data",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "render_rgb_array_minimal_32",
            "category": "render_mode",
            "env_id": "PickCube-v1",
            "kwargs": {
                "obs_mode": "state",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": "rgb_array",
                "shader_dir": "minimal",
                "human_render_camera_configs": human_32,
            },
            "do_render": True,
        },
        {
            "name": "pushcube_rgb_minimal_32",
            "category": "visual_obs",
            "env_id": "PushCube-v1",
            "kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "render_mode": None,
                "shader_dir": "minimal",
                "sensor_configs": sensor_32,
            },
        },
        {
            "name": "ee_delta_pose_state_probe",
            "category": "ee_control",
            "env_id": "PickCube-v1",
            "kwargs": {"obs_mode": "state", "control_mode": "pd_ee_delta_pose", "render_mode": None},
            "do_step": True,
        },
        {
            "name": "ee_delta_pos_state_probe",
            "category": "ee_control",
            "env_id": "PickCube-v1",
            "kwargs": {"obs_mode": "state", "control_mode": "pd_ee_delta_pos", "render_mode": None},
            "do_step": True,
        },
    ]


def _flatten_for_table(row: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(row.get("kwargs") or {})
    env = dict(row.get("env") or {})
    sensor = kwargs.get("sensor_configs") or {}
    human = kwargs.get("human_render_camera_configs") or {}
    error = row.get("error")
    if error is not None:
        error = str(error).replace("\n", " ")[:500]
    return {
        "name": row.get("name"),
        "category": row.get("category"),
        "env_id": row.get("env_id"),
        "ok": bool(row.get("ok")),
        "obs_mode": kwargs.get("obs_mode"),
        "control_mode": kwargs.get("control_mode"),
        "render_mode": kwargs.get("render_mode"),
        "sim_backend": kwargs.get("sim_backend"),
        "render_backend": kwargs.get("render_backend"),
        "env_overrides": json.dumps(env, sort_keys=True) if env else "",
        "shader_dir": kwargs.get("shader_dir"),
        "sensor_width": sensor.get("width"),
        "sensor_height": sensor.get("height"),
        "human_width": human.get("width"),
        "human_height": human.get("height"),
        "error_type": row.get("error_type"),
        "error": error,
        "returncode": row.get("returncode"),
    }


def _artifact_names(args: argparse.Namespace) -> tuple[str, str]:
    output_tag = str(getattr(args, "output_tag", "") or "").strip()
    attempt_names = list(getattr(args, "attempt_name", []) or [])
    if output_tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in output_tag)
        return f"benchmark_maniskill_visual_probe_{safe_tag}", f"maniskill_visual_{safe_tag}_blocker_report.md"
    if attempt_names:
        return "benchmark_maniskill_visual_probe_filtered", "maniskill_visual_filtered_blocker_report.md"
    return "benchmark_maniskill_visual_probe", "maniskill_visual_blocker_report.md"


def _write_blocker_report(summary: dict[str, Any], report_filename: str = "maniskill_visual_blocker_report.md") -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / report_filename
    visual_failures = [
        row
        for row in summary.get("attempts", [])
        if row.get("category") in {"visual_obs", "render_mode"} and not row.get("ok")
    ]
    ee_failures = [row for row in summary.get("attempts", []) if row.get("category") == "ee_control" and not row.get("ok")]
    lines = [
        "# ManiSkill Visual/EE-Control Probe",
        "",
        "This report is generated by `experiments/benchmark_maniskill_visual_probe.py`.",
        "",
        f"- attempted: `{summary.get('attempted')}`",
        f"- available: `{summary.get('available')}`",
        f"- ManiSkill version: `{summary.get('mani_skill_version')}`",
        f"- Python: `{summary.get('python')}`",
        f"- platform: `{summary.get('platform')}`",
        f"- state baseline ok: `{summary.get('state_baseline_ok')}`",
        f"- any visual success: `{summary.get('any_visual_success')}`",
        f"- any EE-control success: `{summary.get('any_ee_control_success')}`",
        "",
        "## Visual Attempts",
        "",
    ]
    for row in [r for r in summary.get("attempts", []) if r.get("category") in {"visual_obs", "render_mode"}]:
        lines.append(
            f"- `{row.get('name')}` on `{row.get('env_id')}`: ok=`{row.get('ok')}`, "
            f"error=`{row.get('error_type')}: {row.get('error')}`"
        )
    lines.extend(["", "## EE-Control Attempts", ""])
    for row in [r for r in summary.get("attempts", []) if r.get("category") == "ee_control"]:
        lines.append(
            f"- `{row.get('name')}` on `{row.get('env_id')}`: ok=`{row.get('ok')}`, "
            f"error=`{row.get('error_type')}: {row.get('error')}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "ManiSkill state-mode validation is artifact-backed separately. This probe does not verify ManiSkill RGB/RGB-D WAM validation; it documents why the local Windows/SAPIEN/Vulkan path could not produce RGB/RGB-D observations in this environment.",
        ]
    )
    if visual_failures:
        first = visual_failures[0]
        lines.append(
            f"The recurring visual blocker was `{first.get('error_type')}: {first.get('error')}` even with minimal shaders and 32x32 cameras."
        )
    if ee_failures:
        first = ee_failures[0]
        lines.append(f"The EE-control blocker was `{first.get('error_type')}: {first.get('error')}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    artifact_stem, report_filename = _artifact_names(args)
    summary: dict[str, Any] = {
        "experiment": "benchmark_maniskill_visual_probe",
        "attempted": True,
        "available": is_maniskill_available(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "canonical_artifact": artifact_stem == "benchmark_maniskill_visual_probe",
    }
    if not summary["available"]:
        summary.update(
            {
                "mani_skill_version": None,
                "state_baseline_ok": False,
                "any_visual_success": False,
                "any_ee_control_success": False,
                "attempts": [],
                "reason": "ManiSkill import not found",
            }
        )
        table_path = results_dir() / "tables" / f"{artifact_stem}.csv"
        pd.DataFrame([]).to_csv(table_path, index=False)
        summary["artifacts"] = {"table": str(table_path)}
        report_path = _write_blocker_report(summary, report_filename)
        summary["artifacts"]["report"] = str(report_path)
        write_json(results_dir() / f"{artifact_stem}.json", summary)
        return summary

    attempts = _attempts()
    if getattr(args, "attempt_name", None):
        requested = set(args.attempt_name)
        attempts = [attempt for attempt in attempts if attempt["name"] in requested]
    rows = [_run_attempt(attempt, args.timeout_s) for attempt in attempts]
    table_path = results_dir() / "tables" / f"{artifact_stem}.csv"
    pd.DataFrame([_flatten_for_table(row) for row in rows]).to_csv(table_path, index=False)

    mani_versions = [row.get("mani_skill_version") for row in rows if row.get("mani_skill_version")]
    visual_rows = [row for row in rows if row.get("category") in {"visual_obs", "render_mode"}]
    ee_rows = [row for row in rows if row.get("category") == "ee_control"]
    state_rows = [row for row in rows if row.get("category") == "state_baseline"]
    summary.update(
        {
            "mani_skill_version": mani_versions[0] if mani_versions else "unknown",
            "attempts": rows,
            "state_baseline_ok": any(bool(row.get("ok")) for row in state_rows),
            "any_visual_success": any(bool(row.get("ok")) for row in visual_rows),
            "any_ee_control_success": any(bool(row.get("ok")) for row in ee_rows),
            "visual_attempt_count": len(visual_rows),
            "ee_control_attempt_count": len(ee_rows),
            "artifacts": {"table": str(table_path)},
        }
    )
    visual_errors = [str(row.get("error", "")) for row in visual_rows if not row.get("ok")]
    if visual_rows and not summary["any_visual_success"]:
        summary["visual_blocker"] = visual_errors[0] if visual_errors else "unknown visual failure"
    ee_errors = [str(row.get("error", "")) for row in ee_rows if not row.get("ok")]
    if ee_rows and not summary["any_ee_control_success"]:
        summary["ee_control_blocker"] = ee_errors[0] if ee_errors else "unknown EE-control failure"
    report_path = _write_blocker_report(summary, report_filename)
    summary["artifacts"]["report"] = str(report_path)
    write_json(results_dir() / f"{artifact_stem}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=int, default=75)
    parser.add_argument(
        "--attempt-name",
        action="append",
        default=[],
        help="Run only the named attempt. May be repeated. Defaults to the full probe matrix.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional artifact suffix. Filtered runs default to non-canonical filtered artifacts.",
    )
    args = parser.parse_args()
    summary = run(args)
    print(
        "ManiSkill visual probe complete: "
        f"available={summary.get('available')}, "
        f"state_baseline_ok={summary.get('state_baseline_ok')}, "
        f"any_visual_success={summary.get('any_visual_success')}, "
        f"any_ee_control_success={summary.get('any_ee_control_success')}"
    )


if __name__ == "__main__":
    main()
