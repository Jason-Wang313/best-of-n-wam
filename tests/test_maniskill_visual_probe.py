from __future__ import annotations

import importlib.util
import types
from pathlib import Path


def load_visual_probe_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "benchmark_maniskill_visual_probe.py"
    spec = importlib.util.spec_from_file_location("benchmark_maniskill_visual_probe_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visual_probe_attempts_include_process_renderer_overrides() -> None:
    module = load_visual_probe_module()
    attempts = module._attempts()
    by_name = {attempt["name"]: attempt for attempt in attempts}

    required = {
        "rgb_minimal_32_env_cpu_render",
        "rgb_minimal_32_env_swiftshader",
        "rgb_minimal_32_env_disable_vk_layers",
    }
    assert required.issubset(by_name)
    assert by_name["rgb_minimal_32_env_cpu_render"]["env"]["SAPIEN_RENDER_DEVICE"] == "cpu"
    assert by_name["rgb_minimal_32_env_swiftshader"]["env"]["SAPIEN_VULKAN_DEVICE"] == "swiftshader"
    assert by_name["rgb_minimal_32_env_disable_vk_layers"]["env"]["VK_LOADER_LAYERS_DISABLE"] == "*"


def test_visual_probe_table_records_env_overrides() -> None:
    module = load_visual_probe_module()
    row = module._flatten_for_table(
        {
            "name": "probe",
            "category": "visual_obs",
            "env_id": "PickCube-v1",
            "ok": False,
            "kwargs": {"obs_mode": "rgb", "control_mode": "pd_joint_delta_pos"},
            "env": {"SAPIEN_RENDER_DEVICE": "cpu"},
        }
    )

    assert row["env_overrides"] == '{"SAPIEN_RENDER_DEVICE": "cpu"}'


def test_filtered_visual_probe_uses_noncanonical_artifact_names() -> None:
    module = load_visual_probe_module()

    stem, report = module._artifact_names(types.SimpleNamespace(attempt_name=["rgb_minimal_32"], output_tag=""))
    tagged_stem, tagged_report = module._artifact_names(types.SimpleNamespace(attempt_name=[], output_tag="cpu-only"))
    canonical_stem, canonical_report = module._artifact_names(types.SimpleNamespace(attempt_name=[], output_tag=""))

    assert stem == "benchmark_maniskill_visual_probe_filtered"
    assert report == "maniskill_visual_filtered_blocker_report.md"
    assert tagged_stem == "benchmark_maniskill_visual_probe_cpu-only"
    assert tagged_report == "maniskill_visual_cpu-only_blocker_report.md"
    assert canonical_stem == "benchmark_maniskill_visual_probe"
    assert canonical_report == "maniskill_visual_blocker_report.md"
