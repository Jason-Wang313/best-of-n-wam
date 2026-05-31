from __future__ import annotations

import types

import numpy as np
import pytest

from wam_inference_value.benchmarks.robocasa_adapter import RoboCasaAdapter, RoboCasaUnavailableError, is_robocasa_available


def test_robocasa_adapter_skips_when_dependency_missing() -> None:
    ok, reason = is_robocasa_available()
    if not ok:
        with pytest.raises(RoboCasaUnavailableError, match="robocasa|gymnasium|import"):
            RoboCasaAdapter()
        assert reason
        return
    pytest.skip("RoboCasa is installed; heavy task reset is covered by optional benchmark_robocasa_smoke.py")


def _adapter_with_obs(raw: dict) -> RoboCasaAdapter:
    adapter = RoboCasaAdapter.__new__(RoboCasaAdapter)
    adapter._raw_obs = types.MethodType(lambda self: raw, adapter)  # type: ignore[method-assign]
    return adapter


def test_robocasa_object_distance_prefers_broad_task_targets() -> None:
    adapter = _adapter_with_obs(
        {
            "counter_to_robot0_eef_pos": np.array([0.01, 0.0, 0.0]),
            "drawer_obj_to_robot0_eef_pos": np.array([0.3, 0.4, 0.0]),
        }
    )

    assert adapter.object_distance() == pytest.approx(0.5)


def test_robocasa_object_distance_falls_back_to_named_targets() -> None:
    adapter = _adapter_with_obs(
        {
            "random_to_robot0_eef_pos": np.array([0.1, 0.0, 0.0]),
            "handle_to_robot0_eef_pos": np.array([0.0, 0.6, 0.8]),
            "counter_to_robot0_eef_pos": np.array([0.01, 0.0, 0.0]),
        }
    )

    assert adapter.object_distance() == pytest.approx(1.0)
