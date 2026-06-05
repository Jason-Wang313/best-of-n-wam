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
    adapter.env_id = "robocasa/PickPlaceCounterToCabinet"
    adapter.inner = types.SimpleNamespace()
    return adapter


def _adapter_with_fixture(env_id: str, fixture_name: str, fixture: object, behavior: str = "") -> RoboCasaAdapter:
    adapter = RoboCasaAdapter.__new__(RoboCasaAdapter)
    adapter.env_id = env_id
    inner = types.SimpleNamespace(**{fixture_name: fixture})
    if behavior:
        inner.behavior = behavior
    adapter.inner = inner
    adapter._raw_obs = types.MethodType(lambda self: {}, adapter)  # type: ignore[method-assign]
    return adapter


class _DoorFixture:
    def __init__(self, value: float):
        self.value = value

    def get_door_state(self, env=None) -> dict[str, float]:
        return {"joint": self.value}


class _StateFixture:
    def __init__(self, state: dict):
        self.state = state

    def get_state(self, env=None) -> dict:
        return self.state


class _HandleFixture:
    def __init__(self, state: dict):
        self.state = state

    def get_handle_state(self, env=None) -> dict:
        return self.state


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


def test_robocasa_task_distance_uses_drawer_joint_thresholds() -> None:
    open_adapter = _adapter_with_fixture("robocasa/OpenDrawer", "drawer", _DoorFixture(0.2), behavior="open")
    close_adapter = _adapter_with_fixture("robocasa/CloseDrawer", "drawer", _DoorFixture(0.2), behavior="close")

    assert open_adapter.task_distance() == pytest.approx(0.75)
    assert close_adapter.task_distance() == pytest.approx(0.15)
    assert open_adapter.object_distance() == pytest.approx(0.75)


def test_robocasa_task_distance_uses_fixture_state_thresholds() -> None:
    kettle_open = _adapter_with_fixture(
        "robocasa/OpenElectricKettleLid",
        "electric_kettle",
        _StateFixture({"lid": 0.2}),
    )
    mixer_close = _adapter_with_fixture(
        "robocasa/CloseStandMixerHead",
        "stand_mixer",
        _StateFixture({"head": 0.4}),
    )
    toaster_on = _adapter_with_fixture(
        "robocasa/TurnOnToasterOven",
        "toaster_oven",
        _StateFixture({"time": 0.03}),
    )

    assert kettle_open.task_distance() == pytest.approx(0.75)
    assert mixer_close.task_distance() == pytest.approx(0.39)
    assert toaster_on.task_distance() == pytest.approx(0.07)


def test_robocasa_task_distance_uses_boolean_and_handle_state() -> None:
    kettle_on = _adapter_with_fixture(
        "robocasa/TurnOnElectricKettle",
        "electric_kettle",
        _StateFixture({"turned_on": False}),
    )
    sink_on = _adapter_with_fixture(
        "robocasa/TurnOnSinkFaucet",
        "sink",
        _HandleFixture({"water_on": False}),
        behavior="turn_on",
    )
    spout = _adapter_with_fixture(
        "robocasa/TurnSinkSpout",
        "sink",
        _HandleFixture({"spout_ori": "left"}),
        behavior="right",
    )

    assert kettle_on.task_distance() == pytest.approx(1.0)
    assert sink_on.task_distance() == pytest.approx(1.0)
    assert spout.task_distance() == pytest.approx(1.0)
