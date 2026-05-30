from __future__ import annotations

import pytest

from wam_inference_value.benchmarks.gym_robotics_adapter import GymRoboticsAdapter, is_gym_robotics_available
from wam_inference_value.benchmarks.gym_robotics_rollouts import sample_rollout_pool


def test_gym_robotics_adapter_smoke_if_available():
    ok, reason = is_gym_robotics_available()
    if not ok:
        pytest.skip(reason)
    adapter = GymRoboticsAdapter("FetchPush-v4")
    try:
        state = adapter.reset(123)
        assert state.shape[0] == adapter.state_dim
        assert adapter.action_dim == 4
        feature_state = adapter.feature_state(state)
        assert feature_state.shape[0] > state.shape[0]
        distance = adapter.distance_to_goal()
        adapter.step(adapter.heuristic_action())
        assert adapter.distance_to_goal() >= 0.0
        adapter.set_state(state)
        assert adapter.distance_to_goal() == pytest.approx(distance)
        pool = sample_rollout_pool(adapter, state, n_rollouts=4, horizon=2, seed=7)
        assert pool["actions"].shape == (4, 2, adapter.action_dim)
        assert len(pool["records"]) == 4
    finally:
        adapter.close()
