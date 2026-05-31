import numpy as np
import pytest

from wam_inference_value.benchmarks.robosuite_adapter import RoboSuiteAdapter, is_robosuite_available
from wam_inference_value.benchmarks.robosuite_rollouts import sample_rollout_pool


def test_robosuite_adapter_smoke_if_available():
    ok, reason = is_robosuite_available()
    if not ok:
        pytest.skip(reason)
    adapter = RoboSuiteAdapter("Lift", horizon=4)
    try:
        state = adapter.reset(123)
        assert state.shape[0] == adapter.state_dim
        assert adapter.action_dim >= 6
        distance = adapter.distance_to_goal()
        adapter.step(adapter.heuristic_action())
        assert adapter.distance_to_goal() >= 0.0
        adapter.set_state(state)
        assert adapter.distance_to_goal() == pytest.approx(distance)
        pool = sample_rollout_pool(adapter, state, n_rollouts=3, horizon=2, seed=7)
        assert pool["actions"].shape == (3, 2, adapter.action_dim)
        assert len(pool["records"]) == 3
        assert np.isfinite([r["utility"] for r in pool["records"]]).all()
    finally:
        adapter.close()
