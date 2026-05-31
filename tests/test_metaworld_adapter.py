import pytest

from wam_inference_value.benchmarks.metaworld_adapter import MetaWorldAdapter, is_metaworld_available
from wam_inference_value.benchmarks.metaworld_rollouts import sample_rollout_pool


def test_metaworld_adapter_smoke_if_available():
    ok, reason = is_metaworld_available()
    if not ok:
        pytest.skip(reason)
    adapter = MetaWorldAdapter("reach-v3")
    try:
        state = adapter.reset(123)
        assert state.shape[0] == adapter.state_dim
        assert adapter.action_dim == 4
        distance = adapter.distance_to_goal()
        adapter.step(adapter.heuristic_action())
        assert adapter.distance_to_goal() >= 0.0
        adapter.set_state(state)
        assert adapter.distance_to_goal() == pytest.approx(distance)
        pool = sample_rollout_pool(adapter, state, n_rollouts=3, horizon=2, seed=7)
        assert pool["actions"].shape == (3, 2, adapter.action_dim)
        assert len(pool["records"]) == 3
    finally:
        adapter.close()
