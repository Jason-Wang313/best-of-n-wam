from __future__ import annotations

import pytest

from wam_inference_value.benchmarks.gym_manip_adapter import GymManipAdapter, is_gym_manip_available


def test_gym_manip_adapter_smoke_if_available():
    ok, reason = is_gym_manip_available()
    if not ok:
        pytest.skip(reason)
    adapter = GymManipAdapter()
    try:
        state = adapter.reset(123)
        assert state.shape[0] == adapter.state_dim
        assert adapter.action_dim > 0
        feature_state = adapter.feature_state(state)
        assert feature_state.shape[0] > state.shape[0]
    finally:
        adapter.close()
