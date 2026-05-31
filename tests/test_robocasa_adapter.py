from __future__ import annotations

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
