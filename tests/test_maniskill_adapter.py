from __future__ import annotations

import pytest

from wam_inference_value.benchmarks.maniskill_adapter import ManiSkillAdapter, is_maniskill_available


def test_maniskill_adapter_imports_without_optional_dependency():
    adapter = ManiSkillAdapter(require_installed=False)
    for name in ["reset_task", "sample_rollouts", "score_rollouts", "evaluate_real_success", "run_closed_loop"]:
        assert callable(getattr(adapter, name))


def test_maniskill_runtime_path_skips_when_dependency_missing():
    if not is_maniskill_available():
        pytest.skip("ManiSkill is not installed; optional benchmark adapter is skipped")
    adapter = ManiSkillAdapter(require_installed=True)
    assert adapter.available
