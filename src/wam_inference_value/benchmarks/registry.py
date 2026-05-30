from __future__ import annotations

from wam_inference_value.benchmarks.base import BenchmarkStatus
from wam_inference_value.benchmarks.gym_manip_adapter import is_gym_manip_available
from wam_inference_value.benchmarks.maniskill_adapter import is_maniskill_available


def benchmark_statuses() -> list[BenchmarkStatus]:
    statuses = [
        BenchmarkStatus(
            name="maniskill",
            available=is_maniskill_available(),
            reason="available" if is_maniskill_available() else "ManiSkill import not found",
        )
    ]
    gym_ok, gym_reason = is_gym_manip_available()
    statuses.append(BenchmarkStatus(name="gym_manip", available=gym_ok, reason=gym_reason))
    for name in ["libero", "robocasa"]:
        statuses.append(BenchmarkStatus(name=name, available=False, reason="adapter skeleton only; dependency not installed/validated"))
    return statuses
