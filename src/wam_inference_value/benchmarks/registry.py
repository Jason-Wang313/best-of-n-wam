from __future__ import annotations

from wam_inference_value.benchmarks.base import BenchmarkStatus
from wam_inference_value.benchmarks.gym_manip_adapter import is_gym_manip_available
from wam_inference_value.benchmarks.gym_robotics_adapter import is_gym_robotics_available
from wam_inference_value.benchmarks.libero_adapter import is_libero_available
from wam_inference_value.benchmarks.maniskill_adapter import is_maniskill_available
from wam_inference_value.benchmarks.metaworld_adapter import is_metaworld_available
from wam_inference_value.benchmarks.robosuite_adapter import is_robosuite_available


def benchmark_statuses() -> list[BenchmarkStatus]:
    statuses = [
        BenchmarkStatus(
            name="maniskill",
            available=is_maniskill_available(),
            reason="available with state-mode joint-delta control" if is_maniskill_available() else "ManiSkill import not found",
        )
    ]
    gym_ok, gym_reason = is_gym_manip_available()
    statuses.append(BenchmarkStatus(name="gym_manip", available=gym_ok, reason=gym_reason))
    robotics_ok, robotics_reason = is_gym_robotics_available()
    statuses.append(BenchmarkStatus(name="gym_robotics", available=robotics_ok, reason=robotics_reason))
    metaworld_ok, metaworld_reason = is_metaworld_available()
    statuses.append(BenchmarkStatus(name="metaworld", available=metaworld_ok, reason=metaworld_reason))
    robosuite_ok, robosuite_reason = is_robosuite_available()
    statuses.append(BenchmarkStatus(name="robosuite", available=robosuite_ok, reason=robosuite_reason))
    libero_ok, libero_reason = is_libero_available()
    statuses.append(BenchmarkStatus(name="libero", available=libero_ok, reason=libero_reason))
    statuses.append(BenchmarkStatus(name="robocasa", available=False, reason="adapter skeleton only; dependency not installed/validated"))
    return statuses
