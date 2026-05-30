"""Benchmark adapter interfaces.

External robotics benchmarks are optional. Adapters implement this interface
when the dependency is installed; otherwise scripts write blocker reports and
mark benchmark claims unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BenchmarkAdapter(Protocol):
    task_name: str
    action_dim: int
    state_dim: int
    horizon: int

    def reset(self, seed: int, task_id: str | None = None) -> Any: ...

    def get_state(self) -> Any: ...

    def set_state(self, state: Any) -> None: ...

    def step(self, action: Any) -> Any: ...

    def evaluate_success(self, state: Any | None = None) -> bool: ...

    def compute_utility(self, state: Any | None = None) -> float: ...

    def sample_initial_states(self, n: int, seed: int) -> list[Any]: ...


@dataclass(frozen=True)
class BenchmarkStatus:
    name: str
    available: bool
    reason: str
