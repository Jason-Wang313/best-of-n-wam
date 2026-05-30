"""Optional ManiSkill benchmark integration skeleton.

This module deliberately avoids importing ManiSkill at module import time so
the toy project remains installable and testable without benchmark extras.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any


class ManiSkillUnavailableError(ImportError):
    """Raised when ManiSkill-specific execution is requested without ManiSkill."""


def _available_module_name() -> str | None:
    for module_name in ("mani_skill", "mani_skill2"):
        if importlib.util.find_spec(module_name) is not None:
            return module_name
    return None


def is_maniskill_available() -> bool:
    return _available_module_name() is not None


@dataclass
class ManiSkillAdapter:
    """Interface placeholder for future real-robot-style benchmark runs."""

    env_id: str | None = None
    env: Any | None = None
    require_installed: bool = True
    env_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.module_name = _available_module_name()
        if self.module_name is None and self.require_installed:
            raise ManiSkillUnavailableError(
                "ManiSkill is not installed. Install ManiSkill extras before running this optional benchmark adapter."
            )
        self.env_kwargs = dict(self.env_kwargs or {})

    @property
    def available(self) -> bool:
        return self.module_name is not None

    def _require_available(self) -> None:
        if not self.available:
            raise ManiSkillUnavailableError("ManiSkill is not installed; this optional benchmark path is unavailable.")

    def reset_task(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        self._require_available()
        if self.env is None:
            raise NotImplementedError("ManiSkill environment construction is intentionally left to a future benchmark run.")
        return self.env.reset(seed=seed, options=options)

    def sample_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        self._require_available()
        raise NotImplementedError("ManiSkill rollout sampling skeleton is defined but not implemented in this toy repo.")

    def score_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        self._require_available()
        raise NotImplementedError("ManiSkill rollout scoring skeleton is defined but not implemented in this toy repo.")

    def evaluate_real_success(self, *args: Any, **kwargs: Any) -> Any:
        self._require_available()
        raise NotImplementedError("ManiSkill real-success evaluation skeleton is defined but not implemented in this toy repo.")

    def run_closed_loop(self, *args: Any, **kwargs: Any) -> Any:
        self._require_available()
        raise NotImplementedError("ManiSkill closed-loop execution skeleton is defined but not implemented in this toy repo.")
