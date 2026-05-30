"""Optional ManiSkill benchmark integration.

This module deliberately avoids importing ManiSkill at module import time so
the toy project remains installable and testable without benchmark extras.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any

import numpy as np


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
    """Small Gymnasium-style state adapter for ManiSkill3 tasks.

    The default control mode uses joint deltas because it runs on this CPU-only
    Windows environment without Pinocchio. End-effector controls can be used in
    environments where the optional Pinocchio dependency is available.
    """

    env_id: str | None = "PickCube-v1"
    env: Any | None = None
    require_installed: bool = True
    env_kwargs: dict[str, Any] | None = None
    obs_mode: str = "state"
    control_mode: str = "pd_joint_delta_pos"
    render_mode: str | None = None
    horizon: int = 8
    success_bonus: float = 5.0
    energy_penalty: float = 0.002

    def __post_init__(self) -> None:
        self.module_name = _available_module_name()
        if self.module_name is None and self.require_installed:
            raise ManiSkillUnavailableError(
                "ManiSkill is not installed. Install ManiSkill extras before running this optional benchmark adapter."
            )
        self.env_kwargs = dict(self.env_kwargs or {})
        self.task_name = str(self.env_id or "PickCube-v1")
        self.action_dim = 0
        self.state_dim = 0
        self.last_reward = 0.0
        self.last_info: dict[str, Any] = {}
        if self.env is not None:
            self._sync_spaces()

    @property
    def available(self) -> bool:
        return self.module_name is not None

    def _require_available(self) -> None:
        if not self.available:
            raise ManiSkillUnavailableError("ManiSkill is not installed; this optional benchmark path is unavailable.")

    def _ensure_env(self) -> None:
        self._require_available()
        if self.env is None:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401

            kwargs = {
                "obs_mode": self.obs_mode,
                "control_mode": self.control_mode,
                "render_mode": self.render_mode,
            }
            kwargs.update(self.env_kwargs or {})
            self.env = gym.make(str(self.env_id or "PickCube-v1"), **kwargs)
            self._sync_spaces()

    def _sync_spaces(self) -> None:
        low = np.asarray(self.env.action_space.low, dtype=float).reshape(-1)
        self.action_dim = int(low.size)
        try:
            self.state_dim = int(self.get_state().size)
        except Exception:
            self.state_dim = 0

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        if arr.size == 0:
            return False
        return bool(arr.reshape(-1)[0])

    @staticmethod
    def _to_float(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=float)
        return float(arr.reshape(-1)[0])

    def reset(self, seed: int, task_id: str | None = None) -> np.ndarray:
        if task_id is not None and task_id != self.task_name:
            self.close()
            self.env_id = task_id
            self.task_name = task_id
        self.reset_task(seed=seed)
        return self.get_state()

    def reset_task(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        self._ensure_env()
        return self.env.reset(seed=seed, options=options)

    def get_state(self) -> np.ndarray:
        self._ensure_env()
        state = self.env.unwrapped.get_state()
        return self._to_numpy(state).reshape(-1)

    def set_state(self, state: Any) -> None:
        self._ensure_env()
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        self.env.unwrapped.set_state(arr)

    def feature_state(self, state: Any | None = None) -> np.ndarray:
        if state is None:
            try:
                obs = self.env.unwrapped.get_obs()
                return self._to_numpy(obs).reshape(-1)
            except Exception:
                return self.get_state()
        return np.asarray(state, dtype=float).reshape(-1)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.last_reward = self._to_float(reward)
        self.last_info = dict(info)
        return self.feature_state(), self.last_reward, self._to_bool(terminated), self._to_bool(truncated), self.last_info

    def evaluate_success(self, state: Any | None = None) -> bool:
        self._ensure_env()
        if state is not None:
            self.set_state(state)
        try:
            out = self.env.unwrapped.evaluate()
            if "success" in out:
                return self._to_bool(out["success"])
        except Exception:
            pass
        return self._to_bool(self.last_info.get("success", False))

    def compute_utility(self, state: Any | None = None) -> float:
        success = self.evaluate_success(state)
        return float(self.last_reward + self.success_bonus * float(success))

    def sample_initial_states(self, n: int, seed: int) -> list[np.ndarray]:
        return [self.reset(seed + 9973 * i) for i in range(int(n))]

    def sample_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.maniskill_rollouts import sample_rollout_pool

        return sample_rollout_pool(self, *args, **kwargs)

    def score_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        records = args[0] if args else kwargs.get("records")
        scorer = kwargs.get("scorer", "reward")
        if scorer == "reward":
            return np.asarray([r["utility"] for r in records], dtype=float)
        if scorer == "low_energy":
            return -np.asarray([r["energy"] for r in records], dtype=float)
        if scorer == "random":
            seed = int(kwargs.get("seed", 0))
            return np.random.default_rng(seed).normal(size=len(records))
        raise ValueError(f"unknown ManiSkill rollout scorer: {scorer}")

    def evaluate_real_success(self, *args: Any, **kwargs: Any) -> Any:
        return self.evaluate_success(*args, **kwargs)

    def run_closed_loop(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.maniskill_rollouts import run_closed_loop

        return run_closed_loop(self, *args, **kwargs)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
