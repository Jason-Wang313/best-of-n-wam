"""Optional RoboSuite benchmark integration.

The adapter intentionally imports RoboSuite lazily so normal tests and toy
experiments do not require the optional MuJoCo manipulation stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import random
from typing import Any

import numpy as np


class RoboSuiteUnavailableError(ImportError):
    pass


def _silence_robosuite_logs() -> None:
    logging.getLogger("robosuite_logs").setLevel(logging.ERROR)


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs):
            value = obs[key]
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size:
                parts.append(arr)
        if parts:
            return np.concatenate(parts)
    return np.asarray(obs, dtype=float).reshape(-1)


def _controller_config(controller: str) -> dict[str, Any]:
    from robosuite.controllers import load_composite_controller_config

    return load_composite_controller_config(controller=controller)


def is_robosuite_available(env_name: str = "Lift", robot: str = "Panda") -> tuple[bool, str]:
    if importlib.util.find_spec("robosuite") is None:
        return False, "robosuite import not found"
    try:
        import robosuite as suite

        _silence_robosuite_logs()
        config = _controller_config("BASIC")
        env = suite.make(
            env_name,
            robots=robot,
            controller_configs=config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=8,
        )
        env.reset()
        env.close()
        return True, f"{env_name}/{robot} available"
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"{env_name}/{robot} unavailable: {type(exc).__name__}: {exc}"


@dataclass
class RoboSuiteAdapter:
    """Small state/action adapter for headless RoboSuite manipulation tasks."""

    env_name: str = "Lift"
    robot: str = "Panda"
    horizon: int = 10
    controller: str = "BASIC"
    success_bonus: float = 5.0
    energy_penalty: float = 0.01

    def __post_init__(self) -> None:
        ok, reason = is_robosuite_available(self.env_name, self.robot)
        if not ok:
            raise RoboSuiteUnavailableError(reason)
        import robosuite as suite

        _silence_robosuite_logs()
        config = _controller_config(self.controller)
        self.env = suite.make(
            self.env_name,
            robots=self.robot,
            controller_configs=config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=int(self.horizon),
        )
        self.last_obs_raw = self.env.reset()
        self.last_obs = _flatten_obs(self.last_obs_raw)
        self._refresh_obs()
        low, high = self.env.action_spec
        self.action_low = np.asarray(low, dtype=float).reshape(-1)
        self.action_high = np.asarray(high, dtype=float).reshape(-1)
        self.action_dim = int(self.env.action_dim)
        self.state_dim = int(self.get_state().size)
        self.last_reward = 0.0
        self.last_done = False
        self.last_info: dict[str, Any] = {}

    def close(self) -> None:
        self.env.close()

    def _refresh_obs(self) -> np.ndarray:
        self.last_obs_raw = self.env._get_observations()
        self.last_obs = _flatten_obs(self.last_obs_raw)
        return self.last_obs

    def reset(self, seed: int, task_id: str | None = None) -> np.ndarray:
        if task_id is not None and task_id != self.env_name:
            self.close()
            self.env_name = task_id
            self.__post_init__()
        np.random.seed(int(seed) % (2**32 - 1))
        random.seed(int(seed))
        self.last_obs_raw = self.env.reset()
        self.last_obs = _flatten_obs(self.last_obs_raw)
        self._refresh_obs()
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}
        return self.get_state()

    def reset_task(self, *, seed: int | None = None, task_id: str | None = None) -> np.ndarray:
        return self.reset(0 if seed is None else int(seed), task_id=task_id)

    def get_state(self) -> np.ndarray:
        return np.asarray(self.env.sim.get_state().flatten(), dtype=float).reshape(-1)

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size != self.state_dim:
            raise ValueError(f"state has size {state.size}, expected {self.state_dim}")
        self.env.sim.set_state_from_flattened(state.copy())
        self.env.sim.forward()
        if hasattr(self.env, "timestep"):
            self.env.timestep = 0
        if hasattr(self.env, "done"):
            self.env.done = False
        self.last_reward = self.current_reward()
        self.last_done = False
        self.last_info = {}
        self._refresh_obs()

    def feature_state(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        return np.concatenate([self.get_state(), self.last_obs])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, self.action_low, self.action_high)
        obs, reward, done, info = self.env.step(action)
        self.last_obs_raw = obs
        self.last_obs = _flatten_obs(obs)
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = dict(info)
        return self.last_obs, self.last_reward, self.last_done, False, self.last_info

    def current_reward(self) -> float:
        action = np.zeros(self.action_dim, dtype=float)
        try:
            return float(self.env.reward(action=action))
        except TypeError:
            try:
                return float(self.env.reward(action))
            except TypeError:
                return float(self.env.reward())

    def _obs_value(self, key: str) -> np.ndarray | None:
        if isinstance(self.last_obs_raw, dict) and key in self.last_obs_raw:
            return np.asarray(self.last_obs_raw[key], dtype=float).reshape(-1)
        return None

    def _first_obs_value(self, keys: tuple[str, ...]) -> np.ndarray | None:
        for key in keys:
            arr = self._obs_value(key)
            if arr is not None and arr.size:
                return arr
        return None

    def _body_pos(self, *names: str) -> np.ndarray | None:
        for name in names:
            try:
                body_id = self.env.sim.model.body_name2id(name)
                return np.asarray(self.env.sim.data.body_xpos[body_id], dtype=float).reshape(3)
            except Exception:
                continue
        return None

    def _site_pos(self, *names: str) -> np.ndarray | None:
        for name in names:
            try:
                site_id = self.env.sim.model.site_name2id(name)
                return np.asarray(self.env.sim.data.site_xpos[site_id], dtype=float).reshape(3)
            except Exception:
                continue
        return None

    def _robot_eef_pos(self) -> np.ndarray | None:
        try:
            site_id = self.env.robots[0].eef_site_id
            if isinstance(site_id, dict):
                site_id = next(iter(site_id.values()))
            return np.asarray(self.env.sim.data.site_xpos[int(site_id)], dtype=float).reshape(3)
        except Exception:
            return self._site_pos("gripper0_right_grip_site", "gripper0_grip_site")

    def eef_pos(self) -> np.ndarray:
        direct = self._robot_eef_pos()
        if direct is not None:
            return direct.copy()
        arr = self._first_obs_value(("robot0_eef_pos",))
        return arr[:3].copy() if arr is not None else np.zeros(3, dtype=float)

    def target_pos(self) -> np.ndarray:
        if self.env_name == "Door":
            direct = self._site_pos("Door_handle")
            if direct is not None:
                return direct.copy()
            arr = self._first_obs_value(("handle_pos", "door_pos"))
            return arr[:3].copy() if arr is not None else self.eef_pos()
        if self.env_name == "Stack":
            direct = self._body_pos("cubeA_main")
            if direct is not None:
                return direct.copy()
            arr = self._first_obs_value(("cubeA_pos", "cube_pos"))
            return arr[:3].copy() if arr is not None else self.eef_pos()
        if self.env_name == "NutAssembly":
            direct = self._body_pos("SquareNut_main", "RoundNut_main")
            if direct is not None:
                return direct.copy()
            arr = self._first_obs_value(("SquareNut_pos", "RoundNut_pos"))
            return arr[:3].copy() if arr is not None else self.eef_pos()
        if self.env_name == "PickPlace":
            direct = self._body_pos("Can_main", "Milk_main", "Bread_main", "Cereal_main")
            if direct is not None:
                return direct.copy()
            arr = self._first_obs_value(("Can_pos", "Milk_pos", "Bread_pos", "Cereal_pos"))
            return arr[:3].copy() if arr is not None else self.eef_pos()
        direct = self._body_pos("cube_main", "cubeA_main")
        if direct is not None:
            return direct.copy()
        arr = self._first_obs_value(("cube_pos", "cubeA_pos", "handle_pos"))
        return arr[:3].copy() if arr is not None else self.eef_pos()

    def goal_pos(self) -> np.ndarray:
        if self.env_name == "Stack":
            direct = self._body_pos("cubeB_main")
            if direct is not None:
                return direct.copy()
            arr = self._first_obs_value(("cubeB_pos",))
            return arr[:3].copy() if arr is not None else self.target_pos()
        if self.env_name == "Door":
            arr = self._first_obs_value(("handle_pos",))
            handle = arr[:3].copy() if arr is not None else self.eef_pos()
            return handle + np.array([0.0, -0.20, 0.0])
        target = self.target_pos().copy()
        target[2] += 0.16
        return target

    def task_progress(self) -> float:
        if self.env_name == "Lift":
            cube = self.target_pos()
            eef_dist = float(np.linalg.norm(self.eef_pos() - cube[:3]))
            return float(3.0 * (cube[2] - 0.82) - 0.25 * eef_dist)
        if self.env_name == "Stack":
            delta = self.goal_pos() - self.target_pos()
            return float(-np.linalg.norm(delta[:3]) + 0.5 * max(0.0, -delta[2]))
        if self.env_name == "Door":
            hinge = self._first_obs_value(("hinge_qpos",))
            handle_dist = self.target_pos() - self.eef_pos()
            open_amount = float(hinge[0]) if hinge is not None and hinge.size else 0.0
            approach = -float(np.linalg.norm(handle_dist[:3]))
            return float(open_amount + 0.20 * approach)
        target = self.target_pos()
        return float(-np.linalg.norm(self.eef_pos() - target))

    def distance_to_goal(self) -> float:
        if self.env_name == "Lift":
            cube = self.target_pos()
            return float(max(0.0, 0.98 - cube[2]) + 0.15 * np.linalg.norm(self.eef_pos() - cube[:3]))
        if self.env_name == "Stack":
            return float(np.linalg.norm(self.goal_pos() - self.target_pos()))
        if self.env_name == "Door":
            hinge = self._first_obs_value(("hinge_qpos",))
            handle_dist = self.target_pos() - self.eef_pos()
            open_gap = max(0.0, 0.35 - (float(hinge[0]) if hinge is not None and hinge.size else 0.0))
            approach = float(np.linalg.norm(handle_dist[:3]))
            return float(open_gap + 0.15 * approach)
        return float(np.linalg.norm(self.eef_pos() - self.goal_pos()))

    def evaluate_success(self, state: np.ndarray | None = None) -> bool:
        if state is not None:
            self.set_state(state)
        try:
            return bool(self.env._check_success())
        except Exception:
            return self.distance_to_goal() < 0.05

    def compute_utility(self, state: np.ndarray | None = None) -> float:
        if state is not None:
            self.set_state(state)
        return float(self.current_reward() + self.success_bonus * float(self.evaluate_success()) + self.task_progress() - 0.1 * self.distance_to_goal())

    def sample_initial_states(self, n: int, seed: int) -> list[np.ndarray]:
        return [self.reset(seed + 9973 * i) for i in range(int(n))]

    def heuristic_action(self, scale: float = 6.0) -> np.ndarray:
        eef = self.eef_pos()
        target = self.target_pos()
        goal = self.goal_pos()
        to_target = target[:3] - eef[:3]
        close = float(np.linalg.norm(to_target)) < 0.055
        direction = goal[:3] - target[:3] if close else to_target
        if self.env_name == "Lift" and close:
            direction = np.array([0.0, 0.0, 0.18])
        if self.env_name == "Door" and close:
            direction = np.array([0.0, -0.35, 0.02])
        action = np.zeros(self.action_dim, dtype=float)
        action[: min(3, self.action_dim)] = scale * direction[: min(3, self.action_dim)]
        if self.action_dim >= 6:
            action[3:6] = 0.0
        if self.action_dim >= 1:
            action[-1] = 1.0 if not close else -1.0
        return np.clip(action, self.action_low, self.action_high)

    def sample_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.robosuite_rollouts import sample_rollout_pool

        return sample_rollout_pool(self, *args, **kwargs)

    def score_rollouts(self, rollouts: list[dict[str, Any]], scorer: str = "utility") -> np.ndarray:
        if scorer == "utility":
            return np.asarray([r["utility"] for r in rollouts], dtype=float)
        if scorer == "reward":
            return np.asarray([r["total_reward"] + r["terminal_reward"] for r in rollouts], dtype=float)
        if scorer == "low_energy":
            return -np.asarray([r["energy"] for r in rollouts], dtype=float)
        if scorer == "random":
            return np.random.default_rng(0).normal(size=len(rollouts))
        raise ValueError(f"unknown scorer: {scorer}")

    def evaluate_real_success(self, rollouts: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([float(r["success"]) for r in rollouts], dtype=float)

    def run_closed_loop(self, *args: Any, **kwargs: Any) -> Any:
        from wam_inference_value.benchmarks.robosuite_rollouts import run_closed_loop

        return run_closed_loop(self, *args, **kwargs)
