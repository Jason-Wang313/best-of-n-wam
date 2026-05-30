"""Additional CPU-only toy robotics environments.

These environments are deliberately low-dimensional, deterministic under seed,
and cheap enough for CI. They are not benchmark simulators; they stress
different failure modes for rollout-score selection: joint friction/stiction,
grasp slip/deformation, and nonstationary physical drift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ToyState:
    vector: np.ndarray
    target: np.ndarray
    t: int
    params: dict[str, float]

    def copy_with(self, *, vector: np.ndarray | None = None, t: int | None = None, params: dict[str, float] | None = None) -> "ToyState":
        return ToyState(
            vector=np.asarray(self.vector if vector is None else vector, dtype=float),
            target=np.asarray(self.target, dtype=float),
            t=self.t if t is None else int(t),
            params=dict(self.params if params is None else params),
        )


@dataclass(frozen=True)
class ToyRolloutMetrics:
    initial_distance: float
    final_distance: float
    progress: float
    energy: float
    safety_violation: float
    success: bool
    utility: float


class BaseToyEnv:
    name = "base"
    state_dim = 4
    action_dim = 2
    horizon = 10
    episode_horizon = 12
    target_radius = 0.18
    max_action = 1.0
    nominal_params: dict[str, float] = {}

    def reset(self, seed: int, mismatch: str = "mild", state_id: int = 0) -> ToyState:
        return self.sample_state(seed, mismatch=mismatch, state_id=state_id)

    def clone(self, state: ToyState) -> ToyState:
        return state.copy_with()

    def get_state(self, state: ToyState) -> np.ndarray:
        return np.asarray(state.vector, dtype=float).copy()

    def set_state(self, state: ToyState, vector: np.ndarray) -> ToyState:
        return state.copy_with(vector=np.asarray(vector, dtype=float))

    def sample_state(self, seed: int, mismatch: str = "mild", state_id: int = 0) -> ToyState:
        raise NotImplementedError

    def params_for(self, mismatch: str, rng: np.random.Generator, state_id: int) -> dict[str, float]:
        raise NotImplementedError

    def step(self, state: ToyState, action: np.ndarray, params: dict[str, float] | None = None, *, use_nonstationary_shift: bool = True) -> ToyState:
        raise NotImplementedError

    def imagined_params(self) -> dict[str, float]:
        return dict(self.nominal_params)

    def shifted_params(self, params: dict[str, float], t: int) -> dict[str, float]:
        return dict(params)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        norm = float(np.linalg.norm(action))
        if norm <= self.max_action or norm <= 1e-12:
            return action.copy()
        return action / norm * self.max_action

    def distance_to_target(self, state: ToyState) -> float:
        return float(np.linalg.norm(np.asarray(state.target) - np.asarray(state.vector[: len(state.target)])))

    def rollout(self, state: ToyState, actions: np.ndarray, params: dict[str, float], *, use_nonstationary_shift: bool = True):
        cur = state
        energy = 0.0
        safety = 0.0
        initial_distance = self.distance_to_target(cur)
        trajectory = [cur.vector.copy()]
        for action in np.asarray(actions, dtype=float):
            action = self.clip_action(action)
            energy += float(np.dot(action, action))
            cur = self.step(cur, action, params, use_nonstationary_shift=use_nonstationary_shift)
            trajectory.append(cur.vector.copy())
            safety += self.safety_cost(cur, action)
        final_distance = self.distance_to_target(cur)
        progress = initial_distance - final_distance
        success = final_distance <= self.target_radius
        utility = self.utility(success, progress, final_distance, energy, safety, cur)
        return cur, ToyRolloutMetrics(float(initial_distance), float(final_distance), float(progress), float(energy), float(safety), bool(success), float(utility)), np.asarray(trajectory)

    def rollout_batch_metrics(self, state: ToyState, actions_batch: np.ndarray, params: dict[str, float], *, use_nonstationary_shift: bool = True) -> list[ToyRolloutMetrics]:
        return [self.rollout(state, seq, params, use_nonstationary_shift=use_nonstationary_shift)[1] for seq in np.asarray(actions_batch, dtype=float)]

    def safety_cost(self, state: ToyState, action: np.ndarray) -> float:
        return float(max(0.0, np.linalg.norm(action) - 0.85) ** 2)

    def utility(self, success: bool, progress: float, final_distance: float, energy: float, safety: float, state: ToyState) -> float:
        return 2.5 * float(success) + 0.8 * progress - final_distance - 0.04 * energy - 0.7 * safety


class DrawerPull1D(BaseToyEnv):
    name = "drawer_pull"
    horizon = 10
    episode_horizon = 12
    target_radius = 0.12
    nominal_params = {"friction": 0.08, "lock": 0.0, "spring": 0.03, "angle_error": 0.0, "damage_force": 1.4}

    def params_for(self, mismatch: str, rng: np.random.Generator, state_id: int) -> dict[str, float]:
        if mismatch == "none":
            return dict(self.nominal_params)
        if mismatch == "mild":
            return {"friction": 0.16 + abs(rng.normal(0, 0.03)), "lock": 0.0, "spring": 0.06, "angle_error": 0.08, "damage_force": 1.15}
        if mismatch == "severe":
            return {"friction": 0.34 + abs(rng.normal(0, 0.04)), "lock": 0.12, "spring": 0.14, "angle_error": 0.24, "damage_force": 0.95}
        if mismatch == "stuck_slip":
            return {"friction": 0.42, "lock": 0.36, "spring": 0.18, "angle_error": 0.30, "damage_force": 0.85}
        if mismatch == "nonstationary":
            return {"friction": 0.12, "lock": 0.0, "spring": 0.05, "angle_error": 0.08, "damage_force": 1.1, "shift": 0.35}
        raise ValueError(f"unknown mismatch: {mismatch}")

    def sample_state(self, seed: int, mismatch: str = "mild", state_id: int = 0) -> ToyState:
        rng = np.random.default_rng(seed)
        start = rng.uniform(0.0, 0.08)
        target = np.asarray([rng.uniform(0.82, 1.05)], dtype=float)
        vector = np.asarray([start, 0.0, 0.0, 0.0], dtype=float)
        return ToyState(vector=vector, target=target, t=0, params=self.params_for(mismatch, rng, state_id))

    def shifted_params(self, params: dict[str, float], t: int) -> dict[str, float]:
        p = dict(params)
        if "shift" in p and t >= self.episode_horizon // 2:
            p["friction"] += p["shift"]
            p["lock"] += 0.12
        return p

    def step(self, state: ToyState, action: np.ndarray, params: dict[str, float] | None = None, *, use_nonstationary_shift: bool = True) -> ToyState:
        p = dict(state.params if params is None else params)
        if use_nonstationary_shift:
            p = self.shifted_params(p, state.t)
        a = self.clip_action(action)
        pull = max(0.0, float(a[0]))
        wiggle = abs(float(a[1]))
        angle_loss = 1.0 - min(0.85, p["angle_error"] * abs(float(a[1])))
        lock_release = 1.0 if pull + 0.8 * wiggle > p["lock"] else 0.25
        dx = max(0.0, pull - p["friction"] - p["spring"] * state.vector[0]) * angle_loss * lock_release
        new_x = np.clip(state.vector[0] + dx, 0.0, 1.25)
        damage = max(0.0, pull + wiggle - p["damage_force"])
        return state.copy_with(vector=np.asarray([new_x, pull, wiggle, state.vector[3] + damage * damage]), t=state.t + 1)

    def safety_cost(self, state: ToyState, action: np.ndarray) -> float:
        return float(state.vector[3])


class SlipperyGrasp1D(BaseToyEnv):
    name = "slippery_grasp"
    horizon = 10
    episode_horizon = 12
    target_radius = 0.16
    nominal_params = {"mass": 1.0, "grip_friction": 0.9, "compliance": 0.05, "bad_point": 0.0, "drop_threshold": 0.75}

    def params_for(self, mismatch: str, rng: np.random.Generator, state_id: int) -> dict[str, float]:
        if mismatch == "none":
            return dict(self.nominal_params)
        if mismatch == "mild":
            return {"mass": 1.15, "grip_friction": 0.68, "compliance": 0.09, "bad_point": 0.10, "drop_threshold": 0.68}
        if mismatch == "severe":
            return {"mass": 1.45, "grip_friction": 0.42, "compliance": 0.18, "bad_point": 0.22, "drop_threshold": 0.55}
        if mismatch == "stuck_slip":
            return {"mass": 1.7, "grip_friction": 0.30, "compliance": 0.26, "bad_point": 0.34, "drop_threshold": 0.48}
        if mismatch == "nonstationary":
            return {"mass": 1.15, "grip_friction": 0.75, "compliance": 0.08, "bad_point": 0.10, "drop_threshold": 0.65, "shift": 0.30}
        raise ValueError(f"unknown mismatch: {mismatch}")

    def sample_state(self, seed: int, mismatch: str = "mild", state_id: int = 0) -> ToyState:
        rng = np.random.default_rng(seed)
        vector = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)  # lift, grip, slip, deformation
        target = np.asarray([rng.uniform(0.82, 1.05)], dtype=float)
        return ToyState(vector=vector, target=target, t=0, params=self.params_for(mismatch, rng, state_id))

    def shifted_params(self, params: dict[str, float], t: int) -> dict[str, float]:
        p = dict(params)
        if "shift" in p and t >= self.episode_horizon // 2:
            p["grip_friction"] = max(0.2, p["grip_friction"] - p["shift"])
            p["compliance"] += 0.08
        return p

    def step(self, state: ToyState, action: np.ndarray, params: dict[str, float] | None = None, *, use_nonstationary_shift: bool = True) -> ToyState:
        p = dict(state.params if params is None else params)
        if use_nonstationary_shift:
            p = self.shifted_params(p, state.t)
        a = self.clip_action(action)
        lift_cmd = max(0.0, float(a[0]))
        squeeze = max(0.0, float(a[1]))
        grip = np.clip(0.72 * state.vector[1] + 0.55 * squeeze - p["bad_point"], 0.0, 1.5)
        slip = max(0.0, lift_cmd * p["mass"] - grip * p["grip_friction"])
        deformation = state.vector[3] + max(0.0, squeeze - 0.72) ** 2 * (1.0 + 2.0 * p["compliance"])
        lift_gain = max(0.0, lift_cmd - 0.55 * slip) / max(0.5, p["mass"])
        lift = max(0.0, state.vector[0] + lift_gain - 0.22 * slip)
        dropped = slip > p["drop_threshold"]
        if dropped:
            lift = max(0.0, 0.35 * lift)
        return state.copy_with(vector=np.asarray([lift, grip, state.vector[2] + slip, deformation]), t=state.t + 1)

    def safety_cost(self, state: ToyState, action: np.ndarray) -> float:
        return float(state.vector[2] ** 2 + 1.5 * state.vector[3])


class NonstationaryPhysicalShiftEnv(DrawerPull1D):
    name = "nonstationary_shift"
    episode_horizon = 14

    def params_for(self, mismatch: str, rng: np.random.Generator, state_id: int) -> dict[str, float]:
        mode = "gradual" if state_id % 2 else "abrupt"
        return {"friction": 0.11, "lock": 0.0, "spring": 0.05, "angle_error": 0.06, "damage_force": 1.1, "shift": 0.42, "mode": mode}

    def shifted_params(self, params: dict[str, float], t: int) -> dict[str, float]:
        p = dict(params)
        mode = p.get("mode", "abrupt")
        if mode == "gradual":
            frac = min(1.0, max(0.0, t / max(1, self.episode_horizon - 1)))
        else:
            frac = 1.0 if t >= self.episode_horizon // 2 else 0.0
        p["friction"] += frac * p["shift"]
        p["lock"] += 0.18 * frac
        p["angle_error"] += 0.18 * frac
        return p


class DeformableToyEnv(BaseToyEnv):
    name = "deformable_toy"
    horizon = 10
    episode_horizon = 12
    target_radius = 0.14
    nominal_params = {
        "mass": 1.0,
        "damping": 0.12,
        "compliance": 0.08,
        "elasticity": 0.45,
        "plasticity": 0.02,
        "damage_threshold": 0.65,
    }

    def params_for(self, mismatch: str, rng: np.random.Generator, state_id: int) -> dict[str, float]:
        if mismatch == "none":
            return dict(self.nominal_params)
        if mismatch == "mild":
            return {"mass": 1.08, "damping": 0.16, "compliance": 0.13, "elasticity": 0.38, "plasticity": 0.04, "damage_threshold": 0.56}
        if mismatch == "severe":
            return {"mass": 1.25, "damping": 0.22, "compliance": 0.28, "elasticity": 0.24, "plasticity": 0.09, "damage_threshold": 0.38}
        if mismatch == "stuck_slip":
            return {"mass": 1.35, "damping": 0.26, "compliance": 0.36, "elasticity": 0.16, "plasticity": 0.14, "damage_threshold": 0.30}
        if mismatch == "nonstationary":
            return {"mass": 1.08, "damping": 0.15, "compliance": 0.12, "elasticity": 0.38, "plasticity": 0.04, "damage_threshold": 0.55, "shift": 0.20}
        raise ValueError(f"unknown mismatch: {mismatch}")

    def sample_state(self, seed: int, mismatch: str = "mild", state_id: int = 0) -> ToyState:
        rng = np.random.default_rng(seed)
        vector = np.asarray([rng.uniform(0.0, 0.06), 0.0, 0.0, 0.0], dtype=float)
        target = np.asarray([rng.uniform(0.78, 1.0)], dtype=float)
        return ToyState(vector=vector, target=target, t=0, params=self.params_for(mismatch, rng, state_id))

    def shifted_params(self, params: dict[str, float], t: int) -> dict[str, float]:
        p = dict(params)
        if "shift" in p and t >= self.episode_horizon // 2:
            p["compliance"] += p["shift"]
            p["elasticity"] = max(0.08, p["elasticity"] - 0.5 * p["shift"])
            p["damage_threshold"] = max(0.2, p["damage_threshold"] - 0.6 * p["shift"])
        return p

    def step(self, state: ToyState, action: np.ndarray, params: dict[str, float] | None = None, *, use_nonstationary_shift: bool = True) -> ToyState:
        p = dict(state.params if params is None else params)
        if use_nonstationary_shift:
            p = self.shifted_params(p, state.t)
        a = self.clip_action(action)
        force = max(0.0, float(a[0]))
        support = np.clip(float(a[1]), 0.0, 1.0)
        deformation = max(
            0.0,
            state.vector[2] * (1.0 - p["elasticity"] * 0.35)
            + p["compliance"] * force * force * (1.0 - 0.65 * support)
            + p["plasticity"] * max(0.0, force - 0.45),
        )
        drag = p["damping"] + 0.55 * deformation
        accel = (force * (0.85 + 0.25 * support) - drag * state.vector[1]) / max(0.35, p["mass"])
        velocity = np.clip(0.72 * state.vector[1] + 0.45 * accel, -0.25, 0.55)
        position = np.clip(state.vector[0] + velocity, 0.0, 1.25)
        damage = state.vector[3] + max(0.0, deformation - p["damage_threshold"]) ** 2 + 0.08 * max(0.0, force - 0.92) ** 2
        return state.copy_with(vector=np.asarray([position, velocity, deformation, damage]), t=state.t + 1)

    def safety_cost(self, state: ToyState, action: np.ndarray) -> float:
        return float(0.8 * state.vector[2] ** 2 + 2.0 * state.vector[3])

    def utility(self, success: bool, progress: float, final_distance: float, energy: float, safety: float, state: ToyState) -> float:
        return 2.8 * float(success) + 0.9 * progress - final_distance - 0.035 * energy - 0.95 * safety


ENV_REGISTRY = {
    "drawer_pull": DrawerPull1D,
    "slippery_grasp": SlipperyGrasp1D,
    "nonstationary_shift": NonstationaryPhysicalShiftEnv,
    "deformable_toy": DeformableToyEnv,
}


def make_toy_env(name: str) -> BaseToyEnv:
    if name not in ENV_REGISTRY:
        raise ValueError(f"unknown toy env: {name}")
    return ENV_REGISTRY[name]()


def sample_toy_action_sequences(env: BaseToyEnv, state: ToyState, n_rollouts: int, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions = np.zeros((int(n_rollouts), int(horizon), env.action_dim), dtype=float)
    dist = max(1e-6, env.distance_to_target(state))
    for i in range(int(n_rollouts)):
        mode = rng.choice(["goal", "cautious", "wiggle", "explore"], p=[0.48, 0.20, 0.18, 0.14])
        for t in range(int(horizon)):
            decay = 1.0 - 0.35 * (t / max(1, horizon - 1))
            if mode == "explore":
                a = rng.normal(0.0, 0.45, size=env.action_dim)
            elif mode == "wiggle":
                a = np.asarray([rng.uniform(0.25, 0.75), rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.9)])
            elif mode == "cautious":
                a = np.asarray([rng.uniform(0.12, 0.45), rng.uniform(0.0, 0.25)])
            else:
                a = np.asarray([min(env.max_action, dist / max(1, horizon - t) + rng.uniform(0.2, 0.55)), rng.normal(0.0, 0.12)])
            actions[i, t, :] = env.clip_action(a * decay)
    return actions
