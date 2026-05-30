"""A small CPU-only 2D block-pushing environment.

The environment is intentionally simple, but it exposes the WAM planning
phenomenon the paper needs:

sample action-sequence rollouts -> score imagined futures -> evaluate real
success/utility under possibly mismatched dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DynamicsParams:
    """Hidden physical parameters for the block."""

    mass: float = 1.0
    friction: float = 0.08
    slip: float = 0.0
    stuck: float = 0.0
    lateral_slip: float = 0.0
    nonstationary_shift: float = 0.0


@dataclass(frozen=True)
class BlockPushConfig:
    horizon: int = 12
    episode_horizon: int = 18
    max_push: float = 1.0
    target_radius: float = 0.33
    workspace_radius: float = 3.2
    success_bonus: float = 2.5
    distance_weight: float = 1.0
    progress_weight: float = 0.65
    energy_weight: float = 0.045
    safety_weight: float = 0.4
    release_threshold: float = 0.18


@dataclass(frozen=True)
class BlockPushState:
    obj_xy: np.ndarray
    target_xy: np.ndarray
    t: int
    true_params: DynamicsParams

    def copy_with(self, *, obj_xy: np.ndarray | None = None, t: int | None = None) -> "BlockPushState":
        return BlockPushState(
            obj_xy=np.asarray(self.obj_xy if obj_xy is None else obj_xy, dtype=float),
            target_xy=np.asarray(self.target_xy, dtype=float),
            t=self.t if t is None else int(t),
            true_params=self.true_params,
        )


@dataclass(frozen=True)
class RolloutMetrics:
    initial_distance: float
    final_distance: float
    progress: float
    energy: float
    safety_violation: float
    success: bool
    utility: float


class BlockPush2D:
    """Block-pushing with nominal WAM dynamics and hidden true dynamics."""

    def __init__(
        self,
        config: BlockPushConfig | None = None,
        nominal_params: DynamicsParams | None = None,
    ) -> None:
        self.config = config or BlockPushConfig()
        self.nominal_params = nominal_params or DynamicsParams()

    def sample_state(self, seed: int, mismatch: str = "mild", state_id: int = 0) -> BlockPushState:
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        distance = rng.uniform(1.1, 2.65)
        target = np.array([np.cos(angle), np.sin(angle)]) * distance
        obj = rng.normal(0.0, 0.05, size=2)
        params = self.true_params_for(mismatch, rng, state_id)
        return BlockPushState(obj_xy=obj, target_xy=target, t=0, true_params=params)

    def true_params_for(self, mismatch: str, rng: np.random.Generator, state_id: int = 0) -> DynamicsParams:
        """Return hidden true parameters for a mismatch regime."""

        jitter = lambda scale: float(rng.normal(0.0, scale))
        if mismatch == "none":
            return self.nominal_params
        if mismatch == "mild":
            return DynamicsParams(
                mass=1.12 + jitter(0.04),
                friction=0.14 + abs(jitter(0.025)),
                slip=0.12 + abs(jitter(0.025)),
                stuck=0.02,
                lateral_slip=0.05,
            )
        if mismatch == "severe":
            return DynamicsParams(
                mass=1.58 + abs(jitter(0.08)),
                friction=0.28 + abs(jitter(0.04)),
                slip=0.48 + abs(jitter(0.05)),
                stuck=0.08,
                lateral_slip=0.16,
            )
        if mismatch == "stuck_slip":
            return DynamicsParams(
                mass=1.85 + abs(jitter(0.08)),
                friction=0.36 + abs(jitter(0.04)),
                slip=0.68 + abs(jitter(0.04)),
                stuck=0.16,
                lateral_slip=0.24,
            )
        if mismatch == "nonstationary":
            base = DynamicsParams(
                mass=1.12 + abs(jitter(0.04)),
                friction=0.12 + abs(jitter(0.02)),
                slip=0.10 + abs(jitter(0.02)),
                stuck=0.02,
                lateral_slip=0.04,
                nonstationary_shift=0.48 + 0.05 * (state_id % 3),
            )
            return base
        raise ValueError(f"unknown mismatch regime: {mismatch}")

    def shifted_params(self, params: DynamicsParams, t: int) -> DynamicsParams:
        """Apply a mid-episode shift for the nonstationary stress test."""

        if params.nonstationary_shift <= 0.0 or t < self.config.episode_horizon // 2:
            return params
        return replace(
            params,
            friction=min(0.7, params.friction + params.nonstationary_shift),
            slip=min(0.85, params.slip + 0.6 * params.nonstationary_shift),
            lateral_slip=min(0.35, params.lateral_slip + 0.25 * params.nonstationary_shift),
        )

    def step(
        self,
        state: BlockPushState,
        action: np.ndarray,
        params: DynamicsParams | None = None,
        *,
        use_nonstationary_shift: bool = True,
    ) -> BlockPushState:
        params = state.true_params if params is None else params
        if use_nonstationary_shift:
            params = self.shifted_params(params, state.t)
        action = self.clip_action(action)
        mag = float(np.linalg.norm(action))
        if mag <= 1e-12:
            return state.copy_with(t=state.t + 1)

        direction = action / mag
        force = min(mag, self.config.max_push)
        release = 0.0 if force < self.config.release_threshold and params.stuck > 0.0 else 1.0
        stuck_scale = max(0.0, 1.0 - params.stuck)
        base_gain = max(0.02, 1.0 - params.friction) / max(0.2, params.mass)
        slip_loss = max(0.0, 1.0 - params.slip * force * force)
        step_vec = direction * force * base_gain * slip_loss * stuck_scale * release

        if params.lateral_slip > 0.0:
            perp = np.array([-direction[1], direction[0]])
            sign = 1.0 if (state.t % 2 == 0) else -1.0
            step_vec += sign * perp * params.lateral_slip * force * force

        new_obj = state.obj_xy + step_vec
        return state.copy_with(obj_xy=new_obj, t=state.t + 1)

    def rollout(
        self,
        state: BlockPushState,
        actions: np.ndarray,
        params: DynamicsParams,
        *,
        use_nonstationary_shift: bool = True,
    ) -> tuple[BlockPushState, RolloutMetrics, np.ndarray]:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("actions must have shape (horizon, 2)")
        cur = state
        trajectory = [np.asarray(cur.obj_xy, dtype=float)]
        energy = 0.0
        safety = 0.0
        initial_distance = self.distance_to_target(cur)
        for action in actions:
            action = self.clip_action(action)
            energy += float(np.dot(action, action))
            cur = self.step(cur, action, params, use_nonstationary_shift=use_nonstationary_shift)
            trajectory.append(np.asarray(cur.obj_xy, dtype=float))
            excess = max(0.0, float(np.linalg.norm(cur.obj_xy)) - self.config.workspace_radius)
            safety += excess * excess
        final_distance = self.distance_to_target(cur)
        progress = initial_distance - final_distance
        success = final_distance <= self.config.target_radius
        utility = (
            self.config.success_bonus * float(success)
            + self.config.progress_weight * progress
            - self.config.distance_weight * final_distance
            - self.config.energy_weight * energy
            - self.config.safety_weight * safety
        )
        metrics = RolloutMetrics(
            initial_distance=float(initial_distance),
            final_distance=float(final_distance),
            progress=float(progress),
            energy=float(energy),
            safety_violation=float(safety),
            success=bool(success),
            utility=float(utility),
        )
        return cur, metrics, np.asarray(trajectory, dtype=float)

    def rollout_batch_metrics(
        self,
        state: BlockPushState,
        actions_batch: np.ndarray,
        params: DynamicsParams,
        *,
        use_nonstationary_shift: bool = True,
    ) -> list[RolloutMetrics]:
        """Vectorized rollout metrics for a batch of action sequences."""

        actions = np.asarray(actions_batch, dtype=float)
        if actions.ndim != 3 or actions.shape[2] != 2:
            raise ValueError("actions_batch must have shape (n_rollouts, horizon, 2)")
        n, horizon, _ = actions.shape
        pos = np.repeat(np.asarray(state.obj_xy, dtype=float)[None, :], n, axis=0)
        target = np.asarray(state.target_xy, dtype=float)
        initial_distance = np.linalg.norm(target[None, :] - pos, axis=1)
        energy = np.zeros(n, dtype=float)
        safety = np.zeros(n, dtype=float)

        for h in range(horizon):
            p = params
            if use_nonstationary_shift:
                p = self.shifted_params(params, state.t + h)
            action = actions[:, h, :]
            mag = np.linalg.norm(action, axis=1)
            scale = np.ones(n, dtype=float)
            over = mag > self.config.max_push
            scale[over] = self.config.max_push / np.maximum(mag[over], 1e-12)
            action = action * scale[:, None]
            mag = np.linalg.norm(action, axis=1)
            energy += np.sum(action * action, axis=1)

            direction = np.zeros_like(action)
            active = mag > 1e-12
            direction[active] = action[active] / mag[active, None]
            force = np.minimum(mag, self.config.max_push)
            release = np.ones(n, dtype=float)
            if p.stuck > 0.0:
                release[force < self.config.release_threshold] = 0.0
            stuck_scale = max(0.0, 1.0 - p.stuck)
            base_gain = max(0.02, 1.0 - p.friction) / max(0.2, p.mass)
            slip_loss = np.maximum(0.0, 1.0 - p.slip * force * force)
            step_vec = direction * (force * base_gain * slip_loss * stuck_scale * release)[:, None]
            if p.lateral_slip > 0.0:
                perp = np.column_stack([-direction[:, 1], direction[:, 0]])
                sign = 1.0 if ((state.t + h) % 2 == 0) else -1.0
                step_vec += sign * perp * (p.lateral_slip * force * force)[:, None]
            pos += step_vec
            excess = np.maximum(0.0, np.linalg.norm(pos, axis=1) - self.config.workspace_radius)
            safety += excess * excess

        final_distance = np.linalg.norm(target[None, :] - pos, axis=1)
        progress = initial_distance - final_distance
        success = final_distance <= self.config.target_radius
        utility = (
            self.config.success_bonus * success.astype(float)
            + self.config.progress_weight * progress
            - self.config.distance_weight * final_distance
            - self.config.energy_weight * energy
            - self.config.safety_weight * safety
        )
        return [
            RolloutMetrics(
                initial_distance=float(initial_distance[i]),
                final_distance=float(final_distance[i]),
                progress=float(progress[i]),
                energy=float(energy[i]),
                safety_violation=float(safety[i]),
                success=bool(success[i]),
                utility=float(utility[i]),
            )
            for i in range(n)
        ]

    def distance_to_target(self, state: BlockPushState) -> float:
        return float(np.linalg.norm(np.asarray(state.target_xy) - np.asarray(state.obj_xy)))

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        norm = float(np.linalg.norm(action))
        if norm <= self.config.max_push or norm == 0.0:
            return action.copy()
        return action / norm * self.config.max_push
