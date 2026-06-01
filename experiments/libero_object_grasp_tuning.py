from __future__ import annotations

from argparse import Namespace


ALL_LIBERO_OBJECT_TASKS = [str(i) for i in range(10)]


DEFAULT_TUNED_GRASP = {
    "approach_z_offset": 0.03,
    "grasp_z_offset": 0.0,
    "close_steps": 45,
    "lift_steps": 55,
}


OBJECT_TUNED_GRASPS = {
    "cream_cheese_1": {
        "approach_z_offset": 0.015,
        "grasp_z_offset": -0.015,
        "close_steps": 60,
        "lift_steps": 60,
    },
    "chocolate_pudding_1": {
        "approach_z_offset": 0.005,
        "grasp_z_offset": -0.03,
        "close_steps": 70,
        "lift_steps": 60,
    },
    "tomato_sauce_1": DEFAULT_TUNED_GRASP,
    "butter_1": DEFAULT_TUNED_GRASP,
}


def grasp_profile_name(object_name: str, args: Namespace) -> str:
    if not getattr(args, "object_grasp_tuning", True):
        return "cli"
    if object_name in OBJECT_TUNED_GRASPS:
        return f"object_tuned:{object_name}"
    return "object_tuned:default"


def tuned_args_for_object(args: Namespace, object_name: str) -> Namespace:
    """Return a shallow argparse namespace with LIBERO Object grasp overrides.

    The overrides are still hand-scripted controller parameters. They should be
    reported as benchmark-smoke engineering, not learned policy evidence.
    """

    if not getattr(args, "object_grasp_tuning", True):
        return args
    values = vars(args).copy()
    overrides = DEFAULT_TUNED_GRASP.copy()
    overrides.update(OBJECT_TUNED_GRASPS.get(str(object_name), {}))
    values.update(overrides)
    return Namespace(**values)
