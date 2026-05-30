from __future__ import annotations

import importlib.util


class LIBEROUnavailableError(ImportError):
    pass


def is_libero_available() -> tuple[bool, str]:
    if importlib.util.find_spec("libero") is None:
        return False, "libero import not found; local pip install failed while building hf-egl-probe/egl_probe on Windows"
    try:
        from libero.libero import benchmark  # noqa: F401
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

        return True, "LIBERO imports available"
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"LIBERO import failed: {type(exc).__name__}: {exc}"


class LIBEROAdapter:
    task_name = "libero_optional"
    action_dim = 0
    state_dim = 0
    horizon = 0

    def __init__(self, *_, **__):
        ok, reason = is_libero_available()
        if not ok:
            raise LIBEROUnavailableError(reason)
        raise LIBEROUnavailableError("LIBERO dependency is present, but rollout-pool execution is not implemented in this repo state.")
