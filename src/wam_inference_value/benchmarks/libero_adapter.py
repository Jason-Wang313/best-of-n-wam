from __future__ import annotations


class LIBEROUnavailableError(ImportError):
    pass


class LIBEROAdapter:
    task_name = "libero_optional"
    action_dim = 0
    state_dim = 0
    horizon = 0

    def __init__(self, *_, **__):
        raise LIBEROUnavailableError("LIBERO is optional and is not integrated in this repo state.")
