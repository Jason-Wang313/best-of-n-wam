from __future__ import annotations


class RoboCasaUnavailableError(ImportError):
    pass


class RoboCasaAdapter:
    task_name = "robocasa_optional"
    action_dim = 0
    state_dim = 0
    horizon = 0

    def __init__(self, *_, **__):
        raise RoboCasaUnavailableError("RoboCasa is optional and is not integrated in this repo state.")
