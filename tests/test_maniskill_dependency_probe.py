from __future__ import annotations

import importlib.util
import types
from pathlib import Path


def load_probe_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "benchmark_maniskill_dependency_probe.py"
    spec = importlib.util.spec_from_file_location("benchmark_maniskill_dependency_probe_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_pinocchio_api_rejects_unrelated_package(monkeypatch) -> None:
    module = load_probe_module()
    fake_spec = types.SimpleNamespace(origin="fake-pinocchio/__init__.py")
    fake_module = types.SimpleNamespace(__file__="fake-pinocchio/__init__.py")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: fake_spec if name == "pinocchio" else None)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_module)

    result = module.probe_pinocchio_api()

    assert result["pinocchio_import_available"] is True
    assert result["pinocchio_api_available"] is False
    assert result["pinocchio_missing_symbols"] == list(module.PINOCCHIO_REQUIRED_SYMBOLS)


def test_probe_pinocchio_api_accepts_robotics_symbols(monkeypatch) -> None:
    module = load_probe_module()
    fake_spec = types.SimpleNamespace(origin="pinocchio/__init__.py")
    fake_module = types.SimpleNamespace(
        __file__="pinocchio/__init__.py",
        Model=object,
        GeometryModel=object,
        buildModelFromUrdf=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: fake_spec if name == "pinocchio" else None)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_module)

    result = module.probe_pinocchio_api()

    assert result["pinocchio_import_available"] is True
    assert result["pinocchio_api_available"] is True
    assert result["pinocchio_missing_symbols"] == []
