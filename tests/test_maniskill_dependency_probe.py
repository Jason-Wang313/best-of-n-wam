from __future__ import annotations

import importlib.util
import json
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


def test_discover_external_benchmark_pythons_finds_neighbor_venvs(tmp_path, monkeypatch) -> None:
    module = load_probe_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(module.Path, "home", staticmethod(lambda: home))
    python_path = tmp_path / "external_benchmarks" / ".venvs" / "libero" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")

    result = module.discover_external_benchmark_pythons(repo)

    assert result == [python_path.resolve()]


def test_probe_pinocchio_api_for_python_parses_subprocess_json(tmp_path, monkeypatch) -> None:
    module = load_probe_module()
    python_path = tmp_path / "python.exe"
    python_path.write_text("", encoding="utf-8")
    payload = {
        "pinocchio_import_available": True,
        "pinocchio_api_available": True,
        "pinocchio_module_file": "pinocchio/__init__.py",
        "pinocchio_missing_symbols": [],
        "pinocchio_probe_error": "",
    }

    def fake_run_command(_cmd, _timeout_s):
        return {"returncode": 0, "ok": True, "stdout_tail": json.dumps(payload), "stderr_tail": ""}

    monkeypatch.setattr(module, "run_command", fake_run_command)

    result = module.probe_pinocchio_api_for_python(python_path, timeout_s=1)

    assert result["exists"] is True
    assert result["pinocchio_api_available"] is True
    assert result["pinocchio_missing_symbols"] == []
