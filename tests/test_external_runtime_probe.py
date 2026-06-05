from __future__ import annotations

import sys
from pathlib import Path

from wam_inference_value.external_runtime_probe import probe_external_benchmark_runtimes


def test_external_runtime_probe_uses_env_paths(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    config = tmp_path / "config"
    inner = source / "libero" / "libero"
    benchmark = inner / "benchmark"
    robocasa = source / "robocasa"
    benchmark.mkdir(parents=True)
    robocasa.mkdir(parents=True)
    config.mkdir()

    (source / "libero" / "__init__.py").write_text("", encoding="utf-8")
    (inner / "__init__.py").write_text("def get_libero_path(key): return key\n", encoding="utf-8")
    (benchmark / "__init__.py").write_text("from libero.libero import get_libero_path\n", encoding="utf-8")
    (robocasa / "__init__.py").write_text("", encoding="utf-8")
    (config / "config.yaml").write_text("assets: .\n", encoding="utf-8")

    monkeypatch.setenv("LIBERO_PYTHON", sys.executable)
    monkeypatch.setenv("LIBERO_SOURCE_PATH", str(source))
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config))
    monkeypatch.setenv("ROBOCASA_PYTHON", sys.executable)
    monkeypatch.setenv("ROBOCASA_SOURCE_PATH", str(source))

    payload = probe_external_benchmark_runtimes(tmp_path / "repo")

    assert payload["verified"] is True
    assert payload["libero_import_available"] is True
    assert payload["robocasa_import_available"] is True
    assert payload["vla_libero_joint_runtime_available"] is False
    assert payload["vla_runtime_attempts"]
    assert payload["libero_success"]["name"] == "env_LIBERO_PYTHON"
    assert payload["robocasa_success"]["name"] == "env_ROBOCASA_PYTHON"
    assert "Runtime import probe only" in payload["note"]
    assert "pretrained VLA weights" in payload["note"]
