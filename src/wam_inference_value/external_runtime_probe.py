from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeCandidate:
    name: str
    python_path: Path | None
    source_path: Path | None
    config_path: Path | None = None


def _as_path(value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _existing(path: Path | None) -> Path | None:
    return path if path is not None and path.exists() else None


def _prepend_pythonpath(env: dict[str, str], source_path: Path | None) -> None:
    if source_path is None:
        return
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(source_path) if not existing else f"{source_path}{os.pathsep}{existing}"


def _run_probe(candidate: RuntimeCandidate, code: str, *, timeout_s: int = 60) -> dict[str, Any]:
    python_path = _existing(candidate.python_path)
    source_path = _existing(candidate.source_path)
    config_path = _existing(candidate.config_path)
    if python_path is None:
        return {
            "name": candidate.name,
            "ok": False,
            "reason": "python_path_missing",
            "python_path": str(candidate.python_path) if candidate.python_path else None,
            "source_path": str(candidate.source_path) if candidate.source_path else None,
            "config_path": str(candidate.config_path) if candidate.config_path else None,
        }

    env = os.environ.copy()
    _prepend_pythonpath(env, source_path)
    if config_path is not None:
        env["LIBERO_CONFIG_PATH"] = str(config_path)
    try:
        completed = subprocess.run(
            [str(python_path), "-c", code],
            cwd=str(source_path) if source_path is not None else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive for missing executables/permissions
        return {
            "name": candidate.name,
            "ok": False,
            "reason": type(exc).__name__,
            "error": str(exc),
            "python_path": str(python_path),
            "source_path": str(source_path) if source_path else None,
            "config_path": str(config_path) if config_path else None,
        }

    stdout = completed.stdout.strip()
    payload: dict[str, Any] = {}
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            data = json.loads(last_line)
            if isinstance(data, dict):
                payload = data
        except json.JSONDecodeError:
            payload = {"stdout_tail": last_line}
    stderr_tail = "\n".join(completed.stderr.splitlines()[-10:])
    return {
        "name": candidate.name,
        "ok": completed.returncode == 0 and bool(payload.get("ok", True)),
        "returncode": completed.returncode,
        "python_path": str(python_path),
        "source_path": str(source_path) if source_path else None,
        "config_path": str(config_path) if config_path else None,
        "stdout_tail": stdout.splitlines()[-10:],
        "stderr_tail": stderr_tail,
        "payload": payload,
    }


def _libero_candidates(root: Path) -> list[RuntimeCandidate]:
    parent = root.parent
    external = parent / "external_benchmarks"
    env_python = _as_path(os.environ.get("LIBERO_PYTHON"))
    env_source = _as_path(os.environ.get("LIBERO_SOURCE_PATH"))
    env_config = _as_path(os.environ.get("LIBERO_CONFIG_PATH"))
    default_source = external / "LIBERO"
    default_config = external / ".libero"
    candidates: list[RuntimeCandidate] = []
    if env_python is not None:
        candidates.append(RuntimeCandidate("env_LIBERO_PYTHON", env_python, env_source or default_source, env_config or default_config))
    candidates.extend(
        [
            RuntimeCandidate("external_libero310", external / ".venvs" / "libero310" / "Scripts" / "python.exe", default_source, default_config),
            RuntimeCandidate("external_libero38", external / ".venvs" / "libero" / "Scripts" / "python.exe", default_source, default_config),
        ]
    )
    return candidates


def _robocasa_candidates(root: Path) -> list[RuntimeCandidate]:
    parent = root.parent
    external = parent / "external_benchmarks"
    env_python = _as_path(os.environ.get("ROBOCASA_PYTHON"))
    env_source = _as_path(os.environ.get("ROBOCASA_SOURCE_PATH"))
    default_source = external / "robocasa"
    candidates: list[RuntimeCandidate] = []
    if env_python is not None:
        candidates.append(RuntimeCandidate("env_ROBOCASA_PYTHON", env_python, env_source or default_source))
    candidates.append(RuntimeCandidate("external_robocasa", external / ".venvs" / "robocasa" / "Scripts" / "python.exe", default_source))
    return candidates


LIBERO_PROBE_CODE = r"""
import json
import sys

import libero.libero as inner
import libero.libero.benchmark as benchmark

print(json.dumps({
    "ok": True,
    "python_version": sys.version,
    "libero_file": getattr(inner, "__file__", None),
    "benchmark_file": getattr(benchmark, "__file__", None),
}))
"""


ROBOCASA_PROBE_CODE = r"""
import json
import sys

import robocasa

print(json.dumps({
    "ok": True,
    "python_version": sys.version,
    "robocasa_file": getattr(robocasa, "__file__", None),
}))
"""


def probe_external_benchmark_runtimes(root: Path) -> dict[str, Any]:
    root = root.resolve()
    libero_attempts = [_run_probe(candidate, LIBERO_PROBE_CODE) for candidate in _libero_candidates(root)]
    robocasa_attempts = [_run_probe(candidate, ROBOCASA_PROBE_CODE) for candidate in _robocasa_candidates(root)]
    libero_success = next((attempt for attempt in libero_attempts if attempt.get("ok")), None)
    robocasa_success = next((attempt for attempt in robocasa_attempts if attempt.get("ok")), None)
    checks = [
        {"name": "libero_probe_attempted", "ok": bool(libero_attempts), "detail": f"attempts={len(libero_attempts)}"},
        {"name": "robocasa_probe_attempted", "ok": bool(robocasa_attempts), "detail": f"attempts={len(robocasa_attempts)}"},
        {"name": "at_least_one_external_runtime_available", "ok": bool(libero_success or robocasa_success), "detail": f"libero={bool(libero_success)}, robocasa={bool(robocasa_success)}"},
    ]
    issues = [check for check in checks if not check["ok"]]
    return {
        "experiment": "external_benchmark_runtime_probe",
        "verified": not issues,
        "libero_import_available": bool(libero_success),
        "robocasa_import_available": bool(robocasa_success),
        "libero_success": libero_success,
        "robocasa_success": robocasa_success,
        "libero_attempts": libero_attempts,
        "robocasa_attempts": robocasa_attempts,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": checks,
        "issues": issues,
        "note": "Runtime import probe only. This does not validate modern VLA policy quality, full RoboCasa coverage, or real-robot evidence.",
    }


def external_runtime_probe_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# External Benchmark Runtime Probe",
        "",
        f"- Verified: `{payload.get('verified')}`",
        f"- LIBERO import available: `{payload.get('libero_import_available')}`",
        f"- RoboCasa import available: `{payload.get('robocasa_import_available')}`",
        f"- Checks: `{payload.get('n_checks')}`",
        f"- Issues: `{payload.get('n_issues')}`",
        "",
        "This is a runtime/import probe only; it does not promote benchmark-performance claims.",
        "",
    ]
    for key in ["libero_success", "robocasa_success"]:
        success = payload.get(key) or {}
        if success:
            lines.append(f"## {key}")
            lines.append("")
            lines.append(f"- Candidate: `{success.get('name')}`")
            lines.append(f"- Python: `{success.get('python_path')}`")
            lines.append(f"- Source: `{success.get('source_path')}`")
            lines.append(f"- Config: `{success.get('config_path')}`")
            lines.append("")
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    return "\n".join(lines).rstrip() + "\n"
