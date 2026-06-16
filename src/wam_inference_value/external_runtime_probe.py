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


VLA_RUNTIME_PROBE_CODE = r"""
import importlib.util
import json
import sys

modules = {
    "libero": "libero.libero",
    "lerobot_smolvla": "lerobot.policies.smolvla",
    "torch": "torch",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
}

def safe_find_spec(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

available = {key: safe_find_spec(name) for key, name in modules.items()}
print(json.dumps({
    "ok": True,
    "python_version": sys.version,
    "available": available,
    "libero_smolvla_runtime_ready": bool(
        available["libero"]
        and available["lerobot_smolvla"]
        and available["torch"]
        and available["transformers"]
        and available["huggingface_hub"]
    ),
}))
"""


def _vla_runtime_candidates(root: Path) -> list[RuntimeCandidate]:
    seen: set[tuple[str, str | None, str | None]] = set()
    candidates: list[RuntimeCandidate] = []
    external = root.parent / "external_benchmarks"
    libero_source = _as_path(os.environ.get("LIBERO_SOURCE_PATH")) or root.parent / "external_benchmarks" / "LIBERO"
    libero_config = _as_path(os.environ.get("LIBERO_CONFIG_PATH")) or root.parent / "external_benchmarks" / ".libero"
    env_vla_python = _as_path(os.environ.get("VLA_PYTHON") or os.environ.get("SMOLVLA_PYTHON"))
    if env_vla_python is not None:
        candidates.append(RuntimeCandidate("env_VLA_PYTHON_with_libero_source", env_vla_python, libero_source, libero_config))
    # Keep VLA readiness explicit: generic LIBERO/Robocasa envs can be stubs in tests.
    candidates.extend(
        [
            RuntimeCandidate(
                "external_smolvla_with_libero_source",
                external / ".venvs" / "smolvla" / "Scripts" / "python.exe",
                libero_source,
                libero_config,
            ),
            RuntimeCandidate(
                "external_vla_with_libero_source",
                external / ".venvs" / "vla" / "Scripts" / "python.exe",
                libero_source,
                libero_config,
            ),
            RuntimeCandidate(
                "external_robocasa_with_libero_source",
                external / ".venvs" / "robocasa" / "Scripts" / "python.exe",
                libero_source,
                libero_config,
            ),
        ]
    )
    unique_candidates: list[RuntimeCandidate] = []
    for candidate in candidates:
        key = (
            str(candidate.python_path) if candidate.python_path is not None else None,
            str(candidate.source_path) if candidate.source_path is not None else None,
            str(candidate.config_path) if candidate.config_path is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def probe_external_benchmark_runtimes(root: Path) -> dict[str, Any]:
    root = root.resolve()
    libero_attempts = [_run_probe(candidate, LIBERO_PROBE_CODE) for candidate in _libero_candidates(root)]
    robocasa_attempts = [_run_probe(candidate, ROBOCASA_PROBE_CODE) for candidate in _robocasa_candidates(root)]
    vla_runtime_attempts = [_run_probe(candidate, VLA_RUNTIME_PROBE_CODE) for candidate in _vla_runtime_candidates(root)]
    libero_success = next((attempt for attempt in libero_attempts if attempt.get("ok")), None)
    robocasa_success = next((attempt for attempt in robocasa_attempts if attempt.get("ok")), None)
    vla_runtime_success = next(
        (
            attempt
            for attempt in vla_runtime_attempts
            if attempt.get("ok") and (attempt.get("payload") or {}).get("libero_smolvla_runtime_ready") is True
        ),
        None,
    )
    checks = [
        {"name": "libero_probe_attempted", "ok": bool(libero_attempts), "detail": f"attempts={len(libero_attempts)}"},
        {"name": "robocasa_probe_attempted", "ok": bool(robocasa_attempts), "detail": f"attempts={len(robocasa_attempts)}"},
        {
            "name": "vla_runtime_probe_attempted",
            "ok": bool(vla_runtime_attempts),
            "detail": f"attempts={len(vla_runtime_attempts)}, joint_ready={bool(vla_runtime_success)}",
        },
        {"name": "at_least_one_external_runtime_available", "ok": bool(libero_success or robocasa_success), "detail": f"libero={bool(libero_success)}, robocasa={bool(robocasa_success)}"},
    ]
    issues = [check for check in checks if not check["ok"]]
    return {
        "experiment": "external_benchmark_runtime_probe",
        "verified": not issues,
        "libero_import_available": bool(libero_success),
        "robocasa_import_available": bool(robocasa_success),
        "vla_libero_joint_runtime_available": bool(vla_runtime_success),
        "vla_runtime_success": vla_runtime_success,
        "libero_success": libero_success,
        "robocasa_success": robocasa_success,
        "libero_attempts": libero_attempts,
        "robocasa_attempts": robocasa_attempts,
        "vla_runtime_attempts": vla_runtime_attempts,
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": checks,
        "issues": issues,
        "note": "Runtime import probe only. This does not load pretrained VLA weights or validate modern VLA policy quality, full RoboCasa coverage, or real-robot evidence.",
    }


def external_runtime_probe_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# External Benchmark Runtime Probe",
        "",
        f"- Verified: `{payload.get('verified')}`",
        f"- LIBERO import available: `{payload.get('libero_import_available')}`",
        f"- RoboCasa import available: `{payload.get('robocasa_import_available')}`",
        f"- joint LIBERO+SmolVLA runtime available: `{payload.get('vla_libero_joint_runtime_available')}`",
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
    vla_attempts = payload.get("vla_runtime_attempts") or []
    if vla_attempts:
        lines.append("## VLA Runtime Compatibility")
        lines.append("")
        for attempt in vla_attempts:
            available = ((attempt.get("payload") or {}).get("available") or {}) if isinstance(attempt, dict) else {}
            lines.append(
                f"- `{attempt.get('name')}`: ok=`{attempt.get('ok')}`, "
                f"joint_ready=`{(attempt.get('payload') or {}).get('libero_smolvla_runtime_ready')}`, "
                f"available={available}"
            )
        lines.append("")
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    return "\n".join(lines).rstrip() + "\n"
