from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wam_inference_value.evaluation import ensure_result_dirs, results_dir, write_json


PINOCCHIO_REQUIRED_SYMBOLS = ("Model", "GeometryModel", "buildModelFromUrdf")


def _pinocchio_api_probe_code(target_dir: Path | None = None) -> str:
    sys_path_line = f"sys.path.insert(0, {str(target_dir)!r})" if target_dir is not None else ""
    return f"""
import importlib
import json
import sys

{sys_path_line}
required = {list(PINOCCHIO_REQUIRED_SYMBOLS)!r}
try:
    module = importlib.import_module("pinocchio")
    missing = [name for name in required if not hasattr(module, name)]
    payload = {{
        "pinocchio_import_available": True,
        "pinocchio_api_available": not missing,
        "pinocchio_module_file": getattr(module, "__file__", None),
        "pinocchio_missing_symbols": missing,
        "pinocchio_probe_error": "",
    }}
except Exception as exc:
    payload = {{
        "pinocchio_import_available": False,
        "pinocchio_api_available": False,
        "pinocchio_module_file": None,
        "pinocchio_missing_symbols": required,
        "pinocchio_probe_error": f"{{type(exc).__name__}}: {{exc}}",
    }}
print(json.dumps(payload))
"""


def probe_pinocchio_api() -> dict[str, Any]:
    spec = importlib.util.find_spec("pinocchio")
    if spec is None:
        return {
            "pinocchio_import_available": False,
            "pinocchio_api_available": False,
            "pinocchio_module_file": None,
            "pinocchio_missing_symbols": list(PINOCCHIO_REQUIRED_SYMBOLS),
            "pinocchio_probe_error": "module spec not found",
        }
    try:
        module = importlib.import_module("pinocchio")
    except Exception as exc:  # pragma: no cover - optional dependency path.
        return {
            "pinocchio_import_available": False,
            "pinocchio_api_available": False,
            "pinocchio_module_file": getattr(spec, "origin", None),
            "pinocchio_missing_symbols": list(PINOCCHIO_REQUIRED_SYMBOLS),
            "pinocchio_probe_error": f"{type(exc).__name__}: {exc}",
        }
    missing = [name for name in PINOCCHIO_REQUIRED_SYMBOLS if not hasattr(module, name)]
    return {
        "pinocchio_import_available": True,
        "pinocchio_api_available": not missing,
        "pinocchio_module_file": getattr(module, "__file__", None),
        "pinocchio_missing_symbols": missing,
        "pinocchio_probe_error": "",
    }


def run_command(cmd: list[str], timeout_s: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=int(timeout_s))
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-80:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-120:]),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "returncode": None,
            "ok": False,
            "error_type": "TimeoutExpired",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }


def _pinocchio_api_probe_command(target_dir: Path) -> list[str]:
    return [sys.executable, "-c", _pinocchio_api_probe_code(target_dir)]


def _json_from_stdout_tail(command_result: dict[str, Any]) -> dict[str, Any]:
    for line in reversed(str(command_result.get("stdout_tail") or "").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else {}
    return {}


def discover_external_benchmark_pythons(root: Path) -> list[Path]:
    roots = [
        root.resolve().parent / "external_benchmarks" / ".venvs",
        Path.home() / "external_benchmarks" / ".venvs",
    ]
    seen_roots: set[Path] = set()
    seen_pythons: set[Path] = set()
    pythons: list[Path] = []
    for venv_root in roots:
        resolved_root = venv_root.resolve()
        if resolved_root in seen_roots or not resolved_root.exists():
            continue
        seen_roots.add(resolved_root)
        for env_dir in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
            for rel_path in (("Scripts", "python.exe"), ("bin", "python")):
                python_path = (env_dir / Path(*rel_path)).resolve()
                if python_path.exists() and python_path.is_file() and python_path not in seen_pythons:
                    seen_pythons.add(python_path)
                    pythons.append(python_path)
    return pythons


def probe_pinocchio_api_for_python(python_path: Path, timeout_s: int) -> dict[str, Any]:
    python_path = python_path.resolve()
    if not python_path.exists():
        return {
            "python": str(python_path),
            "exists": False,
            "returncode": None,
            "pinocchio_import_available": False,
            "pinocchio_api_available": False,
            "pinocchio_module_file": None,
            "pinocchio_missing_symbols": list(PINOCCHIO_REQUIRED_SYMBOLS),
            "pinocchio_probe_error": "python executable not found",
        }
    command_result = run_command([str(python_path), "-c", _pinocchio_api_probe_code()], timeout_s)
    payload = _json_from_stdout_tail(command_result)
    return {
        "python": str(python_path),
        "exists": True,
        "returncode": command_result.get("returncode"),
        "pinocchio_import_available": bool(payload.get("pinocchio_import_available")),
        "pinocchio_api_available": bool(payload.get("pinocchio_api_available")),
        "pinocchio_module_file": payload.get("pinocchio_module_file"),
        "pinocchio_missing_symbols": payload.get("pinocchio_missing_symbols", list(PINOCCHIO_REQUIRED_SYMBOLS)),
        "pinocchio_probe_error": payload.get("pinocchio_probe_error", command_result.get("stderr_tail", "")),
    }


def probe_pypi_pinocchio_api(timeout_s: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_dir = results_dir() / "tmp_pin_probe" / "pypi_pinocchio_target"
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    install_result = run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "pinocchio",
            "--only-binary=:all:",
            "--no-deps",
            "--target",
            str(target_dir),
            "-q",
        ],
        timeout_s,
    )
    probe_result = (
        run_command(_pinocchio_api_probe_command(target_dir), timeout_s)
        if install_result.get("ok")
        else {
            "command": _pinocchio_api_probe_command(target_dir),
            "returncode": None,
            "ok": False,
            "stdout_tail": "",
            "stderr_tail": "target install failed; API probe skipped",
        }
    )
    api_payload = _json_from_stdout_tail(probe_result)
    shutil.rmtree(target_dir, ignore_errors=True)
    return [install_result, probe_result], api_payload


def write_report(summary: dict[str, Any]) -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "maniskill_dependency_blocker_report.md"
    lines = [
        "# ManiSkill Dependency Probe",
        "",
        "This report is generated by `experiments/benchmark_maniskill_dependency_probe.py`.",
        "",
        f"- attempted: `{summary.get('attempted')}`",
        f"- Python: `{summary.get('python')}`",
        f"- platform: `{summary.get('platform')}`",
        f"- Pinocchio import available: `{summary.get('pinocchio_import_available')}`",
        f"- Pinocchio robotics API available: `{summary.get('pinocchio_api_available')}`",
        f"- Pinocchio module file: `{summary.get('pinocchio_module_file')}`",
        f"- missing Pinocchio API symbols: `{summary.get('pinocchio_missing_symbols')}`",
        f"- `pin` import available: `{summary.get('pin_import_available')}`",
        f"- binary `pin` wheel available through pip: `{summary.get('pin_binary_wheel_available')}`",
        f"- binary PyPI `pinocchio` wheel available: `{summary.get('pypi_pinocchio_binary_wheel_available')}`",
        f"- PyPI `pinocchio` robotics API available after target install: `{summary.get('pypi_pinocchio_api_available')}`",
        f"- PyPI `pinocchio` missing API symbols: `{summary.get('pypi_pinocchio_missing_symbols')}`",
        f"- binary `cmeel-boost` wheel available through pip: `{summary.get('cmeel_boost_binary_wheel_available')}`",
        f"- external benchmark env Python count: `{summary.get('external_env_python_count')}`",
        f"- any external benchmark env has robotics Pinocchio API: `{summary.get('external_env_pinocchio_api_any_available')}`",
        "",
        "## Existing External Benchmark Env Pinocchio Scan",
        "",
        "",
    ]
    probes = summary.get("external_env_pinocchio_probes") or []
    if probes:
        for probe in probes:
            lines.append(f"- `{probe.get('python')}`")
            lines.append(f"  - import available: `{probe.get('pinocchio_import_available')}`")
            lines.append(f"  - robotics API available: `{probe.get('pinocchio_api_available')}`")
            lines.append(f"  - missing symbols: `{probe.get('pinocchio_missing_symbols')}`")
            if probe.get("pinocchio_probe_error"):
                lines.append(f"  - probe error: `{probe.get('pinocchio_probe_error')}`")
    else:
        lines.append("No external benchmark virtualenv Python executables were found under `../external_benchmarks/.venvs` or `~/external_benchmarks/.venvs`.")
    lines.extend(["", "## Command Results", ""])
    for item in summary.get("commands", []):
        lines.append(f"### `{' '.join(item.get('command') or [])}`")
        lines.append("")
        lines.append(f"- returncode: `{item.get('returncode')}`")
        lines.append(f"- ok: `{item.get('ok')}`")
        if item.get("error_type"):
            lines.append(f"- error_type: `{item.get('error_type')}`")
        if item.get("stdout_tail"):
            lines.append("")
            lines.append("stdout tail:")
            lines.append("```text")
            lines.append(str(item.get("stdout_tail")))
            lines.append("```")
        if item.get("stderr_tail"):
            lines.append("")
            lines.append("stderr tail:")
            lines.append("```text")
            lines.append(str(item.get("stderr_tail")))
            lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "ManiSkill state-mode joint-delta control remains artifact-backed. End-effector control is not claimed in this Windows environment because the robotics Pinocchio API is not available and pip did not expose binary `pin`/`cmeel-boost` wheels for this interpreter. If source-install attempts are enabled, their command tails are included above.",
            "",
            "The probe deliberately target-installs and imports the small PyPI package named `pinocchio`; it is not sufficient for ManiSkill/Sapien end-effector-control evidence unless the required robotics API symbols are present.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_result_dirs()
    commands = [
        run_command([sys.executable, "-m", "pip", "index", "versions", "pin"], args.quick_timeout_s),
        run_command([sys.executable, "-m", "pip", "download", "pinocchio", "--only-binary=:all:", "--no-deps", "-d", str(results_dir() / "tmp_pin_probe")], args.quick_timeout_s),
        run_command([sys.executable, "-m", "pip", "download", "pin", "--only-binary=:all:", "--no-deps", "-d", str(results_dir() / "tmp_pin_probe")], args.quick_timeout_s),
        run_command([sys.executable, "-m", "pip", "download", "cmeel-boost", "--only-binary=:all:", "--no-deps", "-d", str(results_dir() / "tmp_pin_probe")], args.quick_timeout_s),
    ]
    pypi_api_commands, pypi_api_payload = probe_pypi_pinocchio_api(args.quick_timeout_s)
    commands.extend(pypi_api_commands)
    if args.attempt_source_install:
        for version in args.source_versions:
            commands.append(run_command([sys.executable, "-m", "pip", "install", f"pin=={version}", "-q"], args.source_timeout_s))

    pypi_pinocchio_download = next((c for c in commands if "download" in c.get("command", []) and "pinocchio" in c.get("command", [])), {})
    pin_download = next((c for c in commands if "download" in c.get("command", []) and "pin" in c.get("command", [])), {})
    boost_download = next((c for c in commands if "download" in c.get("command", []) and "cmeel-boost" in c.get("command", [])), {})
    pinocchio_probe = probe_pinocchio_api()
    external_env_pythons = discover_external_benchmark_pythons(ROOT)
    external_env_probes = [probe_pinocchio_api_for_python(path, args.quick_timeout_s) for path in external_env_pythons]
    external_api_pythons = [probe["python"] for probe in external_env_probes if probe.get("pinocchio_api_available")]
    summary = {
        "experiment": "benchmark_maniskill_dependency_probe",
        "attempted": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **pinocchio_probe,
        "pin_import_available": importlib.util.find_spec("pin") is not None,
        "pin_binary_wheel_available": bool(pin_download.get("ok")),
        "pypi_pinocchio_binary_wheel_available": bool(pypi_pinocchio_download.get("ok")),
        "pypi_pinocchio_api_available": bool(pypi_api_payload.get("pinocchio_api_available")),
        "pypi_pinocchio_module_file": pypi_api_payload.get("pinocchio_module_file"),
        "pypi_pinocchio_missing_symbols": pypi_api_payload.get("pinocchio_missing_symbols", list(PINOCCHIO_REQUIRED_SYMBOLS)),
        "pypi_pinocchio_probe_error": pypi_api_payload.get("pinocchio_probe_error", ""),
        "cmeel_boost_binary_wheel_available": bool(boost_download.get("ok")),
        "external_env_python_count": len(external_env_probes),
        "external_env_pinocchio_probes": external_env_probes,
        "external_env_pinocchio_api_any_available": bool(external_api_pythons),
        "external_env_pinocchio_api_available_pythons": external_api_pythons,
        "source_install_attempted": bool(args.attempt_source_install),
        "commands": commands,
    }
    shutil.rmtree(results_dir() / "tmp_pin_probe", ignore_errors=True)
    report_path = write_report(summary)
    summary["artifacts"] = {"report": str(report_path)}
    write_json(results_dir() / "benchmark_maniskill_dependency_probe.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-source-install", action="store_true")
    parser.add_argument("--source-versions", nargs="*", default=["4.0.0", "2.7.0"])
    parser.add_argument("--quick-timeout-s", type=int, default=120)
    parser.add_argument("--source-timeout-s", type=int, default=900)
    args = parser.parse_args()
    summary = run(args)
    print(
        "ManiSkill dependency probe complete: "
        f"pinocchio={summary['pinocchio_import_available']}, "
        f"pinocchio_api={summary['pinocchio_api_available']}, "
        f"pin_binary={summary['pin_binary_wheel_available']}, "
        f"boost_binary={summary['cmeel_boost_binary_wheel_available']}, "
        f"external_env_pinocchio_api={summary['external_env_pinocchio_api_any_available']}, "
        f"source_attempted={summary['source_install_attempted']}"
    )


if __name__ == "__main__":
    main()
